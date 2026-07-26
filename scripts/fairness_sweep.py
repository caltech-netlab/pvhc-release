"""
fairness_sweep.py — Paper Section IV.C.3, Eq. (16), Fig. 5 (b.1-b.3)

Gamma ablation of the fairness-augmented PVHC objective (Eq. 16):

    max  sum(zeta_c_bar) + gamma * sum(log(zeta_c_bar + eta))   [Eq. (16)]

where gamma in {1, 10, 100, 1000, 10000} governs the PVHC-vs-fairness
trade-off; eta is fixed at 1e-5 (a numerical floor, negligible effect).

Outputs (in temp/fairness_sweep/<feeder_name>/):
  1. sweep_results.csv     – gamma, PVHC, Jain index for each gamma
  2. tradeoff_frontier.png – PVHC vs Jain trade-off curve, knee annotated
"""

import csv
import pathlib
import sys
import time
from tqdm import tqdm

import numpy as np
import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

file = pathlib.Path(__file__).resolve()
project_root = file.parents[1]
sys.path.append(str(project_root))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, DEFAULT_CANDIDATE_PV_CAPACITY_KW
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery, clarkson_solver
from src.utils import update_num_scenarios, load_capacity_limits

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
FEEDER_NAME = DEFAULT_FEEDER_NAME
SEED = 42
# Number of scenarios to sample; paper used 1404 (final Clarkson iteration).
# Reduce to 200-300 for faster runs; increase for tighter risk guarantees.
NUM_SCENARIOS = 200
# Solvers for the log-fairness (exponential-cone) objective, tried in order --
# CLARABEL/ECOS can occasionally fail on this problem's harder instances now
# that real (non-inflated) network limits make some scenarios numerically
# tight. GUROBI/MOSEK are listed last as an optional, licensed fallback --
# harmless to list even if not installed (fails immediately, moves on).
SOLVERS = [cp.CLARABEL, cp.ECOS, cp.GUROBI, cp.MOSEK]
SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"

# Gamma ablation values
GAMMA_VALUES = [1, 10, 100, 1000, 10000]
ETA = 1e-5  # fixed log floor; negligible effect on results

# Wait-and-judge parameters (Algorithm 1, Theorem III.2)
EPSILON = 0.01  # max violation probability
BETA = 0.05  # confidence failure probability
N_INIT = 50  # initial scenario count
MAX_OUTER_ITER = 20  # safety cap on outer iterations

OUTPUT_DIR = project_root / "temp" / "fairness_sweep"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def jain_fairness_index(values: np.ndarray) -> float:
    """Jain's fairness index J(x) = (Σx)² / (n·Σx²), range [1/n, 1].

    J=1 means all allocations are identical (perfect fairness); J=1/n means
    a single bus holds all capacity.  Used as the fairness metric in Fig. 5 and
    the sweep output CSV (column 'jain').  Cited as reference [36] in the paper.
    """
    v = np.asarray(values, dtype=float)
    denom = len(v) * np.sum(v**2)
    return float(np.sum(v) ** 2 / denom) if denom > 1e-12 else 0.0


def find_knee(fairness: np.ndarray, pvhc: np.ndarray) -> int:
    """
    Knee of the (fairness, PVHC) trade-off curve via maximum perpendicular
    distance from the chord joining the two curve endpoints.
    """
    p0 = np.array([fairness[0], pvhc[0]])
    p1 = np.array([fairness[-1], pvhc[-1]])
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-10:
        return 0
    dists = [
        abs(np.cross(chord, p0 - np.array([f, h]))) / chord_len
        for f, h in zip(fairness, pvhc)
    ]
    return int(np.argmax(dists))


# ---------------------------------------------------------------------------
# Core: solve LP for one (gamma, eta) with pre-built constraints
# ---------------------------------------------------------------------------


def solve_one(
    problem: PVHCProblemWithoutBattery,
    scenario_indices: list,
    gamma: float,
    eta: float,
    constraints_fixed: list,
    x_var: cp.Variable,
) -> tuple:
    """
    Solve the fairness-augmented LP for a single (gamma, eta) pair.

    Args:
        problem:           compiled PVHCProblemWithoutBattery instance
        scenario_indices:  list of scenario indices to include
        gamma:             fairness weight (0 = pure PVHC maximisation)
        eta:               log floor (numerical stabiliser only)
        constraints_fixed: pre-built list of CVXPY constraint objects
                           (capacity bound + all scenario network constraints)
        x_var:             CVXPY variable shared across all solves

    Returns:
        (total_pvhc_kw, jain_index_of_customer_caps, x_optimal_or_None)
    """
    # Objective
    if gamma == 0:
        # gamma=0: baseline LP, no fairness regularisation
        objective = cp.Maximize(cp.sum(x_var))
    else:
        # eta is a constant float baked into the expression tree; because eta
        # only appears inside the log (not as a matrix coefficient) a separate
        # problem object is needed for each distinct eta value.
        fairness_term = gamma * cp.sum(cp.log(x_var + eta))
        objective = cp.Maximize(cp.sum(x_var) + fairness_term)

    prob = cp.Problem(objective, constraints_fixed)
    for solver_idx, solver_choice in enumerate(SOLVERS):
        try:
            prob.solve(solver=solver_choice, verbose=False)
            break
        except cp.error.SolverError as e:
            print(f"  [WARN] {solver_choice} failed ({e}); "
                  f"{'trying next solver...' if solver_idx + 1 < len(SOLVERS) else 'no solvers left.'}")

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"  [WARN] solver status={prob.status} for gamma={gamma}, eta={eta:.0e}")
        return np.nan, np.nan, None

    x_val = x_var.value
    n_util = problem.num_utility_pv  # 0 in all-customer setting
    zeta_c = x_val[n_util:]  # customer PV caps

    return float(np.sum(x_val)), jain_fairness_index(zeta_c), x_val


# ---------------------------------------------------------------------------
# Build scenario constraints (network feasibility) once, outside the sweep
# ---------------------------------------------------------------------------


def build_fixed_constraints(
    problem: PVHCProblemWithoutBattery,
    scenario_indices: list,
    x_var: cp.Variable,
) -> list:
    """
    Assemble capacity bound + network constraints for all scenarios.
    These constraints are identical for all (gamma, eta) pairs, so they
    are built once and reused across the sweep.
    """
    constraints = [x_var <= problem.capacity_limits]
    for t in tqdm(scenario_indices, desc="  Scenarios", unit="scen", ncols=80):
        constraints.extend(problem._build_scenario_constraints_cvxpy(x_var, t))
    return constraints


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _gamma_tick_label(g: float) -> str:
    return "0" if g == 0 else (f"{int(g)}" if g == int(g) else f"{g:.0e}")


def plot_tradeoff_frontier(gamma_vals, pvhc_row, fairness_row, eta_label, out_path):
    """
    2-D trade-off curve: PVHC vs Jain's fairness at fixed eta.
    The knee (maximum curvature point) is annotated.
    """
    # Drop NaN entries
    mask = ~(np.isnan(pvhc_row) | np.isnan(fairness_row))
    g_valid = [gamma_vals[i] for i in range(len(gamma_vals)) if mask[i]]
    pvhc_v = pvhc_row[mask]
    fair_v = fairness_row[mask]

    # Sort by increasing fairness index for a clean curve
    order = np.argsort(fair_v)
    fair_sorted = fair_v[order]
    pvhc_sorted = pvhc_v[order]
    g_sorted = [g_valid[i] for i in order]

    knee_idx = find_knee(fair_sorted, pvhc_sorted)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        fair_sorted, pvhc_sorted, "o-", color="#1f77b4", linewidth=1.8, markersize=6
    )

    # Annotate each point with its gamma value
    for i, (f, h, g) in enumerate(zip(fair_sorted, pvhc_sorted, g_sorted)):
        label = r"$\gamma$=" + _gamma_tick_label(g)
        va = "top" if i == knee_idx else "bottom"
        ax.annotate(
            label,
            xy=(f, h),
            xytext=(4, -6 if va == "top" else 4),
            textcoords="offset points",
            fontsize=8,
            color="#555555",
        )

    # Knee annotation
    ax.plot(
        fair_sorted[knee_idx],
        pvhc_sorted[knee_idx],
        "*",
        color="red",
        markersize=14,
        zorder=5,
        label=f"Knee ($\\gamma$={_gamma_tick_label(g_sorted[knee_idx])})",
    )

    ax.set_xlabel("Jain's Fairness Index", fontsize=12)
    ax.set_ylabel("Total PVHC (kW)", fontsize=12)
    ax.set_title(f"PVHC–Fairness Trade-off Frontier (η = {eta_label})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main():
    np.random.seed(SEED)

    feeder_name = FEEDER_NAME
    cfg = FEEDERS[feeder_name]
    feeder_root = cfg["feeder_root"]
    v_root_nom = cfg["v_root_nom"]
    v_root_mult = cfg["v_root_multiplier"]

    out_dir = OUTPUT_DIR / feeder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFairness sweep - feeder: {feeder_name}")
    print(f"Output directory : {out_dir}")
    print(f"Gamma values     : {GAMMA_VALUES}")
    print(f"Eta (fixed)      : {ETA}")
    print(f"epsilon={EPSILON}, beta={BETA}, N_init={N_INIT}, max_outer={MAX_OUTER_ITER}")

    # ------------------------------------------------------------------
    # 1. Network setup (once)
    # ------------------------------------------------------------------
    print("\n[1/3] Setting up network ...")
    data_loader = DataLoader(feeder_name)
    data_loader._dssParser()
    data_loader.get_circuit_model(feeder_root=feeder_root)
    simulator = Simulator(data_loader)

    # Get structural load_info (same every call; just need bus topology)
    _, _, _, load_info, _ = data_loader.get_scenario_data_by_sampling(
        SOLAR_PROFILE_PATH,
        N=N_INIT,
        v_root_nom=v_root_nom,
        v_root_multiplier=v_root_mult,
        seed=SEED,
    )
    load_info_with_limits = load_capacity_limits(feeder_name)
    if load_info_with_limits is not None:
        print("  Using extracted per-customer capacity limits (Eq. 12).")
        PV_info = load_info_with_limits
    else:
        print(f"  No extracted capacity limits cached; falling back to uniform "
              f"{DEFAULT_CANDIDATE_PV_CAPACITY_KW} kW default (run "
              "calculate_capacity_limit_and_store.py for realistic limits).")
        PV_info = [{**load, "capacity": DEFAULT_CANDIDATE_PV_CAPACITY_KW} for load in load_info]
    utility_pv = []  # all-customer, matching paper Table I
    customer_pv = PV_info
    print(f"  Customer PV sites: {len(customer_pv)}")

    # ------------------------------------------------------------------
    # 2. Gamma ablation with wait-and-judge (Algorithm 1)
    # ------------------------------------------------------------------
    print("\n[2/3] Running gamma ablation (wait-and-judge per gamma) ...")
    n_gamma = len(GAMMA_VALUES)
    csv_path_tmp = out_dir / "sweep_results_partial.csv"
    pvhc_arr = np.full(n_gamma, np.nan)
    fairness_arr = np.full(n_gamma, np.nan)
    conv_N_arr = np.full(n_gamma, np.nan)  # N at convergence

    t_start = time.time()
    for i, gamma in enumerate(GAMMA_VALUES):
        print(f"\n{'='*60}", flush=True)
        print(f"  gamma = {_gamma_tick_label(gamma)}  ({i+1}/{n_gamma})", flush=True)
        print(f"{'='*60}", flush=True)

        N_samples = N_INIT
        outer_iter = 0
        final_result = None

        while outer_iter < MAX_OUTER_ITER:
            current_seed = SEED + outer_iter * 1000
            print(
                f"\n  [outer {outer_iter+1}] N={N_samples}, seed={current_seed}",
                flush=True,
            )

            # Sample N_samples scenarios
            _, pv_profiles, load_profiles, load_info_iter, V0_samples = (
                data_loader.get_scenario_data_by_sampling(
                    SOLAR_PROFILE_PATH,
                    N=N_samples,
                    v_root_nom=v_root_nom,
                    v_root_multiplier=v_root_mult,
                    seed=current_seed,
                )
            )

            # Build and compile problem for this (gamma, N) pair
            problem = PVHCProblemWithoutBattery(
                simulator=simulator,
                utility_pv_info=utility_pv,
                customer_pv_info=customer_pv,
                load_info=load_info_iter,
                is_robust=True,
                with_fairness=True,
                fairness_factor=float(gamma),
                load_cached_limits=True,
            )
            problem.add_scenarios(
                pv_profiles=pv_profiles,
                load_profiles=load_profiles,
                V0_profiles=V0_samples,
                filter_zero_pv=True,
            )
            n_scen = problem.num_scenarios
            print(f"  Scenarios after filtering: {n_scen}", flush=True)
            problem.compile()

            # Clarkson solver
            print("  Running Clarkson ...", end=" ", flush=True)
            t0 = time.time()
            clarkson_result = clarkson_solver(
                problem=problem,
                is_scenario_based=True,
                initial_sample_size=25,
                max_iterations=10,
                solver=SOLVERS,
                verbose=False,
                seed=current_seed,
            )
            k = len(clarkson_result["support_indices"])
            dt = time.time() - t0
            print(
                f"k={k}, PVHC={clarkson_result['total_capacity']:.1f} kW"
                f"  ({dt:.1f}s)",
                flush=True,
            )
            if not clarkson_result["converged"]:
                print("  [WARN] Clarkson did not converge", flush=True)

            # Wait-and-judge criterion (Theorem III.2)
            new_N = update_num_scenarios(
                epsilon=EPSILON,
                beta=BETA,
                num_scenarios=N_samples,
                num_active_constraints=k,
                n_I=1,
            )

            if new_N == -1:
                print(
                    f"  Converged at N={N_samples}, k={k}",
                    flush=True,
                )
                final_result = clarkson_result
                conv_N_arr[i] = N_samples
                break
            else:
                print(
                    f"  phi criterion not satisfied -- increasing N: {N_samples} -> {new_N}",
                    flush=True,
                )
                N_samples = new_N
                outer_iter += 1

        if final_result is None:
            print(
                f"  [WARN] Did not converge within {MAX_OUTER_ITER} outer iterations",
                flush=True,
            )
            final_result = clarkson_result  # use last result anyway

        # Extract PVHC and Jain from optimal capacities
        opt_caps = final_result["optimal_capacities"]
        n_util = len(utility_pv)
        pvhc_arr[i] = float(np.sum(opt_caps))
        fairness_arr[i] = jain_fairness_index(opt_caps[n_util:])

        elapsed = time.time() - t_start
        eta_remaining = (elapsed / (i + 1)) * (n_gamma - i - 1)
        print(
            f"\n  >> gamma={_gamma_tick_label(gamma)}: "
            f"PVHC={pvhc_arr[i]:.1f} kW, Jain={fairness_arr[i]:.4f}, "
            f"N*={int(conv_N_arr[i]) if not np.isnan(conv_N_arr[i]) else '?'}"
            f"  [ETA {eta_remaining/60:.1f} min]",
            flush=True,
        )

        # Save partial results
        with open(csv_path_tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gamma", "pvhc_kw", "jain_index", "N_converged"])
            for ii in range(n_gamma):
                if not np.isnan(pvhc_arr[ii]):
                    writer.writerow(
                        [
                            GAMMA_VALUES[ii],
                            pvhc_arr[ii],
                            fairness_arr[ii],
                            conv_N_arr[ii],
                        ]
                    )

    total_time = time.time() - t_start
    print(f"\nSweep completed in {total_time/60:.1f} min", flush=True)

    # ------------------------------------------------------------------
    # 3. Save CSV, plot, summary
    # ------------------------------------------------------------------
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "pvhc_kw", "jain_index", "N_converged"])
        for i, gamma in enumerate(GAMMA_VALUES):
            writer.writerow([gamma, pvhc_arr[i], fairness_arr[i], conv_N_arr[i]])
    print(f"\nSaved: {csv_path}")

    print("\nGenerating plots ...")
    plot_tradeoff_frontier(
        gamma_vals=GAMMA_VALUES,
        pvhc_row=pvhc_arr,
        fairness_row=fairness_arr,
        eta_label=f"{ETA:.0e}",
        out_path=out_dir / "tradeoff_frontier.png",
    )

    print(f"\n=== Summary Table (eta={ETA:.0e}, epsilon={EPSILON}, beta={BETA}) ===")
    print(f"{'gamma':>8s}  {'PVHC (kW)':>12s}  {'Jain Index':>12s}  {'N*':>6s}")
    print("-" * 46)
    for i, gamma in enumerate(GAMMA_VALUES):
        n_star = int(conv_N_arr[i]) if not np.isnan(conv_N_arr[i]) else "?"
        print(
            f"{_gamma_tick_label(gamma):>8s}  "
            f"{pvhc_arr[i]:>12.1f}  "
            f"{fairness_arr[i]:>12.4f}  "
            f"{n_star!s:>6s}"
        )

    print(f"\nAll outputs written to: {out_dir}")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
