"""
Method Comparison — RSP (this paper's method) vs. CVaR baseline vs. Robust (box) baseline

Compares three ways of handling load/PV uncertainty on the *same* circuit model,
same LinDistFlow constraints, and same objective (max total utility-owned PVHC),
so the only thing that differs is the risk operator applied to the identical
per-scenario loss g_k(zeta_u) = A_u(w^k) . zeta_u + b(w^k) <= 0 (row-wise).

CRITICAL SIMPLIFYING ASSUMPTION: every PV site is utility-owned (utility_pv_ratio=1.0).
There is no customer installation uncertainty (no zeta_c, no robust box over zeta_c);
the only decision variable is zeta_u, and A_u(w^k)/b(w^k) are exactly the coefficients
`PVHCProblemWithoutBattery` already assembles (voltage + branch-flow rows, Eq. 7/9 in
src/analyzer.py) — this script does not re-derive the network model, it only reads the
already-compiled `matrices` / `base_case` / `limits` / `pv_profiles_processed` off an
existing `PVHCProblemWithoutBattery` instance and adds new constraint blocks on top.

Methods:
  RSP           — hard scenario constraints g_k(zeta_u) <= 0 for all k, n set by the
                   Campi-Garatti wait-and-judge loop. Called unmodified via
                   run_pvhc_estimation.run_iterative_pvhc_analysis().
  CVaR baseline — element-wise sample-based CVaR per constraint row (Madavan et al.,
                   "Risk-Based Hosting Capacity Analysis in Distribution Systems,"
                   IEEE TPWRS 2024): Rockafellar-Uryasev epigraph (their Eq. 7), the
                   CVaR<=0 chance-constraint surrogate (their Eq. 8), and the sample
                   LP reformulation with one t_r + N slacks u_{k,r} per row (their
                   Eqs. 17-18). Two interchangeable implementations, selected via
                   CVAR_METHOD:
                     "dense"         — solve_cvar_baseline(): all m constraint rows at
                                       once, exact, but the LP scales as T*m (memory-
                                       limited, see scale note below).
                     "row_generation" — solve_cvar_baseline_row_generation(): cutting-
                                       plane scheme that only ever holds the (usually
                                       tiny) ACTIVE row subset in the LP, mirroring the
                                       Clarkson-style scenario generation RSP already
                                       uses (clarkson_solver in src/analyzer.py), just
                                       applied to constraint rows instead of scenarios.
                   Unlike RSP/Robust, CVaR's training scenarios are built WITHOUT
                   zero-PV filtering (build_compiled_problem(..., filter_zero_pv=False)
                   in main()) -- CVaR is an average over the sampled population, so
                   dropping the always-safe nighttime scenarios would shrink that
                   population and distort what "worst (1-delta) fraction" means
                   relative to a full year of hours. RSP/Robust are unaffected by
                   filtering (RSP: a dropped scenario's constraint was vacuous anyway;
                   Robust: worst-case hull, not an average).
  Robust (box)  — worst case over the interval hull of the *assembled* coefficients
                   across N sampled scenarios (simplified from Chen et al., "Data-Driven
                   DG Capacity Assessment Method for ADNs," IEEE TPWRS 2017, whose
                   moment+support ambiguity set gives an SDP/BMI; here we use a plain
                   sampled box so it stays an LP on this network's model). One
                   deterministic constraint per row, no risk level, no slacks. See
                   solve_robust_baseline().

Scale note — CVaR and Robust are capped by *different* things, hence two separate
sample-count variables (N_BASELINE_CVAR, N_BASELINE_ROBUST) instead of one shared N:
  This example feeder has ~1200 buses -> n_elem = 3*1200 = 3600 -> m = 10*n_elem =
  36,000 constraint rows/scenario (voltage + P/Q + linearized |P+-Q|, upper & lower).
  CVaR's *dense* formulation needs one slack u_{k,r} per (sample, row), so its LP scales
  as T*m. Measured empirically on this machine: the voltage sensitivity matrix M_v is
  ~99.7% dense (every bus's voltage depends on the shared root-to-bus path, so this
  isn't sparse like a typical radial-flow matrix — M_p/M_q *are* sparse, ~1.5-3% dense,
  and are wrapped in scipy.sparse below; M_v is left dense on purpose since CSR overhead
  exceeds a plain array's footprint above ~30-50% density). Consequences: T=63 actual
  training scenarios (N_BASELINE_CVAR=150 requested, ~42% survive zero-PV filtering)
  solved dense CVaR in ~3.6 minutes; T~120+ (N=300) hit a MemoryError building MOSEK's
  constraint matrix; N=1288 needs ~20GiB for one intermediate array. N_BASELINE_CVAR=150
  is set with the dense ceiling in mind. Row generation (CVAR_METHOD="row_generation")
  sidesteps this: its LP only ever holds the active row subset (RSP's own converged
  support was just 4 rows out of tens of thousands, so CVaR's binding set is plausibly
  similarly small), so N_BASELINE_CVAR can likely go much higher under that method — the
  active-row count is printed at runtime so this can be judged/adjusted directly.

  Robust has none of this: solve_robust_baseline() never builds a per-(sample,row)
  variable at all — it reduces the N sampled scenarios down to two numbers per
  coefficient (elementwise min/max across scenarios, Appendix B) *before* building the
  LP, so the LP itself is always the same small size (10 vector constraints of length
  n_elem, no scenario-indexed variables) regardless of N. N_BASELINE_ROBUST can
  therefore be set far higher than N_BASELINE_CVAR — e.g. comparable to RSP's converged
  n — without a runtime or memory cost increase worth worrying about; the reduction
  over N scenarios is a handful of cheap numpy max/min calls either way.

Disjointness: CVaR's training draw, Robust's training draw, and the held-out N_TEST
draw are all mutually guaranteed disjoint by construction (non-overlapping slices of
one random permutation of the same valid 2023-2024 scenario pool my sampler uses — see
draw_disjoint_indices).
RSP's own wait-and-judge training draws happen inside the unmodified existing routine,
which reseeds numpy's *global* RNG per outer iteration; this script never touches the
global RNG (only np.random.RandomState objects), so there's no state interference, but
exact index-level exclusion of RSP's internal draws from this script's N_TEST pool is
NOT guaranteed — only seed-separation, matching the existing codebase's own convention
(e.g. test_seed = seed + 999999 in run_iterative_pvhc_analysis). Documented, not silently
assumed away.
"""

import sys
import pathlib
import time
import json
import csv

import numpy as np
import cvxpy as cp
import scipy.sparse as sp

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.data_loader import DataLoader
from src.constants import FEEDERS, DEFAULT_FEEDER_NAME
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery
from src.utils import PVHCTestConfig, load_optimization_results, load_capacity_limits
from scripts.run_pvhc_estimation import run_iterative_pvhc_analysis

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"

np.set_printoptions(precision=3, threshold=10000, linewidth=170)

# ==============================================================================
# TOP-OF-FILE PARAMETERS
# ==============================================================================
CVAR_LEVEL = 0.9  # delta, CVaR risk level (Madavan Eq. 8)
N_TEST = 10000  # held-out scenarios for violation-probability evaluation
SAMPLE_ABLATION = None  # None -> headline run only; or e.g. [20, 50, 100, 150, 200]
N_BASELINE_CVAR = (
    20  # CVaR training sample count — LP-size-limited, see scale note above (do not
)
# raise this casually under CVAR_METHOD="dense": N=300 already hit a MemoryError, N=1288
# needs ~20GiB just for one intermediate matrix; CVaR's dense LP scales as T * 31,350
# constraint rows. Row generation (CVAR_METHOD="row_generation") is not bound by this.
N_BASELINE_ROBUST = (
    100  # Robust training sample count — cheap regardless of N, can be large
)

CVAR_METHOD = "row_generation"  # "dense" (solve_cvar_baseline: all m rows in one LP,
# exact, memory-limited) or "row_generation" (solve_cvar_baseline_row_generation:
# cutting-plane over constraint rows, scales to much larger N_BASELINE_CVAR). See
# solve_cvar() dispatcher and the scale note above for the tradeoff.
CVAR_MAX_ACTIVE_ROWS = 500  # cap on simultaneously-active rows in row_generation; if a
# single iteration finds more violations than this allows, only a weighted sample gets
# added (mirrors clarkson_solver's cap-and-importance-sample behavior in src/analyzer.py)

RANDOM_SEED = 42  # governs the train/test permutation (np.random.RandomState, local)

SKIP_RSP = False  # If True, don't re-run RSP's wait-and-judge loop — load the result
# saved by a previous run of this script (temp/optimization_results/<feeder>/<RSP_SUB_FOLDER>/)
# instead. Only useful after that first run has produced a saved result to load; leave
# False for a fresh checkout, then flip to True to iterate faster on the baselines alone.
RSP_SUB_FOLDER = "method_comparison_rsp"  # where RSP results are stored/loaded from

FEEDER_NAME = DEFAULT_FEEDER_NAME
SOLVER = cp.HIGHS

# RSP (Campi-Garatti wait-and-judge) parameters — match run_pvhc_estimation.py
EPSILON = 0.01
BETA = 0.05
N_I = 1
INITIAL_NUM_SCENARIOS = 50
INITIAL_SAMPLE_SIZE = 10
MAX_SUBPROBLEM_SIZE = 10
MAX_OUTER_ITERATIONS = 20
RSP_INTERNAL_N_TEST = (
    2000  # RSP's own internal validation set (not used for the 3-way comparison)
)


# ==============================================================================
# SCENARIO POOL & DISJOINT SAMPLING
# ==============================================================================


def load_full_scenario_pool(
    data_loader, v_root_nom, v_root_multiplier, dt=60 * 30
):
    """Load the full year's pool and filter to valid (non-zero, non-NaN V0) timestamps.

    Mirrors DataLoader.get_scenario_data_by_sampling's internal pool-building exactly
    (same date range, same dt, same valid mask), so indices drawn here come from the
    identical universe my RSP sampler draws from.
    """
    t0_str, t1_str = "2023-01-01 00:00:00", "2023-12-31 23:59:59"
    time_array, pv_profiles, load_profiles, load_info, V0_profiles = (
        data_loader.get_scenario_data_by_range(
            SOLAR_PROFILE_PATH, t0_str, t1_str, v_root_nom, v_root_multiplier, dt=dt,
        )
    )
    valid_mask = np.all(V0_profiles != 0, axis=1) & ~np.any(
        np.isnan(V0_profiles), axis=1
    )
    valid_indices = np.where(valid_mask)[0]
    return time_array, pv_profiles, load_profiles, load_info, V0_profiles, valid_indices


def draw_disjoint_indices(valid_indices, n_test, train_sizes, seed):
    """Draw n_test test indices plus one training index array per entry in
    train_sizes (e.g. [n_cvar, n_robust]), all mutually disjoint by construction:
    every set is a non-overlapping slice of a single random permutation of
    valid_indices, not just separated by seed.

    Returns (test_idx, [train_idx_for_train_sizes[0], train_idx_for_train_sizes[1], ...]).
    """
    total = n_test + sum(train_sizes)
    if total > len(valid_indices):
        raise ValueError(
            f"Requested n_test={n_test} + train_sizes={train_sizes} (sum="
            f"{sum(train_sizes)}) exceeds valid pool size {len(valid_indices)}."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(valid_indices)
    test_idx = perm[:n_test]
    train_idxs = []
    start = n_test
    for size in train_sizes:
        train_idxs.append(perm[start : start + size])
        start += size
    return test_idx, train_idxs


def slice_scenarios(pv_profiles, load_profiles, V0_profiles, idx):
    return (
        pv_profiles[idx],
        (load_profiles[0][idx], load_profiles[1][idx]),
        V0_profiles[idx],
    )


def build_compiled_problem(
    simulator,
    utility_pv_info,
    load_info,
    pv_profiles,
    load_profiles,
    V0_profiles,
    filter_zero_pv=True,
):
    """Build and compile a utility-owned-only PVHCProblemWithoutBattery instance.

    This is the *existing* coefficient-assembly machinery (network model, LinDistFlow
    matrices, voltage/line limits) — reused verbatim, not re-derived. customer_pv_info=[]
    per the utility-owned-only simplifying assumption.

    filter_zero_pv: RSP/Robust/the test set all drop zero-PV (nighttime) scenarios —
    for RSP that's a pure efficiency win (a zero-PV scenario's constraint is vacuous,
    always satisfied regardless of zeta_u, so dropping it doesn't change RSP's feasible
    region, only saves wasted scenario budget). CVaR is different: it's an *average*
    over the sampled population, so which population you average over changes the
    answer, not just the cost of computing it. Dropping the (always-safe) nighttime
    scenarios shrinks the pool average is taken over, which shifts what "worst
    (1-delta) fraction" means relative to a full year of hours. So CVaR is built with
    filter_zero_pv=False in main() -- pass False here for that call.
    """
    problem = PVHCProblemWithoutBattery(
        simulator=simulator,
        utility_pv_info=utility_pv_info,
        customer_pv_info=[],
        load_info=load_info,
        is_robust=True,
        with_fairness=False,
        load_cached_limits=True,
    )
    problem.add_scenarios(
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        V0_profiles=V0_profiles,
        filter_zero_pv=filter_zero_pv,
    )
    problem.compile()
    return problem


# ==============================================================================
# CVAR BASELINE — Madavan et al. 2024, Eqs. 7 (RU epigraph), 8 (chance-constraint
# surrogate), 17-18 (sample-based LP). Element-wise: one t_r + N slacks u_{k,r} per row.
# ==============================================================================


def solve_cvar_baseline(problem, cvar_level, solver=SOLVER, verbose=False):
    """Dense formulation: all m constraint rows in a single LP. Exact, but memory scales
    as T*m — see the scale note at the top of this file. Prefer
    solve_cvar_baseline_row_generation() for larger N_BASELINE_CVAR.
    """
    n_util = problem.num_utility_pv
    T = problem.num_scenarios
    n_elem = problem.limits["voltage_upper"].shape[0]
    m = 10 * n_elem

    print(
        f"[CVaR/dense] LP size: T={T} scenarios x m={m} rows -> "
        f"{T * m:,} slack vars (u) + {m:,} vars (t) + {n_util} vars (x)"
    )

    # M_v is ~99.7% dense (voltage sensitivity couples every bus via its shared
    # root-path) so it's left as a plain array; M_p/M_q are genuinely sparse
    # (~1.5-3% dense, branch flow only depends on downstream injections) so they're
    # wrapped in scipy.sparse to cut memory/build time.
    M_v = problem.matrices["M_v_utility"]
    M_p = sp.csr_matrix(problem.matrices["M_p_utility"])
    M_q = sp.csr_matrix(problem.matrices["M_q_utility"])

    x = cp.Variable(n_util, nonneg=True)
    # Y[t, :] = pv_profiles_processed[t, :] * x  (per-scenario PV-scaled capacity, Eq. 8 in analyzer.py)
    Y = cp.multiply(
        problem.pv_profiles_processed, cp.reshape(x, (1, n_util), order="C")
    )

    lhs_v = problem.base_case["voltage"] + Y @ M_v.T
    lhs_p = problem.base_case["active_power"] + Y @ M_p.T
    lhs_q = problem.base_case["reactive_power"] + Y @ M_q.T
    lhs_pq = lhs_p + lhs_q
    lhs_pmq = lhs_p - lhs_q

    lim = problem.limits
    # g_{t,r}(x) <= 0 <=> feasible, in the row order matching
    # PVHCProblemWithoutBattery._decode_constraint_index's constraint_types list.
    g_blocks = [
        lhs_v - lim["voltage_upper"],
        lim["voltage_lower"] - lhs_v,
        lhs_p - lim["S_ratings"],
        -lim["S_ratings"] - lhs_p,
        lhs_q - lim["S_ratings"],
        -lim["S_ratings"] - lhs_q,
        lhs_pq - lim["sqrt2_S_ratings"],
        -lim["sqrt2_S_ratings"] - lhs_pq,
        lhs_pmq - lim["sqrt2_S_ratings"],
        -lim["sqrt2_S_ratings"] - lhs_pmq,
    ]
    G = cp.hstack(g_blocks)  # (T, m)

    t = cp.Variable(m)  # VaR-like auxiliary, one per row (Madavan Eq. 7)
    u = cp.Variable((T, m), nonneg=True)  # epigraph slacks (Madavan Eq. 17-18)

    constraints = [
        x <= problem.capacity_limits,
        u >= G - cp.reshape(t, (1, m), order="C"),
        t + cp.sum(u, axis=0) / ((1 - cvar_level) * T) <= 0,
    ]
    objective = cp.Maximize(cp.sum(x))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"CVaR baseline LP failed to solve: status={prob.status}")

    return x.value, float(np.sum(x.value))


def _empirical_cvar(G, cvar_level):
    """Exact closed-form minimum of the Rockafellar-Uryasev function, evaluated per
    column: CVaR_delta[Z] = min_t {t + (1/((1-delta)T)) sum_k max(0, Z_k - t)}.

    G: (T, m) array, one column per row r, values Z_k = g_{k,r}(x) for that row.
    Returns: (m,) array of exact CVaR values.

    a = (1-delta)*T is generally NOT an integer, so the minimizer interpolates
    fractionally between order statistics: with j = floor(a) and frac = a - j,
        CVaR = (1/a) * [ sum of the j largest values + frac * the (j+1)-th largest ].
    A plain "average of the top ceil(a) values" (no fractional weighting) is only
    exact when a happens to be an integer -- using it here would risk a false
    negative in the row-generation violation check, which would break the
    relaxation/feasibility optimality argument in
    solve_cvar_baseline_row_generation's docstring.
    """
    T = G.shape[0]
    a = (1 - cvar_level) * T
    j = min(int(np.floor(a)), T - 1)
    frac = a - j
    sorted_desc = np.sort(G, axis=0)[::-1]  # (T, m), descending per column
    if j == 0:
        return frac * sorted_desc[0, :] / a
    return (sorted_desc[:j, :].sum(axis=0) + frac * sorted_desc[j, :]) / a


def _cvar_row_type_specs(problem):
    """Build the 10 (direction, M, base, limit) descriptors for CVaR row types, in the
    same order as PVHCProblemWithoutBattery._decode_constraint_index's constraint_types
    list. Shared by the row-generation solver's LP-expression builder and its numpy
    violation-check evaluator, so both index rows identically.
    """
    M_v = problem.matrices["M_v_utility"]
    M_p = problem.matrices["M_p_utility"]
    M_q = problem.matrices["M_q_utility"]
    M_pq, M_pmq = M_p + M_q, M_p - M_q

    base_v = problem.base_case["voltage"]
    base_p = problem.base_case["active_power"]
    base_q = problem.base_case["reactive_power"]
    base_pq, base_pmq = base_p + base_q, base_p - base_q

    lim = problem.limits
    return [
        ("upper", M_v, base_v, lim["voltage_upper"]),
        ("lower", M_v, base_v, lim["voltage_lower"]),
        ("upper", M_p, base_p, lim["S_ratings"]),
        ("lower", M_p, base_p, -lim["S_ratings"]),
        ("upper", M_q, base_q, lim["S_ratings"]),
        ("lower", M_q, base_q, -lim["S_ratings"]),
        ("upper", M_pq, base_pq, lim["sqrt2_S_ratings"]),
        ("lower", M_pq, base_pq, -lim["sqrt2_S_ratings"]),
        ("upper", M_pmq, base_pmq, lim["sqrt2_S_ratings"]),
        ("lower", M_pmq, base_pmq, -lim["sqrt2_S_ratings"]),
    ]


def solve_cvar_baseline_row_generation(
    problem,
    cvar_level,
    solver=SOLVER,
    verbose=False,
    max_iterations=50,
    max_active_rows=CVAR_MAX_ACTIVE_ROWS,
    seed=RANDOM_SEED,
):
    """Row/constraint-generation (cutting-plane) formulation of the same CVaR baseline
    as solve_cvar_baseline() — same math (Madavan Eqs. 7, 8, 17-18), but the LP only
    ever holds the ACTIVE subset of the m=10*n_elem constraint rows, instead of all of
    them at once. This mirrors clarkson_solver in src/analyzer.py (RSP's own scenario-
    generation loop) as closely as the row-vs-scenario difference allows, INCLUDING its
    cap-and-importance-sample behavior for when too many things are violated at once:

    Each iteration:
      1. Solve the master LP with only the currently-active rows' (t_r, u_{:,r}) blocks.
      2. Cheaply evaluate ALL m rows' empirical CVaR at the current x* — no LP needed,
         just the *exact* closed form for the RU minimum over t (see _empirical_cvar
         below), which is what the RU epigraph converges to at its optimum. This is a
         single (T, m) numpy computation, not a tensor over columns.
      3. If few rows are newly violated (active + violated <= max_active_rows), add all
         of them — same as before.
         If too many are violated at once, do what clarkson_solver does: identify the
         current *support* (active rows whose CVaR constraint has a non-zero dual, i.e.
         provably binding), keep only those, and fill the remaining budget up to
         max_active_rows with a sample of the violated rows drawn with probability
         proportional to an importance weight that doubles each time a row is found
         violated (so rows that keep coming back are more likely to be picked next
         time they're capped out). Non-support, non-sampled rows are dropped from the
         active set — if they're still needed, the check in step 2 will catch them
         again on a later iteration.
    Terminates when no previously-inactive row is violated. Correctness argument: the
    active-row LP is a relaxation of the full m-row problem (dropping constraints can
    only enlarge the feasible set), so its optimum is always >= the true optimum while
    rows are missing; once the solution also satisfies every *inactive* row (verified
    by the exact check in step 2), it is simultaneously feasible for the full problem
    and optimal over a superset of its feasible region, hence optimal for the full
    problem too. This only holds if the check in step 2 is exact -- an approximate
    check (e.g. plain top-k averaging when (1-delta)*T isn't an integer) could produce
    a false negative and silently return a suboptimal point, so _empirical_cvar uses
    the exact fractional-order-statistic formula, not a rounded approximation. If the
    cap can't be respected because the support alone already exceeds max_active_rows,
    this raises rather than silently returning an unconverged/possibly-suboptimal point
    (unlike clarkson_solver, which warns and breaks) -- preserves the exactness
    guarantee above.
    """
    n_util = problem.num_utility_pv
    T = problem.num_scenarios
    n_elem = problem.limits["voltage_upper"].shape[0]
    m = 10 * n_elem
    pv = problem.pv_profiles_processed  # (T, n_util)

    type_specs = _cvar_row_type_specs(problem)
    rng = np.random.RandomState(seed)
    weights = np.ones(
        m
    )  # importance weights, mirrors clarkson_solver's scenario weights

    def eval_all_rows(x_val):
        """Numpy (no cvxpy) evaluation of g_{k,r}(x_val) for every (k, r). Returns (T, m)."""
        Y_val = pv * x_val[None, :]
        blocks = []
        for direction, M, base, limit in type_specs:
            lhs = base + Y_val @ M.T  # (T, n_elem)
            blocks.append(lhs - limit if direction == "upper" else limit - lhs)
        return np.hstack(blocks)

    def row_expr(Y, r):
        """cvxpy affine (T,) expression for g_{:,r}(x), reusing the shared Y = pv*x."""
        c, e = divmod(r, n_elem)
        direction, M, base, limit = type_specs[c]
        lhs = base[:, e] + Y @ M[e, :]
        return lhs - limit[e] if direction == "upper" else limit[e] - lhs

    x = cp.Variable(n_util, nonneg=True)
    Y = cp.multiply(pv, cp.reshape(x, (1, n_util), order="C"))

    active_rows = set()
    t_vars, u_vars = {}, {}
    x_val, violated = None, []

    for iteration in range(max_iterations):
        constraints = [x <= problem.capacity_limits]
        cvar_constraint_idx = {}
        for r in active_rows:
            if r not in t_vars:
                t_vars[r] = cp.Variable()
                u_vars[r] = cp.Variable(T, nonneg=True)
            g_r = row_expr(Y, r)
            constraints.append(u_vars[r] >= g_r - t_vars[r])
            constraints.append(
                t_vars[r] + cp.sum(u_vars[r]) / ((1 - cvar_level) * T) <= 0
            )
            cvar_constraint_idx[r] = (
                len(constraints) - 1
            )  # the averaging (CVaR<=0) constraint

        prob = cp.Problem(cp.Maximize(cp.sum(x)), constraints)
        prob.solve(solver=solver, verbose=verbose)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"CVaR row-generation LP failed at iteration {iteration} "
                f"({len(active_rows)} active rows): status={prob.status}"
            )
        x_val = x.value

        # Support = active rows whose CVaR constraint is actually binding (non-zero
        # dual) at this optimum -- mirrors clarkson_solver's _extract_support_from_duals.
        support_rows = {
            r
            for r, idx in cvar_constraint_idx.items()
            if constraints[idx].dual_value is not None
            and np.any(np.abs(constraints[idx].dual_value) > 1e-8)
        }

        G_all = eval_all_rows(x_val)  # (T, m) -- cheap, plain numpy
        row_cvar = _empirical_cvar(G_all, cvar_level)  # (m,) exact CVaR per row
        violated = [r for r in np.where(row_cvar > 1e-6)[0] if r not in active_rows]

        if not violated:
            print(
                f"[CVaR/row-gen] converged: {len(active_rows)} active rows out of {m} total "
                f"(T={T} scenarios) -> peak LP size {T * len(active_rows):,} slack vars (u)"
            )
            return x_val, float(np.sum(x_val))

        # Importance-sampling weight update -- mirrors clarkson_solver exactly: a
        # violated row's weight doubles, making it more likely to be sampled next time
        # it's competing for a capped slot.
        for r in violated:
            weights[r] *= 2.0

        if len(active_rows) + len(violated) > max_active_rows:
            num_to_sample = max_active_rows - len(support_rows)
            if num_to_sample <= 0:
                raise RuntimeError(
                    f"CVaR row-generation: support set already has {len(support_rows)} rows "
                    f">= max_active_rows={max_active_rows}. Increase max_active_rows."
                )
            num_to_sample = min(num_to_sample, len(violated))
            violated_arr = np.array(violated)
            p = weights[violated_arr] / weights[violated_arr].sum()
            sampled = rng.choice(violated_arr, size=num_to_sample, replace=True, p=p)
            active_rows = support_rows | set(sampled.tolist())
            print(
                f"[CVaR/row-gen] iter {iteration}: {len(violated)} violated rows exceeds "
                f"max_active_rows={max_active_rows} -- kept {len(support_rows)} support rows + "
                f"sampled {num_to_sample} (importance-weighted) -> active={len(active_rows)}"
            )
        else:
            active_rows = active_rows | set(violated)
            if verbose or len(violated) > 0:
                print(
                    f"[CVaR/row-gen] iter {iteration}: active_rows={len(active_rows)}, "
                    f"total={x_val.sum():.2f} kW, newly_violated={len(violated)}"
                )

    raise RuntimeError(
        f"CVaR row-generation did not converge within {max_iterations} iterations "
        f"({len(active_rows)} active rows, {len(violated)} still violated last check). "
        f"Increase max_iterations if this is just slow to settle."
    )


def solve_cvar(problem, cvar_level, solver=SOLVER, verbose=False):
    """Dispatch to the CVaR solve method selected by CVAR_METHOD (top of file)."""
    if CVAR_METHOD == "dense":
        return solve_cvar_baseline(problem, cvar_level, solver=solver, verbose=verbose)
    elif CVAR_METHOD == "row_generation":
        return solve_cvar_baseline_row_generation(
            problem, cvar_level, solver=solver, verbose=verbose
        )
    else:
        raise ValueError(
            f"Unknown CVAR_METHOD={CVAR_METHOD!r}; expected 'dense' or 'row_generation'"
        )


# ==============================================================================
# ROBUST (BOX) BASELINE — simplified from Chen et al. 2017 (Appendix B): interval hull
# of the assembled coefficients across N samples, worst case at zeta_u >= 0 attained by
# the elementwise max/min coefficient (same max/min trick as
# PVHCProblemWithoutBattery._compute_robust_lhs_cvxpy, applied over scenarios instead of
# over the customer install box). One deterministic constraint per row, no slacks.
# ==============================================================================


def solve_robust_baseline(problem, solver=SOLVER, verbose=False):
    n_util = problem.num_utility_pv
    pv = problem.pv_profiles_processed  # (T, n_util)
    pv_max = pv.max(axis=0)
    pv_min = pv.min(axis=0)

    M_v, M_p, M_q = (
        problem.matrices["M_v_utility"],
        problem.matrices["M_p_utility"],
        problem.matrices["M_q_utility"],
    )
    M_pq, M_pmq = M_p + M_q, M_p - M_q

    base_v = problem.base_case["voltage"]
    base_p = problem.base_case["active_power"]
    base_q = problem.base_case["reactive_power"]
    base_pq, base_pmq = base_p + base_q, base_p - base_q

    x = cp.Variable(n_util, nonneg=True)
    x_hi = cp.multiply(x, pv_max)
    x_lo = cp.multiply(x, pv_min)

    def worst_upper(M, base):
        M_pos, M_neg = np.maximum(M, 0), np.minimum(M, 0)
        return base.max(axis=0) + M_pos @ x_hi + M_neg @ x_lo

    def worst_lower(M, base):
        M_pos, M_neg = np.maximum(M, 0), np.minimum(M, 0)
        return base.min(axis=0) + M_pos @ x_lo + M_neg @ x_hi

    lim = problem.limits
    constraints = [
        x <= problem.capacity_limits,
        worst_upper(M_v, base_v) <= lim["voltage_upper"],
        worst_lower(M_v, base_v) >= lim["voltage_lower"],
        worst_upper(M_p, base_p) <= lim["S_ratings"],
        worst_lower(M_p, base_p) >= -lim["S_ratings"],
        worst_upper(M_q, base_q) <= lim["S_ratings"],
        worst_lower(M_q, base_q) >= -lim["S_ratings"],
        worst_upper(M_pq, base_pq) <= lim["sqrt2_S_ratings"],
        worst_lower(M_pq, base_pq) >= -lim["sqrt2_S_ratings"],
        worst_upper(M_pmq, base_pmq) <= lim["sqrt2_S_ratings"],
        worst_lower(M_pmq, base_pmq) >= -lim["sqrt2_S_ratings"],
    ]
    objective = cp.Maximize(cp.sum(x))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Robust baseline LP failed to solve: status={prob.status}")

    return x.value, float(np.sum(x.value))


def load_rsp_results(feeder_name, sub_folder):
    """Load an RSP result saved by a previous run of this script (SKIP_RSP=False),
    instead of re-running the wait-and-judge loop. Returns the subset of
    run_iterative_pvhc_analysis's return dict that main() actually needs.
    """
    results_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "temp"
        / "optimization_results"
        / feeder_name
        / sub_folder
    )
    summary_path = results_dir / "iterative_algorithm_results.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"SKIP_RSP=True but no saved RSP result found at {results_dir}. "
            f"Run once with SKIP_RSP=False first to produce one."
        )
    loaded = load_optimization_results(feeder_name, sub_folder=sub_folder)
    with open(summary_path, "r") as f:
        algorithm_summary = json.load(f)
    return {
        "capacities_utility": loaded["capacities_isa_utility"],
        "final_num_scenarios": algorithm_summary["final_num_scenarios"],
        "converged": algorithm_summary["converged"],
        "total_optimization_time": algorithm_summary["total_optimization_time"],
    }


# ==============================================================================
# VIOLATION-PROBABILITY EVALUATION (shared held-out test set, all three methods)
# ==============================================================================


def evaluate_violation_prob(zeta_u, test_problem, n_test_requested, tolerance=1e-6):
    """Empirical violation probability on held-out scenarios: fraction of the N_TEST
    requested scenarios for which zeta_u violates >=1 voltage/line constraint.

    Reuses PVHCProblemWithoutBattery._check_scenario_violations (the same true-limit
    check the reference iterative script calls directly for its own validation).
    Denominator is the *requested* N_TEST (matching run_iterative_pvhc_analysis's own
    convention), not the post-zero-PV-filtering count: scenarios dropped for zero PV
    generation cannot be violated by a PV capacity decision, so they count as compliant.
    """
    customer_scaled = np.zeros(0)
    violations = 0
    for t in range(test_problem.num_scenarios):
        pv_t = test_problem.pv_profiles_processed[t]
        utility_scaled = zeta_u * pv_t[: test_problem.num_utility_pv]
        if test_problem._check_scenario_violations(
            t, utility_scaled, customer_scaled, tolerance
        ):
            violations += 1
    return violations / n_test_requested


# ==============================================================================
# REPORTING
# ==============================================================================


def print_summary(rows):
    print("\n" + "=" * 88)
    print(
        f"{'Method':<10}{'N':>8}{'Total PVHC (kW)':>20}{'Violation Prob':>18}{'Solve Time (s)':>18}"
    )
    print("-" * 88)
    for r in rows:
        n_str = str(r["N"]) if r["N"] is not None else "-"
        t_str = f"{r['solve_time_s']:.1f}" if r.get("solve_time_s") is not None else "-"
        print(
            f"{r['method']:<10}{n_str:>8}{r['total_pvhc_kW']:>20.2f}"
            f"{r['violation_prob']:>18.4f}{t_str:>18}"
        )
    print("=" * 88)

    by_method = {r["method"]: r for r in rows}
    if {"Robust", "RSP", "CVaR"} <= by_method.keys():
        pvhc_order_ok = (
            by_method["Robust"]["total_pvhc_kW"]
            <= by_method["RSP"]["total_pvhc_kW"]
            <= by_method["CVaR"]["total_pvhc_kW"]
        )
        viol_order_ok = (
            by_method["Robust"]["violation_prob"]
            <= by_method["RSP"]["violation_prob"]
            <= by_method["CVaR"]["violation_prob"]
        )
        if not pvhc_order_ok:
            print(
                "[WARNING] Expected PVHC ordering Robust <= RSP <= CVaR did not hold — "
                "check for a sign/coefficient bug."
            )
        if not viol_order_ok:
            print(
                "[WARNING] Expected violation-probability ordering Robust <= RSP <= CVaR "
                "did not hold — check for a sign/coefficient bug."
            )


def save_results(feeder_name, headline, ablation_rows=None):
    out_dir = (
        pathlib.Path(__file__).resolve().parents[1]
        / "temp"
        / "method_comparison"
        / feeder_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(headline, f, indent=2)

    if ablation_rows is not None:
        with open(out_dir / "ablation_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["method", "N", "total_pvhc_kW", "violation_prob"]
            )
            writer.writeheader()
            writer.writerows(ablation_rows)

    print(f"\nResults saved to: {out_dir}")
    return out_dir


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    print("=" * 80)
    print("METHOD COMPARISON: RSP vs. CVaR baseline vs. Robust (box) baseline")
    print("Utility-owned-only PVHC (utility_pv_ratio = 1.0)")
    print(f"CVaR method: {CVAR_METHOD}")
    print("=" * 80)

    feeder_name = FEEDER_NAME
    cfg = FEEDERS[feeder_name]
    feeder_root = cfg["feeder_root"]
    v_root_nom = cfg["v_root_nom"]
    v_root_multiplier = cfg["v_root_multiplier"]

    print(f"Feeder: {feeder_name}")

    # 1. Circuit model — reuse existing loader/simulator, no re-derivation of the network.
    data_loader = DataLoader(feeder_name)
    data_loader._dssParser()
    data_loader.get_circuit_model(feeder_root=feeder_root)
    simulator = Simulator(data_loader)

    # 2. Full historical scenario pool (same sampler my RSP script uses).
    print("\nLoading full year's scenario pool...")
    _, pv_pool, load_pool, load_info, V0_pool, valid_indices = load_full_scenario_pool(
        data_loader, v_root_nom, v_root_multiplier
    )
    print(f"Valid pool size: {len(valid_indices)} timestamps")

    # 3. Utility-owned PV_info — CRITICAL SIMPLIFYING ASSUMPTION: 100% utility-owned.
    # Per-site capacity uses the same extracted, consumption-based limits (Eq. 12) as
    # the RSP baseline below (use_capacity_limits=True), so all three methods are
    # compared under an identical capacity ceiling; falls back to a uniform 10kW cap
    # if calculate_capacity_limit_and_store.py hasn't been run yet.
    load_info_with_limits = load_capacity_limits(feeder_name)
    if load_info_with_limits is not None:
        utility_pv_info = load_info_with_limits
        print(f"Utility PV sites (all utility-owned, extracted per-site limits): {len(utility_pv_info)}")
    else:
        utility_pv_info = [{**load, "capacity": 10.0} for load in load_info]
        print(f"Utility PV sites (all utility-owned, 10kW uniform cap fallback): {len(utility_pv_info)}")

    # 4. Disjoint train/test scenario draw. CVaR and Robust get their OWN training
    # draws (sized independently via N_BASELINE_CVAR / N_BASELINE_ROBUST — see scale
    # note at top of file for why), both still disjoint from the shared N_TEST set
    # (and from each other) by construction.
    test_idx, (cvar_train_idx, robust_train_idx) = draw_disjoint_indices(
        valid_indices, N_TEST, [N_BASELINE_CVAR, N_BASELINE_ROBUST], seed=RANDOM_SEED
    )
    print(
        f"Drew {len(cvar_train_idx)} CVaR training scenarios, {len(robust_train_idx)} "
        f"Robust training scenarios, and {len(test_idx)} held-out test scenarios "
        f"(all mutually disjoint by construction; seed={RANDOM_SEED})."
    )

    pv_cvar_train, load_cvar_train, V0_cvar_train = slice_scenarios(
        pv_pool, load_pool, V0_pool, cvar_train_idx
    )
    pv_robust_train, load_robust_train, V0_robust_train = slice_scenarios(
        pv_pool, load_pool, V0_pool, robust_train_idx
    )
    pv_test, load_test, V0_test = slice_scenarios(pv_pool, load_pool, V0_pool, test_idx)

    # CVaR is built WITHOUT zero-PV filtering: it's a sample average, and dropping the
    # (always-safe) nighttime scenarios would shrink the population the "worst
    # (1-delta) fraction" is taken over, distorting what the risk level means relative
    # to a full year of hours. See build_compiled_problem's filter_zero_pv docstring.
    problem_train_cvar = build_compiled_problem(
        simulator,
        utility_pv_info,
        load_info,
        pv_cvar_train,
        load_cvar_train,
        V0_cvar_train,
        filter_zero_pv=False,
    )
    problem_train_robust = build_compiled_problem(
        simulator,
        utility_pv_info,
        load_info,
        pv_robust_train,
        load_robust_train,
        V0_robust_train,
    )
    problem_test = build_compiled_problem(
        simulator, utility_pv_info, load_info, pv_test, load_test, V0_test
    )
    print(
        f"CVaR training scenarios (no zero-PV filtering): {problem_train_cvar.num_scenarios}"
    )
    print(
        f"Robust training scenarios after zero-PV filtering: {problem_train_robust.num_scenarios}"
    )
    print(
        f"Test scenarios after zero-PV filtering: {problem_test.num_scenarios} (requested {N_TEST})"
    )

    # 5. Baselines (each uses its own training draw — see point 4).
    print("\n--- Solving CVaR baseline ---")
    t0 = time.time()
    zeta_cvar, total_cvar = solve_cvar(problem_train_cvar, CVAR_LEVEL, solver=SOLVER)
    print(f"CVaR solved in {time.time() - t0:.1f}s. Total PVHC = {total_cvar:.2f} kW")

    print("\n--- Solving Robust (box) baseline ---")
    t0 = time.time()
    zeta_robust, total_robust = solve_robust_baseline(
        problem_train_robust, solver=SOLVER
    )
    print(
        f"Robust solved in {time.time() - t0:.1f}s. Total PVHC = {total_robust:.2f} kW"
    )

    # 6. RSP via the existing, unmodified iterative wait-and-judge routine
    #    (or, if SKIP_RSP, load a result saved by a previous run of this script).
    if SKIP_RSP:
        print(
            f"\n--- Skipping RSP (SKIP_RSP=True); loading previously saved result "
            f"from temp/optimization_results/{feeder_name}/{RSP_SUB_FOLDER}/ ---"
        )
        rsp_results = load_rsp_results(feeder_name, RSP_SUB_FOLDER)
    else:
        print("\n--- Running RSP (existing iterative routine) ---")
        rsp_config = PVHCTestConfig(
            feeder_name=feeder_name,
            feeder_root=feeder_root,
            v_root_nom=v_root_nom,
            v_root_multiplier=v_root_multiplier,
            seed=RANDOM_SEED,
            num_scenarios=INITIAL_NUM_SCENARIOS,
            solver=SOLVER,
            initial_sample_size=INITIAL_SAMPLE_SIZE,
            maximum_subproblem_size=MAX_SUBPROBLEM_SIZE,
            sub_folder=RSP_SUB_FOLDER,
            with_fairness=False,
            utility_pv_ratio=1.0,
            use_capacity_limits=True,
            load_cached_limits=True,
        )
        rsp_results = run_iterative_pvhc_analysis(
            config=rsp_config,
            epsilon=EPSILON,
            beta=BETA,
            n_I=N_I,
            max_outer_iterations=MAX_OUTER_ITERATIONS,
            N_test=RSP_INTERNAL_N_TEST,
            verbose=False,
        )
    zeta_rsp = rsp_results["capacities_utility"]
    total_rsp = float(np.sum(zeta_rsp))
    print(
        f"RSP converged={rsp_results['converged']}, final n={rsp_results['final_num_scenarios']}, "
        f"total PVHC = {total_rsp:.2f} kW"
    )

    # 7. Evaluate all three methods on the SAME shared held-out test set.
    viol_cvar = evaluate_violation_prob(zeta_cvar, problem_test, N_TEST)
    viol_robust = evaluate_violation_prob(zeta_robust, problem_test, N_TEST)
    viol_rsp = evaluate_violation_prob(zeta_rsp, problem_test, N_TEST)

    rows = [
        {
            "method": "RSP",
            "N": rsp_results["final_num_scenarios"],
            "total_pvhc_kW": total_rsp,
            "violation_prob": viol_rsp,
        },
        {
            "method": "CVaR",
            "N": problem_train_cvar.num_scenarios,
            "total_pvhc_kW": total_cvar,
            "violation_prob": viol_cvar,
        },
        {
            "method": "Robust",
            "N": problem_train_robust.num_scenarios,
            "total_pvhc_kW": total_robust,
            "violation_prob": viol_robust,
        },
    ]
    print_summary(rows)

    headline = {
        "feeder_name": feeder_name,
        "cvar_level": CVAR_LEVEL,
        "cvar_method": CVAR_METHOD,
        "n_baseline_cvar": N_BASELINE_CVAR,
        "n_baseline_robust": N_BASELINE_ROBUST,
        "n_test": N_TEST,
        "random_seed": RANDOM_SEED,
        "methods": {
            "RSP": {
                "total_pvhc_kW": total_rsp,
                "per_bus_capacity_kW": np.asarray(zeta_rsp).tolist(),
                "violation_prob": viol_rsp,
                "n_scenarios": rsp_results["final_num_scenarios"],
            },
            "CVaR": {
                "total_pvhc_kW": total_cvar,
                "per_bus_capacity_kW": np.asarray(zeta_cvar).tolist(),
                "violation_prob": viol_cvar,
                "n_scenarios": problem_train_cvar.num_scenarios,
            },
            "Robust": {
                "total_pvhc_kW": total_robust,
                "per_bus_capacity_kW": np.asarray(zeta_robust).tolist(),
                "violation_prob": viol_robust,
                "n_scenarios": problem_train_robust.num_scenarios,
            },
        },
    }

    # 8. Optional sample-size ablation (CVaR & Robust only; RSP is N-independent).
    ablation_rows = None
    if SAMPLE_ABLATION:
        print("\n" + "=" * 80)
        print(f"SAMPLE ABLATION: N in {SAMPLE_ABLATION}")
        print("=" * 80)
        pool_excl_test = np.setdiff1d(valid_indices, test_idx)
        rng_ablation = np.random.RandomState(RANDOM_SEED + 1)

        ablation_rows = [
            {
                "method": "RSP",
                "N": rsp_results["final_num_scenarios"],
                "total_pvhc_kW": total_rsp,
                "violation_prob": viol_rsp,
            },
        ]
        for N in SAMPLE_ABLATION:
            if N > len(pool_excl_test):
                raise ValueError(
                    f"Ablation N={N} exceeds available pool ({len(pool_excl_test)})"
                )
            print(f"\n--- Ablation N={N} ---")
            idx_N = rng_ablation.choice(pool_excl_test, size=N, replace=False)
            pv_N, load_N, V0_N = slice_scenarios(pv_pool, load_pool, V0_pool, idx_N)
            # Same draw, but CVaR sees it unfiltered (no zero-PV drop) while Robust
            # gets the filtered version — same reasoning as the headline run above.
            problem_N_cvar = build_compiled_problem(
                simulator,
                utility_pv_info,
                load_info,
                pv_N,
                load_N,
                V0_N,
                filter_zero_pv=False,
            )
            problem_N_robust = build_compiled_problem(
                simulator, utility_pv_info, load_info, pv_N, load_N, V0_N
            )

            zeta_cvar_N, total_cvar_N = solve_cvar(
                problem_N_cvar, CVAR_LEVEL, solver=SOLVER
            )
            zeta_robust_N, total_robust_N = solve_robust_baseline(
                problem_N_robust, solver=SOLVER
            )
            viol_cvar_N = evaluate_violation_prob(zeta_cvar_N, problem_test, N_TEST)
            viol_robust_N = evaluate_violation_prob(zeta_robust_N, problem_test, N_TEST)

            ablation_rows.append(
                {
                    "method": "CVaR",
                    "N": N,
                    "total_pvhc_kW": total_cvar_N,
                    "violation_prob": viol_cvar_N,
                }
            )
            ablation_rows.append(
                {
                    "method": "Robust",
                    "N": N,
                    "total_pvhc_kW": total_robust_N,
                    "violation_prob": viol_robust_N,
                }
            )
            print(f"  CVaR:   total={total_cvar_N:.2f} kW, viol_prob={viol_cvar_N:.4f}")
            print(
                f"  Robust: total={total_robust_N:.2f} kW, viol_prob={viol_robust_N:.4f}"
            )

    save_results(feeder_name, headline, ablation_rows)


if __name__ == "__main__":
    main()
