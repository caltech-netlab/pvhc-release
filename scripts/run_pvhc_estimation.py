"""
PVHC Estimation via Robust-Scenario Program — Paper Section IV.C, Algorithm 1, Table I

Implements the outer scenario-update loop (Algorithm 1) that increases the number
of i.i.d. scenarios until the Campi-Garatti phi criterion (Theorem III.2) is met,
guaranteeing that the optimal solution violates at most epsilon of unseen
scenarios with confidence 1-beta.

Algorithm outline:
    1. Start with N scenarios sampled i.i.d. (Section III.B)
    2. Solve the LP subproblem via clarkson_solver (src/analyzer.py)
    3. Count support constraints k = |{support set}|
    4. Check phi_{k,N,beta}(1-epsilon) >= 0 via update_num_scenarios (src/utils.py)
    5. If not satisfied, increase N to the required N' and repeat from step 1
    6. Validate with N_TEST held-out scenarios (empirical violation rate)

Outputs (Table I / Fig. 4-5 style, on the example feeder):
  - convergence N, support size k, total PVHC per configuration
  - histogram of customer PV capacities (no fairness vs. fairness)
  - total PVHC across utility ratios
"""

import numpy as np
import sys
import pathlib
import time
import json
import cvxpy as cp

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.data_loader import DataLoader
from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, DEFAULT_CANDIDATE_PV_CAPACITY_KW
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery, clarkson_solver
from src.utils import (
    store_optimization_results,
    load_optimization_results,
    PVHCTestConfig,
    update_num_scenarios,
    get_low_capacity_bus_indices,
    load_capacity_limits,
)

np.set_printoptions(precision=3, threshold=10000, linewidth=170)

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"


def run_iterative_pvhc_analysis(
    config: PVHCTestConfig,
    epsilon: float = 0.01,
    beta: float = 0.05,
    n_I: int = 1,
    max_outer_iterations: int = 20,
    N_test: int = 5000,
    verbose: bool = True,
    data_loader: DataLoader = None,
):
    """
    Run PVHC estimation with iterative scenario addition until the epsilon-beta
    guarantee (Algorithm 1) is met.

    Parameters:
    -----------
    config : PVHCTestConfig
        Configuration object for the PVHC run
    epsilon : float
        Tolerance level for reliability (violation probability)
    beta : float
        Confidence level
    n_I : int
        Increment for number of scenarios in each iteration
    data_loader : DataLoader, optional
        A pre-built DataLoader to reuse instead of constructing/parsing a fresh
        one (e.g. `scripts/dual_analysis.py` passes one with modified branch
        impedances for the infrastructure-upgrade study, Table V).
    max_outer_iterations : int
        Maximum number of outer iterations for the scenario-update loop
    N_test : int
        Number of held-out test scenarios for validation
    verbose : bool
        Whether to print detailed progress

    Returns:
    --------
    dict : results and algorithm info
    """
    feeder_name = config.feeder_name
    feeder_root = config.feeder_root
    v_root_multiplier = config.v_root_multiplier
    seed = config.seed if hasattr(config, "seed") else 42

    print("=" * 80)
    print("Iterative PVHC Estimation with Epsilon-Beta Guarantees")
    print("=" * 80)
    print(f"Epsilon (violation probability): {epsilon}")
    print(f"Beta (confidence level): {beta}")
    print(f"Scenario increment (n_I): {n_I}")
    print(f"Max outer iterations: {max_outer_iterations}")
    print(f"Test samples for validation: {N_test}")

    if data_loader is None:
        print("\n1. Initializing DataLoader...")
        data_loader = DataLoader(feeder_name)

        print("2. Compiling OpenDSS files...")
        network_data = data_loader._dssParser()
        assert network_data, "Network data should not be empty"

        print("3. Computing LinDistFlow model parameters...")
        data_loader.get_circuit_model(feeder_root=feeder_root)
    else:
        print("\n1-3. Reusing pre-built DataLoader (already parsed/compiled).")

    print("4. Creating simulator...")
    simulator = Simulator(data_loader)

    N_samples = config.num_scenarios  # Initial number of scenarios
    outer_iter = 0
    converged = False
    iteration_history = []

    optimal_capacities = None
    final_support_size = 0
    total_optimization_time = 0

    print("\n" + "=" * 80)
    print("Starting Iterative Scenario Update Loop")
    print("=" * 80)

    while outer_iter < max_outer_iterations and not converged:
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{max_outer_iterations}")
        print(f"{'='*60}")
        print(f"Current number of scenarios: {N_samples}")

        iter_start_time = time.time()
        current_seed = seed + outer_iter * 1000  # Different seed for each outer iteration

        time_array, pv_profiles, load_profiles, load_info, V0_samples = (
            data_loader.get_scenario_data_by_sampling(
                SOLAR_PROFILE_PATH,
                N=N_samples,
                v_root_nom=config.v_root_nom,
                v_root_multiplier=v_root_multiplier,
                seed=current_seed,
            )
        )

        if verbose:
            print(f"\nScenario Data Summary:")
            print(f"   - Sampled scenarios: {len(time_array)}")
            print(f"   - PV profiles shape: {pv_profiles.shape}")
            print(f"   - V0 samples shape: {V0_samples.shape}")

        # Setup PV information (only on first iteration)
        if outer_iter == 0:
            utility_pv_ratio = getattr(config, "utility_pv_ratio", 0.0)
            if not (0.0 <= utility_pv_ratio <= 1.0):
                raise ValueError(f"utility_pv_ratio must be in [0,1], got {utility_pv_ratio}")

            use_capacity_limits = getattr(config, "use_capacity_limits", False)
            load_info_with_limits = None

            if use_capacity_limits:
                print("\n   Loading capacity limits...")
                load_info_with_limits = load_capacity_limits(feeder_name)
                if load_info_with_limits is not None:
                    limits = [load["capacity"] for load in load_info_with_limits]
                    print(f"   - Loaded capacity limits for {len(load_info_with_limits)} loads")
                    print(f"   - Capacity limit range: [{min(limits):.3f}, {max(limits):.3f}]")
                    PV_info = load_info_with_limits
                else:
                    print("   Capacity limits not found; falling back to uniform capacity")
                    PV_info = [{**load, "capacity": DEFAULT_CANDIDATE_PV_CAPACITY_KW} for load in load_info]
            else:
                PV_info = [{**load, "capacity": DEFAULT_CANDIDATE_PV_CAPACITY_KW} for load in load_info]

            num_total_pv = len(PV_info)
            num_utility_pv_sites = int(np.round(num_total_pv * utility_pv_ratio))
            prioritize_low_capacity = getattr(config, "prioritize_low_capacity_buses", False)

            if utility_pv_ratio == 0.0:
                utility_pv_info = []
                customer_pv_info = PV_info
            elif utility_pv_ratio == 1.0:
                utility_pv_info = PV_info
                customer_pv_info = []
            else:
                if prioritize_low_capacity:
                    reference_subfolder = getattr(config, "reference_subfolder", "without_fairness")
                    print(f"   - Loading baseline results from {reference_subfolder}...")
                    try:
                        sorted_indices, sorted_capacities = get_low_capacity_bus_indices(
                            feeder_name, reference_subfolder
                        )
                        print(f"   - Prioritizing buses with low baseline capacity")
                        utility_indices = sorted_indices[:num_utility_pv_sites]
                        customer_indices = sorted_indices[num_utility_pv_sites:]
                    except Exception as e:
                        print(f"   - Warning: Could not load baseline results: {e}")
                        print(f"   - Falling back to random selection")
                        prioritize_low_capacity = False

                if not prioritize_low_capacity:
                    np.random.seed(seed)
                    indices = np.random.permutation(num_total_pv)
                    utility_indices = indices[:num_utility_pv_sites]
                    customer_indices = indices[num_utility_pv_sites:]

                utility_pv_info = [PV_info[i] for i in utility_indices]
                customer_pv_info = [PV_info[i] for i in customer_indices]

            num_utility_pv = len(utility_pv_info)
            num_customer_pv = len(customer_pv_info)
            print(f"   - Utility PV sites: {num_utility_pv}")
            print(f"   - Customer PV sites: {num_customer_pv}")
            print(f"   - Total PV sites: {num_total_pv}")

            # Apply min_capacity to zero-capacity loads
            min_capacity = 0.1
            zero_count = 0
            for pv in utility_pv_info + customer_pv_info:
                if pv.get("capacity", 0) < 1e-6:
                    pv["capacity"] = min_capacity
                    zero_count += 1
            if zero_count > 0:
                print(f"   - Assigned min capacity ({min_capacity} kW) to {zero_count} loads with zero consumption")

        load_cached_limits = getattr(config, "load_cached_limits", False)

        problem = PVHCProblemWithoutBattery(
            simulator=simulator,
            utility_pv_info=utility_pv_info,
            customer_pv_info=customer_pv_info,
            load_info=load_info,
            is_robust=True,
            with_fairness=config.with_fairness,
            fairness_factor=100.0,
            load_cached_limits=load_cached_limits,
        )

        problem.add_scenarios(
            pv_profiles=pv_profiles,
            load_profiles=load_profiles,
            V0_profiles=V0_samples,
            filter_zero_pv=True,
        )
        actual_scenarios = problem.num_scenarios
        print(f"   - Scenarios after filtering: {actual_scenarios}")

        problem.compile()

        solve_start_time = time.time()
        results_isa = clarkson_solver(
            problem=problem,
            is_scenario_based=True,
            initial_sample_size=config.initial_sample_size,
            max_iterations=10,
            maximum_subproblem_size=config.maximum_subproblem_size,
            solver=config.solver,
            verbose=verbose,
            seed=current_seed,
            solve_all=False,
        )
        solve_time = time.time() - solve_start_time
        total_optimization_time += solve_time

        num_active_constraints = len(results_isa["support_indices"])
        optimal_capacities = results_isa["optimal_capacities"]

        print(f"\nClarkson Solver Results:")
        print(f"   - Converged: {results_isa['converged']}")
        print(f"   - Total PV Capacity: {results_isa['total_capacity']:.2f} kW")
        print(f"   - Number of support constraints: {num_active_constraints}")
        print(f"   - Solve time: {solve_time:.2f} seconds")

        if not results_isa["converged"]:
            print(f"   Warning: Clarkson solver did not converge!")
            print(f"   Violated scenarios: {len(results_isa['violated_scenarios'])}")

        new_N_samples = update_num_scenarios(
            epsilon=epsilon,
            beta=beta,
            num_scenarios=N_samples,
            num_active_constraints=num_active_constraints,
            n_I=n_I,
        )

        iter_time = time.time() - iter_start_time
        iteration_history.append({
            "iteration": outer_iter + 1,
            "num_scenarios": N_samples,
            "actual_scenarios": actual_scenarios,
            "num_active_constraints": num_active_constraints,
            "total_pv_capacity": float(results_isa["total_capacity"]),
            "clarkson_converged": results_isa["converged"],
            "solve_time": solve_time,
            "iteration_time": iter_time,
            "new_N_samples": new_N_samples,
        })

        print(f"\nEpsilon-Beta Check:")
        print(f"   - Required new N_samples: {new_N_samples}")

        if new_N_samples == -1:
            print(f"\n{'='*60}")
            print(f"CONVERGENCE ACHIEVED!")
            print(f"Number of scenarios ({actual_scenarios}) is sufficient for")
            print(f"epsilon={epsilon}, beta={beta} with {num_active_constraints} support constraints")
            print(f"{'='*60}")
            converged = True
            final_support_size = num_active_constraints
        else:
            print(f"   - Need to increase scenarios from {N_samples} to {new_N_samples}")
            N_samples = new_N_samples
            outer_iter += 1

    if not converged:
        print(f"\n{'='*60}")
        print(f"WARNING: Maximum outer iterations ({max_outer_iterations}) reached")
        print(f"Algorithm did not converge to epsilon-beta guarantees")
        print(f"{'='*60}")

    capacities_isa_utility = optimal_capacities[:num_utility_pv] if optimal_capacities is not None else None
    capacities_isa_customer = optimal_capacities[num_utility_pv:] if optimal_capacities is not None else None

    # =========================================================================
    # VALIDATION: Test violation probability with a large held-out test sample
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION: Testing Violation Probability")
    print("=" * 80)
    print(f"Generating {N_test} test scenarios...")

    validation_start_time = time.time()
    test_seed = seed + 999999  # Different seed from training scenarios

    time_array_test, pv_profiles_test, load_profiles_test, _, V0_samples_test = (
        data_loader.get_scenario_data_by_sampling(
            SOLAR_PROFILE_PATH,
            N=N_test,
            v_root_nom=config.v_root_nom,
            v_root_multiplier=v_root_multiplier,
            seed=test_seed,
        )
    )

    validation_problem = PVHCProblemWithoutBattery(
        simulator=simulator,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        load_info=load_info,
        is_robust=True,
        with_fairness=config.with_fairness,
        fairness_factor=100.0,
        # Must match the main problem's load_cached_limits (was hardcoded
        # True here regardless of config -- harmless for the standard,
        # disk-cached circuit model this script always uses, but breaks when
        # a caller supplies a data_loader with a freshly recomputed (and thus
        # differently node/edge-ordered) network, e.g. dual_analysis.py's
        # solve_optimization_upgraded, where this reused a stale limits cache
        # from a different network and caused base feasibility to explode
        # from ~1 violation to 700,000+).
        load_cached_limits=load_cached_limits,
    )
    validation_problem.add_scenarios(
        pv_profiles=pv_profiles_test,
        load_profiles=load_profiles_test,
        V0_profiles=V0_samples_test,
        filter_zero_pv=True,
    )
    validation_problem.compile()

    actual_test_scenarios = validation_problem.num_scenarios
    print(f"Test scenarios after filtering: {actual_test_scenarios}")

    if optimal_capacities is not None:
        violation_count = 0
        tolerance = 1e-6
        utility_cap = optimal_capacities[:num_utility_pv]
        customer_cap = optimal_capacities[num_utility_pv:]

        for t in range(actual_test_scenarios):
            pv_t = validation_problem.pv_profiles_processed[t]
            utility_scaled = utility_cap * pv_t[:num_utility_pv]
            customer_scaled = customer_cap * pv_t[num_utility_pv:]
            has_violation = validation_problem._check_scenario_violations(
                t, utility_scaled, customer_scaled, tolerance
            )
            if has_violation:
                violation_count += 1

        violation_probability = violation_count / N_test
        validation_time = time.time() - validation_start_time

        print(f"\nValidation Results:")
        print(f"   - Test scenarios (requested): {N_test}")
        print(f"   - Test scenarios (after filtering): {actual_test_scenarios}")
        print(f"   - Violated scenarios: {violation_count}")
        print(f"   - Empirical violation probability: {violation_probability:.4f}")
        print(f"   - Target epsilon: {epsilon}")
        print(f"   - Validation passed: {violation_probability <= epsilon}")
        print(f"   - Validation time: {validation_time:.2f} seconds")
    else:
        violation_probability = None
        violation_count = 0
        validation_time = 0
        print("   No optimal solution found - skipping validation")

    # =========================================================================
    # Store results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Storing Results")
    print("=" * 80)

    if optimal_capacities is not None:
        store_optimization_results(
            feeder_name,
            utility_pv_info,
            customer_pv_info,
            capacities_isa_utility,
            capacities_isa_customer,
            config,
            simulator.data_loader.network_data_dict["node_list"],
            simulator.data_loader.network_data_dict["edge_list"],
            dual_var=results_isa.get("dual_var", None),
            pv_profiles=problem._pv_profiles,
            load_profiles=problem._load_profiles,
            load_info=problem.load_info,
            scenario_indices=results_isa.get("scenario_indices", None),
        )

    root_path = pathlib.Path(__file__).parent.parent
    results_dir = root_path / "temp" / "optimization_results" / feeder_name
    sub_folder = config.resolved_sub_folder()
    if sub_folder is not None:
        results_dir = results_dir / sub_folder
    results_dir.mkdir(parents=True, exist_ok=True)

    algorithm_summary = {
        "algorithm": "iterative_scenario_addition",
        "epsilon": epsilon,
        "beta": beta,
        "n_I": n_I,
        "converged": converged,
        "final_num_scenarios": N_samples,
        "final_support_size": final_support_size,
        "total_optimization_time": total_optimization_time,
        "num_outer_iterations": outer_iter + 1,
        "validation": {
            "N_test": N_test,
            "actual_test_scenarios": actual_test_scenarios,
            "violation_count": violation_count,
            "violation_probability": violation_probability,
            "validation_passed": (
                violation_probability <= epsilon if violation_probability is not None else False
            ),
            "validation_time": validation_time,
        },
        "iteration_history": iteration_history,
    }

    with open(results_dir / "iterative_algorithm_results.json", "w") as f:
        json.dump(algorithm_summary, f, indent=2)

    print(f"Results stored in: {results_dir}")

    return {
        "converged": converged,
        "optimal_capacities": optimal_capacities,
        "capacities_utility": capacities_isa_utility,
        "capacities_customer": capacities_isa_customer,
        "total_capacity": np.sum(optimal_capacities) if optimal_capacities is not None else 0,
        "final_num_scenarios": N_samples,
        "final_support_size": final_support_size,
        "validation_violation_probability": violation_probability,
        "validation_passed": (
            violation_probability <= epsilon if violation_probability is not None else False
        ),
        "total_optimization_time": total_optimization_time,
        "iteration_history": iteration_history,
        "utility_pv_info": utility_pv_info,
        "customer_pv_info": customer_pv_info,
        "results_dir": str(results_dir),
    }


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME
    feeder_cfg = FEEDERS[feeder_name]

    # Set to True to use per-customer capacity limits derived from annual
    # consumption (Section III, Eq. 12); False uses a uniform default (10 kW).
    USE_EXTRACTED_CAPACITY_LIMITS = True

    EPSILON = 0.01  # Target violation probability (1%)
    BETA = 0.05     # Confidence level (95% confidence)
    N_I = 1         # Scenario increment per iteration
    N_TEST = 5000   # Number of held-out test samples for validation

    # Ablation study: utility-scale PV ratio x fairness objective (Eq. 16).
    # Uncomment additional rows to run the full 6-configuration sweep from the paper.
    ablation_configs = [
        # (utility_ratio, with_fairness, sub_folder)
        # (1.0, False, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_100pct"),
        # (1.0, True, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_100pct_fairness"),
        # (0.3, False, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_30pct"),
        # (0.3, True, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_30pct_fairness"),
        (0.0, False, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_0pct"),
        (0.0, True, f"iterative_eps{EPSILON}_beta{BETA}_utility_ratio_0pct_fairness"),
    ]

    common_params = {
        "feeder_name": feeder_name,
        "feeder_root": feeder_cfg["feeder_root"],
        "v_root_nom": feeder_cfg["v_root_nom"],
        "v_root_multiplier": feeder_cfg["v_root_multiplier"],
        "num_scenarios": 50,  # Initial count; grown iteratively by Algorithm 1
        "initial_sample_size": 10,
        "seed": 42,  # Fixed seed ensures scenario consistency across configurations
        "maximum_subproblem_size": 10,
    }

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY: Running {len(ablation_configs)} configurations")
    print(f"Feeder: {common_params['feeder_name']}")
    print(f"Seed: {common_params['seed']} (ensures same scenarios across all runs)")
    print(f"Capacity Limits: {'EXTRACTED (consumption/generation)' if USE_EXTRACTED_CAPACITY_LIMITS else 'UNIFORM'}")
    print(f"Epsilon: {EPSILON}, Beta: {BETA}")
    print("=" * 80)

    for idx, (utility_ratio, fairness, subfolder) in enumerate(ablation_configs):
        print("\n" + "#" * 80)
        print(f"# CONFIGURATION {idx+1}/{len(ablation_configs)}")
        print(f"# Utility Ratio: {utility_ratio*100:.0f}%, Fairness: {fairness}")
        print(f"# Sub-folder: {subfolder}")
        print("#" * 80 + "\n")

        config = PVHCTestConfig(
            **common_params,
            sub_folder=subfolder,
            with_fairness=fairness,
            # HiGHS (LP-only) is fast but can't handle the log-fairness objective
            # (Eq. 16, an exponential-cone problem). CLARABEL and ECOS both can,
            # but either can occasionally fail on this problem's harder instances
            # once real (non-inflated) network limits make some scenarios
            # numerically tight; clarkson_solver tries each in order and falls
            # back to the next on a SolverError. GUROBI/MOSEK are listed last as
            # an optional, licensed fallback -- harmless to list even if not
            # installed (that just fails immediately and moves on), and only
            # actually used if the open-source solvers above it couldn't solve.
            solver=[cp.CLARABEL, cp.ECOS, cp.GUROBI, cp.MOSEK] if fairness else cp.HIGHS,
            utility_pv_ratio=utility_ratio,
            use_capacity_limits=USE_EXTRACTED_CAPACITY_LIMITS,
            load_cached_limits=True,
        )

        results = run_iterative_pvhc_analysis(
            config=config,
            epsilon=EPSILON,
            beta=BETA,
            n_I=N_I,
            max_outer_iterations=20,
            N_test=N_TEST,
            verbose=True,
        )

        print("\n" + "=" * 80)
        print(f"CONFIGURATION {idx+1} SUMMARY")
        print("=" * 80)
        print(f"Algorithm Converged: {results['converged']}")
        print(f"Total PV Capacity: {results['total_capacity']:.2f} kW")
        print(f"Final Number of Scenarios: {results['final_num_scenarios']}")
        print(f"Final Support Size: {results['final_support_size']}")
        viol_prob = results["validation_violation_probability"]
        viol_prob_str = f"{viol_prob:.4f}" if viol_prob is not None else "N/A"
        print(f"Validation Violation Probability: {viol_prob_str}")
        print(f"Validation Passed: {results['validation_passed']}")
        print(f"Total Optimization Time: {results['total_optimization_time']:.2f} seconds")
        print(f"Results stored in: {results['results_dir']}")

        # Sanity check that results were saved correctly -- distinct from the
        # epsilon/beta statistical guarantee checked above (that's `results['validation_passed']`).
        loaded_results = load_optimization_results(
            config.feeder_name, sub_folder=config.resolved_sub_folder()
        )
        if loaded_results is None:
            print(f"\n[Config {idx+1}] Output check failed: could not load saved results")
        else:
            util_total = float(loaded_results["capacities_isa_utility"].sum())
            cust_total = float(loaded_results["capacities_isa_customer"].sum())
            if np.isnan(util_total) or np.isnan(cust_total):
                print(f"\n[Config {idx+1}] Output check failed: NaN detected in saved results")
            else:
                print(f"\n[Config {idx+1}] Output check passed")
                print(f"   Total capacity: {util_total + cust_total:.2f} kW")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
