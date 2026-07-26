import numpy as np
import sys, pathlib
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

import cvxpy as cp
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery, clarkson_solver
from src.constants import FEEDERS, DEFAULT_FEEDER_NAME

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"
DEFAULT_CANDIDATE_CAPACITY_KW = 10.0


def _build_problem(with_fairness, num_scenarios=30):
    """
    Compile a PVHCProblemWithoutBattery instance on the example feeder with a
    small, fixed scenario set -- enough to exercise compile()/solve_subproblem()
    without paying for a full Algorithm 1 outer loop.
    """
    cfg = FEEDERS[DEFAULT_FEEDER_NAME]
    data_loader = DataLoader(DEFAULT_FEEDER_NAME)
    data_loader.get_circuit_model(feeder_root=cfg["feeder_root"])
    simulator = Simulator(data_loader)

    _, pv_profiles, load_profiles, load_info, V0_profiles = data_loader.get_scenario_data_by_sampling(
        SOLAR_PROFILE_PATH,
        N=num_scenarios,
        v_root_nom=cfg["v_root_nom"],
        v_root_multiplier=cfg["v_root_multiplier"],
    )
    customer_pv_info = [{**load, "capacity": DEFAULT_CANDIDATE_CAPACITY_KW} for load in load_info]

    problem = PVHCProblemWithoutBattery(
        simulator=simulator,
        utility_pv_info=[],
        customer_pv_info=customer_pv_info,
        load_info=load_info,
        with_fairness=with_fairness,
        load_cached_limits=True,
    )
    problem.add_scenarios(load_profiles, pv_profiles, V0_profiles)
    problem.compile()
    return problem


def test_clarkson_solution_respects_capacity_and_is_feasible():
    """
    The Clarkson LP solution must (a) converge with no remaining violations
    among the active scenarios, (b) stay within [0, capacity_limits] per site,
    and (c) find some positive hosting capacity on this feeder.
    """
    problem = _build_problem(with_fairness=False)
    result = clarkson_solver(
        problem, initial_sample_size=10, max_iterations=5, solver=cp.HIGHS, verbose=False
    )

    assert result["converged"], "Clarkson solver should converge on a small scenario set"
    x = result["optimal_capacities"]
    assert x is not None
    assert np.all(x >= -1e-6), f"PV capacities should be non-negative, got min {x.min()}"
    assert np.all(x <= problem.capacity_limits + 1e-6), \
        "PV capacities must respect per-site capacity limits"
    assert result["total_capacity"] > 0, "Expected positive hosting capacity on this feeder"


def test_fairness_objective_solves_with_clarabel():
    """
    The log-barrier fairness objective (Eq. 16) is an exponential-cone problem
    that HiGHS cannot handle. CLARABEL is the documented open-source default
    for it, but can occasionally fail on this problem's harder (tightly-
    constrained) instances -- see README "Known limitations". Try the open-
    source solvers first, then fall back to a licensed one if installed and
    available, so this test doesn't require a commercial license to exist,
    but still passes wherever one happens to be present.
    """
    problem = _build_problem(with_fairness=True)
    result = None
    for solver in (cp.CLARABEL, cp.ECOS, cp.GUROBI, cp.MOSEK):
        try:
            result = clarkson_solver(
                problem, initial_sample_size=10, max_iterations=5, solver=solver, verbose=False
            )
            break
        except Exception:
            continue
    if result is None:
        raise AssertionError("No available solver could solve the fairness objective")

    assert result["converged"], "Clarkson solver should converge with the fairness objective"
    x = result["optimal_capacities"]
    assert x is not None
    assert np.all(x >= -1e-6)
    assert np.all(x <= problem.capacity_limits + 1e-6)


if __name__ == "__main__":
    test_clarkson_solution_respects_capacity_and_is_feasible()
    test_fairness_objective_solves_with_clarabel()
    print("All tests passed.")
