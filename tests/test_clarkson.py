"""
Synthetic tests for the solver fallback (`src/solver_utils.py`), the dual-based support
detection (`support_from_duals`), `clarkson_solver`'s termination handling, and the exact
constraint row norms of `PVHCProblemWithoutBattery` / `PVHCProblemWithVoltVar`.

Nothing here builds a feeder: every problem is a few variables and scenarios, and the PVHC
classes are created with `cls.__new__(cls)` and small random matrices.
"""
import functools
import sys, pathlib
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[1] / "scripts"))

import numpy as np
import cvxpy as cp
import pytest

from src.analyzer import ClarksonProblem, PVHCProblemWithoutBattery, clarkson_solver, support_from_duals
from src.solver_utils import solve_with_fallback
from analyzer_voltvar import PVHCProblemWithVoltVar

# A user-limited or ECOS-limited attempt makes CVXPY warn "Solution may be inaccurate".
pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate")


# ==============================================================================
# solve_with_fallback
# ==============================================================================

class PanicException(BaseException):
    """Stands in for pyo3_runtime.PanicException (a Rust panic in Clarabel)."""


def _lp():
    x = cp.Variable(3, nonneg=True)
    A = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    return cp.Problem(cp.Maximize(cp.sum(x)), [A @ x <= [4.0, 5.0], x <= 2.0])


def _exp_cone_problem():
    """A small instance of the log-fairness objective (exponential cone)."""
    x = cp.Variable(3, nonneg=True)
    A = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    return cp.Problem(cp.Maximize(cp.sum(x) + cp.sum(cp.log(x + 1e-3))), [A @ x <= [4.0, 5.0]])


def _missing_solver():
    installed = set(cp.installed_solvers())
    for name in (cp.MOSEK, cp.GUROBI, cp.CPLEX, cp.XPRESS, cp.COPT):
        if name not in installed:
            return name
    pytest.skip("every candidate solver is installed")


def _spy_solve(monkeypatch, problem, raise_on=None):
    """Record the solver of every problem.solve call; raise raise_on[solver] instead of solving."""
    real_solve = problem.solve
    calls = []

    def solve(*args, solver=None, **kwargs):
        calls.append(solver)
        if raise_on and solver in raise_on:
            raise raise_on[solver]
        return real_solve(*args, solver=solver, **kwargs)

    monkeypatch.setattr(problem, "solve", solve)
    return calls


def test_fallback_after_user_limit_reuses_no_settings(capsys):
    # Same solver twice: without warm_start=False CVXPY would rerun Clarabel with max_iter=1.
    problem = _exp_cone_problem()
    status, used = solve_with_fallback(problem, [(cp.CLARABEL, {"max_iter": 1}), cp.CLARABEL], label="test")
    assert status == cp.OPTIMAL and used == cp.CLARABEL
    assert "CLARABEL(max_iter=1) ended with status user_limit" in capsys.readouterr().out

    assert problem.value == pytest.approx(_exp_cone_problem().solve(solver=cp.CLARABEL), rel=1e-6)


def test_fallback_after_panic(monkeypatch, capsys):
    problem = _lp()
    calls = _spy_solve(monkeypatch, problem,
                       raise_on={cp.CLARABEL: PanicException("attempt to subtract with overflow")})
    status, used = solve_with_fallback(problem, [cp.CLARABEL, cp.HIGHS], label="test")
    assert (status, used) == (cp.OPTIMAL, cp.HIGHS)
    assert calls == [cp.CLARABEL, cp.HIGHS]
    assert "CLARABEL failed (PanicException: attempt to subtract with overflow)" in capsys.readouterr().out
    assert problem.value == pytest.approx(_lp().solve(solver=cp.HIGHS), rel=1e-6)


def test_fallback_skips_solver_not_installed(monkeypatch, capsys):
    missing = _missing_solver()
    problem = _lp()
    calls = _spy_solve(monkeypatch, problem)
    status, used = solve_with_fallback(problem, [missing, cp.HIGHS], label="test")
    assert (status, used) == (cp.OPTIMAL, cp.HIGHS)
    assert calls == [cp.HIGHS]
    assert f"{missing} is not installed, skipping" in capsys.readouterr().out


def test_fallback_raises_listing_every_failure(monkeypatch, capsys):
    missing = _missing_solver()
    problem = _exp_cone_problem()
    _spy_solve(monkeypatch, problem, raise_on={cp.ECOS: PanicException("attempt to subtract with overflow")})
    with pytest.raises(cp.error.SolverError) as excinfo:
        solve_with_fallback(problem, [(cp.CLARABEL, {"max_iter": 1}), missing, cp.ECOS], label="test")
    message = str(excinfo.value)
    assert "test: every solver failed" in message
    assert "CLARABEL(max_iter=1): status user_limit" in message
    assert f"{missing}: not installed" in message
    assert "ECOS: PanicException: attempt to subtract with overflow" in message
    # Every failure is printed as it happens, verbose or not
    out = capsys.readouterr().out
    assert "user_limit" in out and "not installed" in out and "PanicException" in out


def test_fallback_reraises_keyboard_interrupt(monkeypatch):
    problem = _lp()
    calls = _spy_solve(monkeypatch, problem, raise_on={cp.CLARABEL: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        solve_with_fallback(problem, [cp.CLARABEL, cp.HIGHS], label="test")
    assert calls == [cp.CLARABEL]


def test_fallback_tries_every_solver_before_reporting_infeasible(monkeypatch, capsys):
    # The subproblems are feasible by construction, so an INFEASIBLE status is more likely a
    # numerical misdiagnosis: the next solver is tried, and INFEASIBLE is returned only if
    # none finds a solution.
    x = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Maximize(cp.sum(x)), [x[0] + x[1] <= -1.0])
    calls = _spy_solve(monkeypatch, problem)
    status, used = solve_with_fallback(problem, [cp.CLARABEL, cp.HIGHS], label="test")
    assert (status, used) == (cp.INFEASIBLE, cp.CLARABEL)
    assert calls == [cp.CLARABEL, cp.HIGHS]
    assert "no solver found a solution; reporting infeasible" in capsys.readouterr().out


# ==============================================================================
# support_from_duals
# ==============================================================================

N_VARS, N_SCENARIOS = 5, 30


def _support_lp(scale=None, objective_scale=1.0, ub=10.0, seed=0):
    """
    maximize sum(x) s.t. 0 <= x <= ub and, per scenario, two constraint types (2 rows and
    1 row) on dense positive rows, the second in units 1e3 times the first. scale[k]
    multiplies scenario k's rows. Returns (problem, blocks for support_from_duals).
    """
    rng = np.random.default_rng(seed)
    scale = scale or {}
    x = cp.Variable(N_VARS, nonneg=True)
    blocks = {}
    for k in range(N_SCENARIOS):
        A = np.abs(rng.normal(size=(3, N_VARS))) + 0.1
        b = rng.uniform(5.0, 10.0, size=3)
        A[2:], b[2:] = 1e3 * A[2:], 1e3 * b[2:]
        A, b = scale.get(k, 1.0) * A, scale.get(k, 1.0) * b
        blocks[k] = [(A[:2] @ x <= b[:2], np.linalg.norm(A[:2], axis=1)),
                     (A[2:] @ x <= b[2:], np.linalg.norm(A[2:], axis=1))]
    constraints = [x <= ub] + [c for items in blocks.values() for c, _ in items]
    return cp.Problem(cp.Maximize(objective_scale * cp.sum(x)), constraints), blocks


@functools.lru_cache(maxsize=None)
def _exact_support():
    """Support of _support_lp() from a simplex (HiGHS) solve: the scenarios with a nonzero dual."""
    problem, blocks = _support_lp()
    problem.solve(solver=cp.HIGHS)
    assert problem.status == cp.OPTIMAL
    support = frozenset(k for k, items in blocks.items() if any(np.any(c.dual_value != 0) for c, _ in items))
    # Same as Definition III.1: dropping a scenario raises the optimum iff it is support
    for k in blocks:
        kept = [problem.constraints[0]] + [c for j, items in blocks.items() if j != k for c, _ in items]
        relaxed_value = cp.Problem(problem.objective, kept).solve(solver=cp.HIGHS)
        assert (relaxed_value > problem.value + 1e-7 * abs(problem.value)) == (k in support)
    return support


def _interior_point_support(**kwargs):
    problem, blocks = _support_lp(**kwargs)
    problem.solve(solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    return support_from_duals(blocks)


def test_support_from_duals_matches_simplex_support():
    exact = _exact_support()
    assert 0 < len(exact) <= N_VARS
    assert _interior_point_support() == exact


@pytest.mark.parametrize("which", ["support", "non-support"])
@pytest.mark.parametrize("factor", [1e4, 1e-4])
def test_support_from_duals_ignores_scenario_scaling(which, factor):
    exact = _exact_support()
    k = min(exact) if which == "support" else min(set(range(N_SCENARIOS)) - exact)
    assert _interior_point_support(scale={k: factor}) == exact


@pytest.mark.parametrize("objective_scale", [1e4, 1e-3])
def test_support_from_duals_ignores_objective_scaling(objective_scale):
    assert _interior_point_support(objective_scale=objective_scale) == _exact_support()


@pytest.mark.parametrize("solver", [cp.CLARABEL, cp.HIGHS])
def test_support_from_duals_empty_when_every_site_at_its_cap(solver):
    problem, blocks = _support_lp(ub=0.5)
    problem.solve(solver=solver)
    assert np.allclose(problem.variables()[0].value, 0.5)
    assert support_from_duals(blocks) == set()


@pytest.mark.parametrize("solver", [cp.CLARABEL, cp.HIGHS])
def test_support_from_duals_all_at_cap_does_not_warn(solver, capsys):
    problem, blocks = _support_lp(ub=0.5)
    problem.solve(solver=solver)
    support_from_duals(blocks)
    assert "may be inaccurate" not in capsys.readouterr().out


def test_support_from_duals_none_without_duals():
    _, blocks = _support_lp()  # never solved
    assert support_from_duals(blocks) is None


# ==============================================================================
# clarkson_solver
# ==============================================================================

class ScenarioLP(ClarksonProblem):
    """
    maximize sum(x) s.t. 0 <= x <= ub and A[k] @ x <= b[k] for every scenario k: a
    ClarksonProblem small enough to also solve whole.
    """

    def __init__(self, A, b, ub=10.0):
        self.A, self.b, self.ub = A, b, ub

    @property
    def num_scenarios(self):
        return len(self.A)

    @property
    def num_constraints_per_scenario(self):
        return self.A[0].shape[0]

    @property
    def num_variables(self):
        return self.A[0].shape[1]

    def compile(self):
        self.row_norms = [np.linalg.norm(A_k, axis=1) for A_k in self.A]

    def solve_subproblem(self, indices, is_scenario_based=True, solver=cp.CLARABEL, verbose=False):
        x = cp.Variable(self.num_variables, nonneg=True)
        blocks = {k: [(self.A[k] @ x <= self.b[k], self.row_norms[k])] for k in indices}
        constraints = [x <= self.ub] + [c for k in indices for c, _ in blocks[k]]
        problem = cp.Problem(cp.Maximize(cp.sum(x)), constraints)
        solve_with_fallback(problem, solver, verbose, label=f"test subproblem ({len(indices)} scenarios)")
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None, float("-inf"), None, None, None
        support = support_from_duals(blocks)
        dual_var = np.array([blocks[k][0][0].dual_value for k in indices])
        return x.value, problem.value, set(indices) if support is None else support, dual_var, list(indices)

    def find_violated_indices(self, solution, tolerance=1e-6, is_scenario_based=True, exclude_indices=None):
        exclude = exclude_indices or set()
        return {k for k in range(self.num_scenarios)
                if k not in exclude and np.max(self.A[k] @ solution - self.b[k]) > tolerance}


def _scenario_lp(n=6, N=40, seed=0, infeasible_scenario=False):
    """Scenario k mostly bounds x[k % n] (support: one scenario per variable)."""
    rng = np.random.default_rng(seed)
    A = [(np.eye(n)[k % n] + 0.05 * rng.uniform(size=n))[None, :] for k in range(N)]
    b = [rng.uniform(1.0, 2.0, size=1) for _ in range(N)]
    if infeasible_scenario:
        A.append(np.ones((1, n)))
        b.append(np.array([-1.0]))
    problem = ScenarioLP(A, b)
    problem.compile()
    return problem


def _full_solution(problem):
    """Optimum and exact (removal-based) support of the whole LP, with HiGHS."""
    everything = list(range(problem.num_scenarios))
    x, value, _, _, _ = problem.solve_subproblem(everything, solver=cp.HIGHS)
    support = {k for k in everything
               if problem.solve_subproblem([j for j in everything if j != k], solver=cp.HIGHS)[1]
               > value + 1e-7 * abs(value)}
    return x, value, support


def test_clarkson_converges_to_full_optimum():
    problem = _scenario_lp()
    x_full, value_full, support_full = _full_solution(problem)
    assert len(support_full) == 6

    result = clarkson_solver(problem, initial_sample_size=5, max_iterations=50,
                             maximum_subproblem_size=20, solver=cp.CLARABEL, verbose=False)
    assert result["converged"] and result["termination_reason"] == "converged"
    assert result["total_capacity"] == pytest.approx(value_full, rel=1e-6)
    np.testing.assert_allclose(result["optimal_capacities"], x_full, atol=1e-5)
    assert set(result["support_indices"]) == support_full
    assert result["violated_scenarios"] == []
    # The duals and their scenarios belong to the solve that produced the capacities
    assert set(result["support_indices"]) <= set(result["scenario_indices"])
    assert len(result["dual_var"]) == len(result["scenario_indices"])


def test_clarkson_grows_subproblem_when_support_fills_cap(capsys):
    problem = _scenario_lp()
    _, value_full, support_full = _full_solution(problem)
    cap = 4
    assert len(support_full) > cap

    result = clarkson_solver(problem, initial_sample_size=2, max_iterations=50,
                             maximum_subproblem_size=cap, solver=cp.CLARABEL, verbose=True)
    assert "fills the subproblem cap" in capsys.readouterr().out
    assert result["converged"] and result["termination_reason"] == "converged"
    assert result["total_capacity"] == pytest.approx(value_full, rel=1e-6)
    assert set(result["support_indices"]) == support_full


def test_clarkson_stops_when_subproblem_would_grow_too_large(capsys):
    problem = _scenario_lp()
    result = clarkson_solver(problem, initial_sample_size=2, max_iterations=50,
                             maximum_subproblem_size=4, max_subproblem_growth=1.0,
                             solver=cp.CLARABEL, verbose=False)
    assert not result["converged"] and result["termination_reason"] == "subproblem_too_large"
    assert result["violated_scenarios"]
    assert result["support_indices"] is not None
    assert "stopping without convergence" in capsys.readouterr().out


def test_clarkson_reports_max_iterations(capsys):
    problem = _scenario_lp()
    result = clarkson_solver(problem, initial_sample_size=2, max_iterations=0,
                             solver=cp.CLARABEL, verbose=False)
    assert not result["converged"] and result["termination_reason"] == "max_iterations"
    assert result["violated_scenarios"]
    assert "Maximum iterations (0) reached" in capsys.readouterr().out


def test_clarkson_reports_failed_subproblem(capsys):
    problem = _scenario_lp(infeasible_scenario=True)
    result = clarkson_solver(problem, initial_sample_size=problem.num_scenarios, max_iterations=10,
                             solver=cp.CLARABEL, verbose=False)
    assert not result["converged"] and result["termination_reason"] == "subproblem_failed"
    assert result["support_indices"] is None
    assert result["optimal_capacities"] is None
    assert "has no solution" in capsys.readouterr().out


# ==============================================================================
# Exact row norms of the PVHC scenario constraints
# ==============================================================================

N_ELEMENTS, N_TIMES = 6, 8


def _random_pvhc(cls, n_u, n_c, is_robust, seed=0):
    """A cls instance with small random impact matrices, base case and limits (no feeder)."""
    rng = np.random.default_rng(seed)
    problem = cls.__new__(cls)
    problem.num_utility_pv, problem.num_customer_pv, problem.is_robust = n_u, n_c, is_robust
    problem.matrices = {}
    for side, n in (("utility", n_u), ("customer", n_c)):
        problem.matrices[f"M_v_{side}"] = rng.uniform(0.0, 2e-3, size=(N_ELEMENTS, n))
        problem.matrices[f"M_p_{side}"] = -rng.uniform(0.2, 1.0, size=(N_ELEMENTS, n))
        problem.matrices[f"M_q_{side}"] = rng.normal(0.0, 0.1, size=(N_ELEMENTS, n))
    problem.pv_profiles_processed = rng.uniform(0.2, 1.0, size=(N_TIMES, n_u + n_c))
    problem.base_case = {"voltage": rng.uniform(0.95, 1.0, size=(N_TIMES, N_ELEMENTS)),
                         "active_power": rng.uniform(0.0, 5.0, size=(N_TIMES, N_ELEMENTS)),
                         "reactive_power": rng.uniform(0.0, 2.0, size=(N_TIMES, N_ELEMENTS))}
    S = rng.uniform(20.0, 40.0, size=N_ELEMENTS)
    problem.limits = {"voltage_upper": np.full(N_ELEMENTS, 1.05 ** 2), "voltage_lower": np.full(N_ELEMENTS, 0.9 ** 2),
                      "S_ratings": S, "sqrt2_S_ratings": np.sqrt(2) * S}
    problem.with_fairness, problem.fairness_factor = False, 1.0
    if cls is PVHCProblemWithVoltVar:
        problem.matrices_qinj = {"M_v_qinj": rng.uniform(0.0, 1e-3, size=(N_ELEMENTS, n_u)),
                                 "M_P_qinj": rng.normal(0.0, 0.05, size=(N_ELEMENTS, n_u)),
                                 "M_Q_qinj": -rng.uniform(0.2, 1.0, size=(N_ELEMENTS, n_u))}
        problem._static = problem._compute_static_combined_matrices()
        problem.inverter_rating_factor, problem.voltvar_period = 1.1, 3
        problem.capacity_limits_utility, problem.capacity_limits_customer = np.full(n_u, 100.0), np.full(n_c, 100.0)
        problem._build_variable_offsets()
    else:
        problem.capacity_limits = np.full(n_u + n_c, 100.0)
        problem._num_elements, problem._num_variables = N_ELEMENTS, n_u + n_c
    return problem


def _true_row_norms(constraints, variables):
    """Norms of every constraint's coefficient rows w.r.t. `variables`, from its expression at unit vectors."""
    for v in variables:
        v.value = np.zeros(v.shape)
    offsets = [np.asarray(c.expr.value, dtype=float).reshape(-1) for c in constraints]
    squared = [np.zeros_like(o) for o in offsets]
    for v in variables:
        for j in np.ndindex(*v.shape):
            unit = np.zeros(v.shape)
            unit[j] = 1.0
            v.value = unit
            for i, c in enumerate(constraints):
                squared[i] += (np.asarray(c.expr.value, dtype=float).reshape(-1) - offsets[i]) ** 2
        v.value = np.zeros(v.shape)
    return [np.sqrt(s) for s in squared]


SITE_COUNTS = [(3, 4), (0, 4), (3, 0)]


@pytest.mark.parametrize("n_u,n_c", SITE_COUNTS)
@pytest.mark.parametrize("is_robust", [True, False])
def test_row_norms_without_battery(n_u, n_c, is_robust):
    problem = _random_pvhc(PVHCProblemWithoutBattery, n_u, n_c, is_robust)
    squared = problem._squared_constraint_matrices()
    x = cp.Variable(n_u + n_c, nonneg=True)
    for t in range(N_TIMES):
        constraints = problem._build_scenario_constraints_cvxpy(x, t)
        norms = problem._scenario_row_norms(t, squared)
        assert len(norms) == len(constraints) == problem.NUM_CONSTRAINT_TYPES
        for got, want in zip(norms, _true_row_norms(constraints, [x])):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("n_u,n_c", SITE_COUNTS)
@pytest.mark.parametrize("is_robust", [True, False])
def test_row_norms_voltvar(n_u, n_c, is_robust):
    problem = _random_pvhc(PVHCProblemWithVoltVar, n_u, n_c, is_robust)
    squared = problem._squared_constraint_matrices()
    zeta_u = cp.Variable(n_u, nonneg=True) if n_u > 0 else None
    zeta_cbar = cp.Variable(n_c, nonneg=True) if n_c > 0 else None
    q_vv = cp.Variable((problem.voltvar_period, n_u)) if n_u > 0 else None
    variables = [v for v in (zeta_u, zeta_cbar, q_vv) if v is not None]
    for t in range(N_TIMES):
        constraints = problem._build_scenario_constraints_cvxpy(zeta_u, zeta_cbar, q_vv, t)
        norms = problem._scenario_row_norms(t, squared)
        assert len(norms) == len(constraints)
        for got, want in zip(norms, _true_row_norms(constraints, variables)):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def _removal_support(problem):
    """Scenarios whose removal raises the optimum (Definition III.1), with HiGHS."""
    def optimum(indices):
        return problem._solve_scenario_based(indices, cp.HIGHS, False)[1]
    everything = list(range(N_TIMES))
    value = optimum(everything)
    return {k for k in everything if optimum([j for j in everything if j != k]) > value + 1e-7 * abs(value)}


@pytest.mark.parametrize("n_u,n_c", [(2, 3), (0, 3)])
def test_support_without_battery_matches_removal_support(n_u, n_c):
    # Also checks that _extract_support_from_duals pairs each scenario with its own constraints
    problem = _random_pvhc(PVHCProblemWithoutBattery, n_u, n_c, is_robust=True)
    exact = _removal_support(problem)
    assert exact
    _, _, support, _, _ = problem._solve_scenario_based(list(range(N_TIMES)), cp.CLARABEL, False)
    assert support == exact


@pytest.mark.parametrize("n_u,n_c", [(2, 3), (0, 3), (2, 0)])
def test_support_voltvar_matches_removal_support(n_u, n_c):
    problem = _random_pvhc(PVHCProblemWithVoltVar, n_u, n_c, is_robust=True)
    exact = _removal_support(problem)
    assert exact
    _, _, support = problem._solve_scenario_based(list(range(N_TIMES)), cp.CLARABEL, False)
    assert support == exact
