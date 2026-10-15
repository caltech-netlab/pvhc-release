"""
Solving a CVXPY problem with a list of solvers, falling back on any failure.

A subproblem can fail in several ways that a plain `problem.solve(solver=...)` either
raises in different exception types or does not raise at all:
  - `cp.error.SolverError` (e.g. Clarabel's NumericalError / InsufficientProgress);
  - a plain `Exception` from inside a solver interface;
  - a Rust panic in Clarabel, which reaches Python as `pyo3_runtime.PanicException`, a
    `BaseException` that `except Exception` does not catch;
  - no exception but no usable solution: status USER_LIMIT (iteration/time limit) or
    INFEASIBLE_INACCURATE / UNBOUNDED_INACCURATE.

`solve_with_fallback` tries the given solvers in order, prints every failed attempt
whatever the verbosity, and raises `cp.error.SolverError` listing all of them if none
works. An INFEASIBLE or UNBOUNDED status also moves on to the next solver (the subproblems here
are feasible and bounded by construction, so it is most likely a numerical misdiagnosis),
and is returned only if no solver finds a solution. Errors that no other solver can fix
(a non-DCP problem, running out of memory) are raised at once.

Clarabel >= 0.11 picks its "faer" linear solver by default ("auto") for the dense
subproblems here, which has panicked; `CLARABEL_QDLDL` pins the QDLDL solver that Clarabel
0.10 used by default.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cvxpy as cp

SolverSpec = Union[str, Tuple[str, Dict[str, Any]]]

CLARABEL_QDLDL: SolverSpec = (cp.CLARABEL, {"direct_solve_method": "qdldl"})

# The Clarabel retries change its iteration path (a shorter step, no equilibration):
# repeating the same deterministic settings would fail the same way.
CLARABEL_RETRIES: List[SolverSpec] = [
    (cp.CLARABEL, {"direct_solve_method": "qdldl", "max_iter": 1000, "max_step_fraction": 0.95}),
    (cp.CLARABEL, {"direct_solve_method": "qdldl", "equilibrate_enable": False}),
]

# For the log-fairness objective (exponential cone): open-source first. GUROBI is not listed:
# CVXPY's GUROBI interface does not accept exponential cones.
FAIRNESS_SOLVERS: List[SolverSpec] = [
    CLARABEL_QDLDL,
    *CLARABEL_RETRIES,
    (cp.ECOS, {"max_iters": 500}),
    cp.MOSEK,
]

_ACCEPTED = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
_DEFINITIVE = (cp.INFEASIBLE, cp.UNBOUNDED)


def _as_spec_list(solver: Union[SolverSpec, Sequence[SolverSpec], None]) -> List[Tuple[str, Dict[str, Any]]]:
    if solver is None:
        return [(None, {})]
    if isinstance(solver, str) or (isinstance(solver, tuple) and len(solver) == 2 and isinstance(solver[1], dict)):
        solver = [solver]
    specs = []
    for s in solver:
        name, opts = (s if isinstance(s, tuple) else (s, {}))
        specs.append((name, dict(opts)))
    return specs


def describe(spec: Tuple[str, Dict[str, Any]]) -> str:
    name, opts = spec
    return f"{name}({', '.join(f'{k}={v}' for k, v in opts.items())})" if opts else str(name)


def solve_with_fallback(
    problem: cp.Problem,
    solver: Union[SolverSpec, Sequence[SolverSpec], None],
    verbose: bool = False,
    label: str = "subproblem",
) -> Tuple[str, Optional[str]]:
    """
    Solve `problem` with the first solver in `solver` that succeeds.

    Args:
        solver: a CVXPY solver name, a (name, options) tuple, or a list of either, tried in
            order. Solvers that are not installed are skipped (with a message).
        verbose: passed to the solver.
        label: what to call the problem in messages.

    Returns:
        (status, solver description). The status is OPTIMAL or OPTIMAL_INACCURATE (the
        latter with a printed warning), or INFEASIBLE / UNBOUNDED if that is all any solver
        reported.

    Raises:
        cp.error.SolverError if every solver failed, listing why.
    """
    installed = set(cp.installed_solvers())
    failures, definitive = [], None
    for spec in _as_spec_list(solver):
        name, opts = spec
        if isinstance(name, str):
            name = name.upper()
            spec = (name, opts)
        if name is not None and name not in installed:
            failures.append(f"{describe(spec)}: not installed")
            print(f"  [solver] {label}: {describe(spec)} is not installed, skipping.")
            continue
        try:
            # warm_start=False: otherwise CVXPY reuses the previous attempt's solver object
            # and its settings (e.g. Clarabel's direct_solve_method) on the same problem.
            problem.solve(solver=name, verbose=verbose, warm_start=False, **opts)
        except BaseException as e:  # noqa: BLE001 -- see the PanicException check below
            if not isinstance(e, Exception) and type(e).__name__ != "PanicException":
                raise  # KeyboardInterrupt, SystemExit, ...
            if isinstance(e, (cp.error.DCPError, cp.error.DGPError, MemoryError)):
                raise  # the same for every solver
            reason = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
            failures.append(f"{describe(spec)}: {reason}")
            print(f"  [solver] {label}: {describe(spec)} failed ({reason}); trying the next solver.")
            continue
        status = problem.status
        if status in _ACCEPTED:
            if status == cp.OPTIMAL_INACCURATE:
                print(f"  [solver] {label}: {describe(spec)} returned {status}; using its solution.")
            return status, describe(spec)
        if status in _DEFINITIVE and definitive is None:
            definitive = (status, describe(spec))
        failures.append(f"{describe(spec)}: status {status}")
        print(f"  [solver] {label}: {describe(spec)} ended with status {status}; trying the next solver.")
    if definitive is not None:
        print(f"  [solver] {label}: no solver found a solution; reporting {definitive[0]} from {definitive[1]}.")
        return definitive
    raise cp.error.SolverError(f"{label}: every solver failed -- " + "; ".join(failures))
