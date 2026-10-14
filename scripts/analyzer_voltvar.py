"""scripts/analyzer_voltvar.py

Extension of the robust-scenario PVHC framework in `src/analyzer.py` that adds
a *fixed-schedule Volt-VAR control* capability to utility-owned PV inverters,
on top of the standard robust PVHC formulation (`PVHCProblemWithoutBattery`).

This module intentionally does not modify anything under `src/`. It is a
develop-on-top-of-a-copy of the relevant pieces of `PVHCProblemWithoutBattery`
/ `PVHCProblemWithBattery` (matrix construction, network limits, Clarkson
scenario interface) plus new logic for the reactive-power schedule and the
inverter apparent-power constraint. The generic `ClarksonProblem` interface
and `clarkson_solver` iterative solver are reused unmodified from
`src/analyzer.py`.

Model
-----
Only utility-owned PV sites are assumed to have Volt-VAR-capable inverters
(customer-owned sites remain uncontrolled, q = 0, as in the base paper).
The reactive-power dispatch q_vv(s, i) for utility PV site i is a *fixed,
open-loop schedule* indexed by a repeating time-of-day slot
s in {0, ..., period-1} -- this mirrors the existing open-loop battery
dispatch pattern in `PVHCProblemWithBattery` (`p_b`, `q_b` reused via
`t % battery_period`). The schedule is a decision variable, jointly
optimized with the utility/customer PV capacities.

The extra network-impact terms of a *unit* reactive-power injection at a
utility PV bus (voltage impact via X, and cross-coupling into P/Q branch
flow for delta-connected sites) are computed with the same delta-to-wye
transform used elsewhere in this codebase, following exactly the same math
used in `PVHCProblemWithBattery._compute_battery_impact_matrices` for a
battery's (p, q) dispatch pair -- here restricted to the "q-only" terms,
since the "p" terms are already covered by the standard PV real-power
impact matrices (M_p_utility, M_v_utility, M_q_utility) with q assumed 0.

Inverter apparent-power rating: for each utility PV site i,

    p(t,i)^2 + q(t,i)^2 <= s_rated_i^2

is linearized with the same 4-inequality outer approximation used for
branch/line thermal limits in the paper (eq. (9a)-(9d)): separate bounds on
p, on q, on p+q, and on p-q. s_rated_i = inverter_rating_factor * zeta_u,i
scales with the siting DECISION VARIABLE ζ_u,i itself (the inverter is sized
relative to whatever capacity ends up sited at that location), not with the
site's fixed maximum candidate capacity. Both p(t,i) = sigma_i(t)*zeta_u,i
and s_rated_i are linear in zeta_u,i, so every constraint below (e.g.
`cp.multiply(zeta_u, sigma) <= factor * zeta_u`) is still an affine (LP)
inequality in (ζ_u, q_vv) -- no SOCP needed.

Because q_vv's schedule rows are shared across scenario-days, a *global* box
constraint on q_vv (independent of which scenarios happen to be in the
active Clarkson subproblem) is also always imposed in `_solve_scenario_based`
-- otherwise a schedule slot with no currently-active scenario would have no
constraint at all referencing that row of q_vv (a free LP variable).
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import cvxpy as cp

# NOTE: import order matters. `src.data_loader` inserts the `src/` directory
# onto sys.path (so that `three_phase_lindistflow` becomes importable as a
# top-level module), exactly as every other script in this repo relies on.
from src.data_loader import DataLoader  # noqa: F401  (see note above)
from src.constants import *
from src.simulator import Simulator
from src.analyzer import ClarksonProblem

import three_phase_lindistflow as ldf


class PVHCProblemWithVoltVar(ClarksonProblem):
    """
    Robust-scenario PVHC problem with fixed-schedule Volt-VAR control on
    utility-owned PV inverters.

    Decision variables (packed into a single vector via `_pack_solution` /
    `_unpack_solution` for the generic Clarkson interface):
      - zeta_u    (num_utility_pv,)                 utility PV capacities
      - zeta_cbar (num_customer_pv,)                 regulated customer PV caps
      - q_vv      (voltvar_period, num_utility_pv)   reactive-power schedule
    """

    DEFAULT_LOWER_VOLTAGE_RATIO = 4.0
    DEFAULT_RELAXATION_FACTOR = 1.5
    DEFAULT_FAIRNESS_FACTOR = 100
    DEFAULT_VOLTVAR_PERIOD = 48  # half-hourly, one day (mirrors battery period)

    NUM_NETWORK_CONSTRAINT_TYPES = 10   # 2 voltage + 4 power + 4 approximated (same as PVHCProblemWithoutBattery)
    NUM_INVERTER_CONSTRAINT_TYPES = 8   # p, q, p+q, p-q, each upper/lower (same linearization as line capacity)

    def __init__(
        self,
        simulator: Simulator,
        utility_pv_info: List[Dict[str, Any]],
        customer_pv_info: List[Dict[str, Any]],
        load_info: List[Dict[str, Any]],
        is_robust: bool = True,
        with_fairness: bool = False,
        fairness_factor: float = DEFAULT_FAIRNESS_FACTOR,
        inverter_rating_factor: float = 1.0,
        voltvar_period: int = DEFAULT_VOLTVAR_PERIOD,
        load_cached_limits: bool = False,
    ):
        """
        Args:
            simulator: Simulator instance for power flow / network data.
            utility_pv_info: dicts with keys "bus", "phases", "capacity" (and
                "connected_transformer" for delta-connected sites). These are
                exactly the sites given Volt-VAR-capable inverters.
            customer_pv_info: same shape, customer-owned (no Volt-VAR).
            load_info: dicts with keys "bus", "phases" (used for base case).
            inverter_rating_factor: s_rated_i(zeta_u) = factor * zeta_u,i,
                i.e. the inverter apparent-power rating scales with however
                much PV capacity the optimizer actually sites at location i
                (the decision variable), not with the site's fixed maximum
                candidate capacity. 1.0 = inverter sized exactly to whatever
                DC capacity ends up sited (unity power factor); >1.0 models
                an intentionally oversized/smart inverter that keeps
                reactive headroom even at full real-power output.
            voltvar_period: number of fixed schedule slots per period (e.g.
                48 half-hour slots for one day); reused across all sampled
                days via `t % voltvar_period`, exactly like the existing
                open-loop battery dispatch pattern.
        """
        self.simulator = simulator
        self.utility_pv_info = utility_pv_info or []
        self.customer_pv_info = customer_pv_info or []
        self.load_info = load_info
        self.is_robust = is_robust
        self.with_fairness = with_fairness
        self.fairness_factor = fairness_factor
        self.inverter_rating_factor = float(inverter_rating_factor)
        self.voltvar_period = int(voltvar_period)

        self.vol_tol = VOL_TOL
        self.lower_voltage_limit_ratio = self.DEFAULT_LOWER_VOLTAGE_RATIO
        self.relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        self.v_root_nom = self._get_feeder_nominal_voltage()

        self.num_utility_pv = len(self.utility_pv_info)
        self.num_customer_pv = len(self.customer_pv_info)

        self.network_data = self.simulator.get_network_data()
        self._n_buses = self.network_data["n_buses"]
        self._num_elements = self._n_buses * 3

        # Scenario storage
        self._num_scenarios = 0
        self._load_profiles: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pv_profiles: Optional[np.ndarray] = None
        self._V0_profiles: Optional[np.ndarray] = None
        self._scenario_is_processed = False

        # Populated by compile()
        self._is_compiled = False
        self.base_case: Optional[Dict[str, np.ndarray]] = None
        self.pv_profiles_processed: Optional[np.ndarray] = None
        self.limits: Optional[Dict[str, np.ndarray]] = None
        self.matrices: Optional[Dict[str, np.ndarray]] = None
        self._static: Optional[Dict[str, np.ndarray]] = None
        self.matrices_qinj: Optional[Dict[str, np.ndarray]] = None
        self.capacity_limits_utility: Optional[np.ndarray] = None
        self.capacity_limits_customer: Optional[np.ndarray] = None

        self._num_variables = 0
        self._offsets: Dict[str, int] = {}

        self.adaptive_flag = True
        self.load_cached_limits = load_cached_limits
        # Shared cache directory/format with src/analyzer.py's problem classes,
        # so a limits cache computed by any of them is reusable here (and
        # vice versa) -- important for a fair, apples-to-apples comparison.
        self.limits_cache_dir = "temp/cache/limits"

    # --------------------------------------------------------------------------
    # ClarksonProblem interface
    # --------------------------------------------------------------------------
    @property
    def num_scenarios(self) -> int:
        return self._num_scenarios

    @property
    def num_constraints_per_scenario(self) -> int:
        n = self.NUM_NETWORK_CONSTRAINT_TYPES * self._num_elements
        if self.num_utility_pv > 0:
            n += self.NUM_INVERTER_CONSTRAINT_TYPES * self.num_utility_pv
        return n

    @property
    def num_variables(self) -> int:
        return self._num_variables

    def solve_subproblem(
        self,
        indices: List[int],
        is_scenario_based: bool = True,
        solver: Optional[Any] = cp.HIGHS,
        verbose: bool = False,
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")
        if not is_scenario_based:
            raise NotImplementedError(
                "PVHCProblemWithVoltVar only supports scenario-based Clarkson solving."
            )
        return self._solve_scenario_based(indices, solver, verbose)

    def find_violated_indices(
        self,
        solution: np.ndarray,
        tolerance: float = 1e-6,
        is_scenario_based: bool = True,
        exclude_indices: Optional[Set[int]] = None,
    ) -> Set[int]:
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")
        if not is_scenario_based:
            raise NotImplementedError(
                "PVHCProblemWithVoltVar only supports scenario-based Clarkson solving."
            )
        exclude = exclude_indices or set()
        return self._find_violated_scenarios(solution, tolerance, exclude)

    # --------------------------------------------------------------------------
    # Scenario management
    # --------------------------------------------------------------------------
    def add_scenarios(
        self,
        load_profiles: Tuple[np.ndarray, np.ndarray],
        pv_profiles: np.ndarray,
        V0_profiles: np.ndarray,
        filter_zero_pv: bool = False,
    ) -> None:
        """
        Add scenarios. Defaults to NOT filtering zero-PV (nighttime) scenarios,
        because the fixed reactive-power schedule relies on `t % voltvar_period`
        staying aligned with time-of-day across sampled days (exactly like the
        existing open-loop battery dispatch pattern) -- dropping individual
        nighttime timesteps would break that alignment.
        """
        if filter_zero_pv:
            valid_idx = self._get_nonzero_pv_indices(pv_profiles)
            pv_profiles = pv_profiles[valid_idx]
            V0_profiles = V0_profiles[valid_idx]
            load_profiles = (load_profiles[0][valid_idx], load_profiles[1][valid_idx])

        self._num_scenarios = pv_profiles.shape[0]
        self._load_profiles = load_profiles
        self._pv_profiles = pv_profiles
        self._V0_profiles = V0_profiles
        self._scenario_is_processed = False
        self._is_compiled = False

    def compile(self) -> None:
        if self._pv_profiles is None or self._load_profiles is None or self._V0_profiles is None:
            raise RuntimeError("Scenarios not added. Call add_scenarios() before compile().")

        if not self._scenario_is_processed:
            self._num_scenarios = self._pv_profiles.shape[0]

            base_v, base_P, base_Q, _, _, _ = self.simulator.simulate(
                load_profiles=self._load_profiles,
                pv_profiles=self._pv_profiles,
                V0_profiles=self._V0_profiles,
                load_info=self.load_info,
                PV_info=None,
            )
            self.base_case = {
                "voltage": np.square(base_v),
                "active_power": base_P,
                "reactive_power": base_Q,
            }
            self.pv_profiles_processed = self._preprocess_pv_profiles(self._pv_profiles)
            self._scenario_is_processed = True

        self.matrices = self._compute_pv_impact_matrices()
        self.matrices_qinj = self._compute_qinj_impact_matrices(self.utility_pv_info)
        self._static = self._compute_static_combined_matrices()

        if self.load_cached_limits:
            loaded = self._load_limits_from_cache()
            if loaded is not None:
                self.limits = loaded
                print(f"[Limit Cache] Loaded cached limits for {self.simulator.feeder_name}")
                self._check_basic_feasibility()
            else:
                print("[Limit Cache] No cached limits found, computing new limits")
                self.limits = self._compute_network_limits()
                self._check_basic_feasibility()
                self._save_limits_to_cache()
        else:
            self.limits = self._compute_network_limits()
            self._check_basic_feasibility()
            self._save_limits_to_cache()

        self.capacity_limits_utility = (
            np.array([info["capacity"] for info in self.utility_pv_info], dtype=float)
            if self.num_utility_pv > 0 else np.zeros(0)
        )
        self.capacity_limits_customer = (
            np.array([info["capacity"] for info in self.customer_pv_info], dtype=float)
            if self.num_customer_pv > 0 else np.zeros(0)
        )

        self._build_variable_offsets()
        self._is_compiled = True

    # --------------------------------------------------------------------------
    # Variable packing
    # --------------------------------------------------------------------------
    def _build_variable_offsets(self) -> None:
        off = 0
        self._offsets = {}
        self._offsets["zeta_u"] = off
        off += self.num_utility_pv
        self._offsets["zeta_cbar"] = off
        off += self.num_customer_pv
        self._offsets["q_vv"] = off
        off += self.voltvar_period * self.num_utility_pv
        self._num_variables = off

    def _pack_solution(
        self,
        zeta_u: Optional[cp.Variable],
        zeta_cbar: Optional[cp.Variable],
        q_vv: Optional[cp.Variable],
    ) -> np.ndarray:
        x = np.zeros(self._num_variables, dtype=float)
        if self.num_utility_pv > 0:
            x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv] = \
                np.array(zeta_u.value).reshape(-1)
            n_q = self.voltvar_period * self.num_utility_pv
            x[self._offsets["q_vv"]:self._offsets["q_vv"] + n_q] = np.array(q_vv.value).reshape(-1)
        if self.num_customer_pv > 0:
            x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv] = \
                np.array(zeta_cbar.value).reshape(-1)
        return x

    def _unpack_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.asarray(x).reshape(-1)
        out: Dict[str, np.ndarray] = {}
        out["zeta_u"] = (
            x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv]
            if self.num_utility_pv > 0 else np.zeros(0)
        )
        out["zeta_cbar"] = (
            x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv]
            if self.num_customer_pv > 0 else np.zeros(0)
        )
        if self.num_utility_pv > 0:
            n_q = self.voltvar_period * self.num_utility_pv
            out["q_vv"] = x[self._offsets["q_vv"]:self._offsets["q_vv"] + n_q].reshape(
                self.voltvar_period, self.num_utility_pv
            )
        else:
            out["q_vv"] = np.zeros((self.voltvar_period, 0))
        return out

    # --------------------------------------------------------------------------
    # Scenario-based solving
    # --------------------------------------------------------------------------
    def _solve_scenario_based(
        self,
        scenario_indices: List[int],
        solver: Any,
        verbose: bool,
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        n_u, n_c = self.num_utility_pv, self.num_customer_pv

        zeta_u = cp.Variable(n_u, nonneg=True) if n_u > 0 else None
        zeta_cbar = cp.Variable(n_c, nonneg=True) if n_c > 0 else None
        q_vv = cp.Variable((self.voltvar_period, n_u)) if n_u > 0 else None  # free (signed)

        obj_expr = 0
        if zeta_u is not None:
            obj_expr += cp.sum(zeta_u)
        if zeta_cbar is not None:
            obj_expr += cp.sum(zeta_cbar)

        if self.with_fairness:
            terms = []
            if zeta_u is not None:
                terms.append(cp.sum(cp.log(zeta_u + 1e-5)))
            if zeta_cbar is not None:
                terms.append(cp.sum(cp.log(zeta_cbar + 1e-5)))
            if terms:
                obj_expr += self.fairness_factor * cp.sum(terms)

        objective = cp.Maximize(obj_expr)

        constraints: List = []
        if zeta_u is not None:
            constraints.append(zeta_u <= self.capacity_limits_utility)
        if zeta_cbar is not None:
            constraints.append(zeta_cbar <= self.capacity_limits_customer)

        if n_u > 0:
            # Global schedule box (always present, regardless of which
            # scenarios this subproblem's `scenario_indices` happen to
            # include): every row of q_vv is bounded by +/- the inverter
            # rating implied by the *current* zeta_u. Without this, a
            # schedule slot tau with no scenario in `scenario_indices` this
            # round would have zero constraints referencing that row of
            # q_vv -- a free LP variable with no bound and no objective
            # term. This is required precisely because q_vv's rows are
            # shared across scenario-days while the Clarkson active set
            # only ever samples a subset of scenarios per iteration (and
            # can even evict previously-active ones once the subproblem
            # size cap is hit).
            s_bound_row = cp.reshape(self.inverter_rating_factor * zeta_u, (1, n_u), order="C")
            schedule_box = np.ones((self.voltvar_period, 1)) @ s_bound_row
            constraints.append(q_vv <= schedule_box)
            constraints.append(q_vv >= -schedule_box)

        for t in scenario_indices:
            constraints.extend(self._build_scenario_constraints_cvxpy(zeta_u, zeta_cbar, q_vv, t))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=verbose)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None, float("-inf"), set()

        support = self._extract_support_from_duals(constraints, scenario_indices)
        sol = self._pack_solution(zeta_u, zeta_cbar, q_vv)
        return sol, float(problem.value), support

    def _build_scenario_constraints_cvxpy(
        self,
        zeta_u: Optional[cp.Variable],
        zeta_cbar: Optional[cp.Variable],
        q_vv: Optional[cp.Variable],
        t: int,
    ) -> List:
        n_u, n_c = self.num_utility_pv, self.num_customer_pv
        pv_t = self.pv_profiles_processed[t]

        util_scaled = cp.multiply(zeta_u, pv_t[:n_u]) if n_u > 0 else None
        cust_scaled = cp.multiply(zeta_cbar, pv_t[n_u:n_u + n_c]) if n_c > 0 else None
        q_vv_t = q_vv[t % self.voltvar_period, :] if n_u > 0 else None

        M = self.matrices
        Mq = self.matrices_qinj
        St = self._static

        def lhs(base: np.ndarray, M_u: np.ndarray, M_c_pos: np.ndarray, M_c_neg: np.ndarray,
                M_qinj: np.ndarray, use_max: bool):
            expr = base
            if n_u > 0:
                expr = expr + M_u @ util_scaled
                if M_qinj.size > 0:
                    expr = expr + M_qinj @ q_vv_t
            if n_c > 0:
                expr = expr + (M_c_pos if use_max else M_c_neg) @ cust_scaled
            return expr

        constraints = []

        # --- Voltage upper/lower ---
        base_v = self.base_case["voltage"][t]
        constraints.append(lhs(base_v, M["M_v_utility"], St["M_v_customer_pos"], St["M_v_customer_neg"], Mq["M_v_qinj"], True)
                            <= self.limits["voltage_upper"])
        constraints.append(lhs(base_v, M["M_v_utility"], St["M_v_customer_pos"], St["M_v_customer_neg"], Mq["M_v_qinj"], False)
                            >= self.limits["voltage_lower"])

        # --- Active flow upper/lower ---
        base_P = self.base_case["active_power"][t]
        constraints.append(lhs(base_P, M["M_p_utility"], St["M_p_customer_pos"], St["M_p_customer_neg"], Mq["M_P_qinj"], True)
                            <= self.limits["S_ratings"])
        constraints.append(lhs(base_P, M["M_p_utility"], St["M_p_customer_pos"], St["M_p_customer_neg"], Mq["M_P_qinj"], False)
                            >= -self.limits["S_ratings"])

        # --- Reactive flow upper/lower ---
        base_Q = self.base_case["reactive_power"][t]
        constraints.append(lhs(base_Q, M["M_q_utility"], St["M_q_customer_pos"], St["M_q_customer_neg"], Mq["M_Q_qinj"], True)
                            <= self.limits["S_ratings"])
        constraints.append(lhs(base_Q, M["M_q_utility"], St["M_q_customer_pos"], St["M_q_customer_neg"], Mq["M_Q_qinj"], False)
                            >= -self.limits["S_ratings"])

        # --- Approximated P+Q upper/lower ---
        base_pq = base_P + base_Q
        constraints.append(lhs(base_pq, St["M_pq_utility"], St["M_pq_customer_pos"], St["M_pq_customer_neg"], St["M_pq_qinj"], True)
                            <= self.limits["sqrt2_S_ratings"])
        constraints.append(lhs(base_pq, St["M_pq_utility"], St["M_pq_customer_pos"], St["M_pq_customer_neg"], St["M_pq_qinj"], False)
                            >= -self.limits["sqrt2_S_ratings"])

        # --- Approximated P-Q upper/lower ---
        base_pmq = base_P - base_Q
        constraints.append(lhs(base_pmq, St["M_pmq_utility"], St["M_pmq_customer_pos"], St["M_pmq_customer_neg"], St["M_pmq_qinj"], True)
                            <= self.limits["sqrt2_S_ratings"])
        constraints.append(lhs(base_pmq, St["M_pmq_utility"], St["M_pmq_customer_pos"], St["M_pmq_customer_neg"], St["M_pmq_qinj"], False)
                            >= -self.limits["sqrt2_S_ratings"])

        # --- Inverter apparent-power rating (utility PV sites only) ---
        # p^2 + q^2 <= s_rated^2, linearized exactly like the branch-capacity
        # outer approximation (9a)-(9d): bounds on p, q, p+q, p-q. s_rated
        # scales with the zeta_u decision variable itself (not a fixed
        # capacity cap), so this remains affine (linear) in zeta_u.
        if n_u > 0:
            s_rated = self.inverter_rating_factor * zeta_u
            sqrt2 = np.sqrt(2.0)
            constraints.append(util_scaled <= s_rated)
            constraints.append(util_scaled >= -s_rated)
            constraints.append(q_vv_t <= s_rated)
            constraints.append(q_vv_t >= -s_rated)
            constraints.append(util_scaled + q_vv_t <= sqrt2 * s_rated)
            constraints.append(util_scaled + q_vv_t >= -sqrt2 * s_rated)
            constraints.append(util_scaled - q_vv_t <= sqrt2 * s_rated)
            constraints.append(util_scaled - q_vv_t >= -sqrt2 * s_rated)

        return constraints

    def _extract_support_from_duals(self, constraints: List, scenario_indices: List[int]) -> Set[int]:
        support = set()
        num_non_scenario = int(self.num_utility_pv > 0) + int(self.num_customer_pv > 0)
        if self.num_utility_pv > 0:
            num_non_scenario += 2  # global schedule-box upper/lower (always present)
        per_scenario = self.NUM_NETWORK_CONSTRAINT_TYPES + (
            self.NUM_INVERTER_CONSTRAINT_TYPES if self.num_utility_pv > 0 else 0
        )

        for i, scenario_idx in enumerate(scenario_indices):
            start = num_non_scenario + i * per_scenario
            end = start + per_scenario
            is_support = False
            for j in range(start, min(end, len(constraints))):
                try:
                    dual_val = constraints[j].dual_value
                    if dual_val is not None and np.any(np.abs(dual_val) > 1e-6):
                        is_support = True
                        break
                except (AttributeError, TypeError):
                    continue
            if is_support:
                support.add(scenario_idx)

        if len(support) == 0:
            support = set(scenario_indices)
        return support

    # --------------------------------------------------------------------------
    # Violation checking (numpy, vectorized per scenario)
    # --------------------------------------------------------------------------
    def _find_violated_scenarios(
        self,
        solution: np.ndarray,
        tolerance: float,
        exclude: Set[int],
    ) -> Set[int]:
        parts = self._unpack_solution(solution)
        zeta_u, zeta_cbar, q_vv = parts["zeta_u"], parts["zeta_cbar"], parts["q_vv"]
        n_u, n_c = self.num_utility_pv, self.num_customer_pv

        M = self.matrices
        Mq = self.matrices_qinj
        St = self._static
        v_upper, v_lower = self.limits["voltage_upper"], self.limits["voltage_lower"]
        S = self.limits["S_ratings"]
        sqrt2S = self.limits["sqrt2_S_ratings"]
        # s_rated scales with the *current* numerical zeta_u (decision
        # variable), not a precomputed constant -- recomputed every call
        # (this is a cheap (n_u,)-sized elementwise multiply, NOT the
        # (3153 x n_customer)-sized matrix recomputation that used to
        # happen per scenario here -- see _compute_static_combined_matrices).
        s_rated = self.inverter_rating_factor * zeta_u
        sqrt2 = np.sqrt(2.0)

        violated = set()
        for t in range(self._num_scenarios):
            if t in exclude:
                continue

            pv_t = self.pv_profiles_processed[t]
            util_scaled = zeta_u * pv_t[:n_u] if n_u > 0 else np.zeros(0)
            cust_scaled = zeta_cbar * pv_t[n_u:n_u + n_c] if n_c > 0 else np.zeros(0)
            q_t = q_vv[t % self.voltvar_period] if n_u > 0 else np.zeros(0)

            def lhs(base: np.ndarray, M_u: np.ndarray, M_c_pos: np.ndarray, M_c_neg: np.ndarray,
                    M_qinj: np.ndarray, use_max: bool) -> np.ndarray:
                val = base
                if n_u > 0:
                    val = val + M_u @ util_scaled
                    if M_qinj.size > 0:
                        val = val + M_qinj @ q_t
                if n_c > 0:
                    val = val + (M_c_pos if use_max else M_c_neg) @ cust_scaled
                return val

            bad = False
            base_v = self.base_case["voltage"][t]
            if np.any(lhs(base_v, M["M_v_utility"], St["M_v_customer_pos"], St["M_v_customer_neg"], Mq["M_v_qinj"], True) > v_upper + tolerance):
                bad = True
            elif np.any(lhs(base_v, M["M_v_utility"], St["M_v_customer_pos"], St["M_v_customer_neg"], Mq["M_v_qinj"], False) < v_lower - tolerance):
                bad = True

            base_P = self.base_case["active_power"][t]
            base_Q = self.base_case["reactive_power"][t]

            if not bad:
                if np.any(lhs(base_P, M["M_p_utility"], St["M_p_customer_pos"], St["M_p_customer_neg"], Mq["M_P_qinj"], True) > S + tolerance):
                    bad = True
                elif np.any(lhs(base_P, M["M_p_utility"], St["M_p_customer_pos"], St["M_p_customer_neg"], Mq["M_P_qinj"], False) < -S - tolerance):
                    bad = True

            if not bad:
                if np.any(lhs(base_Q, M["M_q_utility"], St["M_q_customer_pos"], St["M_q_customer_neg"], Mq["M_Q_qinj"], True) > S + tolerance):
                    bad = True
                elif np.any(lhs(base_Q, M["M_q_utility"], St["M_q_customer_pos"], St["M_q_customer_neg"], Mq["M_Q_qinj"], False) < -S - tolerance):
                    bad = True

            if not bad:
                base_pq = base_P + base_Q
                if np.any(lhs(base_pq, St["M_pq_utility"], St["M_pq_customer_pos"], St["M_pq_customer_neg"], St["M_pq_qinj"], True) > sqrt2S + tolerance):
                    bad = True
                elif np.any(lhs(base_pq, St["M_pq_utility"], St["M_pq_customer_pos"], St["M_pq_customer_neg"], St["M_pq_qinj"], False) < -sqrt2S - tolerance):
                    bad = True

            if not bad:
                base_pmq = base_P - base_Q
                if np.any(lhs(base_pmq, St["M_pmq_utility"], St["M_pmq_customer_pos"], St["M_pmq_customer_neg"], St["M_pmq_qinj"], True) > sqrt2S + tolerance):
                    bad = True
                elif np.any(lhs(base_pmq, St["M_pmq_utility"], St["M_pmq_customer_pos"], St["M_pmq_customer_neg"], St["M_pmq_qinj"], False) < -sqrt2S - tolerance):
                    bad = True

            if not bad and n_u > 0:
                p, q = util_scaled, q_t
                if (np.any(p > s_rated + tolerance) or np.any(p < -s_rated - tolerance) or
                        np.any(q > s_rated + tolerance) or np.any(q < -s_rated - tolerance) or
                        np.any(p + q > sqrt2 * s_rated + tolerance) or np.any(p + q < -sqrt2 * s_rated - tolerance) or
                        np.any(p - q > sqrt2 * s_rated + tolerance) or np.any(p - q < -sqrt2 * s_rated - tolerance)):
                    bad = True

            if bad:
                violated.add(t)

        return violated

    # --------------------------------------------------------------------------
    # Matrix construction (adapted from src/analyzer.py)
    # --------------------------------------------------------------------------
    def _get_feeder_nominal_voltage(self) -> float:
        if self.simulator.feeder_name not in FEEDERS:
            raise ValueError(f"Feeder name {self.simulator.feeder_name} not recognized.")
        return FEEDERS[self.simulator.feeder_name]["v_root_nom"]

    def _preprocess_pv_profiles(self, pv_profiles: np.ndarray) -> np.ndarray:
        n_pv_total = self.num_utility_pv + self.num_customer_pv
        if n_pv_total == 0:
            return np.zeros((pv_profiles.shape[0], 0))
        if pv_profiles.ndim == 1:
            return np.repeat(pv_profiles.reshape(-1, 1), n_pv_total, axis=1)
        return pv_profiles

    def _get_nonzero_pv_indices(self, pv_profiles: np.ndarray) -> np.ndarray:
        if pv_profiles.ndim == 1:
            return np.where(pv_profiles > 0)[0]
        return np.where(np.any(pv_profiles > 0, axis=1))[0]

    def _get_transformation_vector(self, conn_trans: Dict, phase_idx: int) -> np.ndarray:
        if not conn_trans:
            return ldf.D2Y[:, phase_idx]
        element_type = conn_trans.get("element_type", "line")
        if element_type in {"line", "YY", "DD", "YD", "DY", "YD_alt",
                             "DY_alt", "openD_openD", "openY_openY"}:
            return ldf.D2Y[:, phase_idx]
        if element_type in {"openY_openD_split", "openD_openD_split"}:
            assert conn_trans.get("split_phase") in conn_trans.get("winding_pairing", [])
            _, permutation = ldf.build_permutation_matrix_open_split(conn_trans)
            return (permutation.T @ ldf.openD_split2Y @ permutation)[:, phase_idx]
        raise ValueError(f"Unknown transformer type: {element_type}")

    def _construct_power_injection_matrices(
        self,
        dev_info: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Impact of a unit REAL power injection at each device (PV, q assumed 0)."""
        R = self.network_data["R"]
        X = self.network_data["X"]
        C_tilde_S = self.network_data["C_tilde_S"]
        node2idx = self.network_data["node2idx"]
        n_buses = self.network_data["n_buses"]

        n_dev = len(dev_info)
        M_v = np.zeros((n_buses * 3, n_dev))
        M_p = np.zeros((n_buses * 3, n_dev))
        M_q = np.zeros((n_buses * 3, n_dev))

        for i, info in enumerate(dev_info):
            bus_name = info["bus"]
            phases = info["phases"]
            if bus_name not in node2idx:
                raise KeyError(f"Bus '{bus_name}' not found in node2idx")
            bus_idx = node2idx[bus_name]

            if phases[1] == 0:
                idx = (bus_idx - 1) * 3 + phases[0] - 1
                M_v[:, i] = np.real(1e-3 * R[:, idx])
                M_p[:, i] = np.real(C_tilde_S[:, idx])
                # Nonzero only below a delta winding (C_tilde_S is complex there)
                M_q[:, i] = np.imag(C_tilde_S[:, idx])
            else:
                phase_idx = ldf.STD_DELTA_ORDER.index(tuple(sorted(phases)))
                trans_vec = self._get_transformation_vector(info.get("connected_transformer", {}), phase_idx)
                idx0 = (bus_idx - 1) * 3
                M_v[:, i] = (
                    1e-3 * R[:, idx0:idx0 + 3] @ np.real(trans_vec) +
                    1e-3 * X[:, idx0:idx0 + 3] @ np.imag(trans_vec)
                )
                branch_pwr = C_tilde_S[:, idx0:idx0 + 3] @ trans_vec
                M_p[:, i] = np.real(branch_pwr)
                M_q[:, i] = np.imag(branch_pwr)

        return M_v, M_p, M_q

    def _compute_pv_impact_matrices(self) -> Dict[str, np.ndarray]:
        M_v_u, M_p_u, M_q_u = self._construct_power_injection_matrices(self.utility_pv_info)
        M_v_c, M_p_c, M_q_c = self._construct_power_injection_matrices(self.customer_pv_info)
        return {
            "M_v_utility": M_v_u, "M_p_utility": M_p_u, "M_q_utility": M_q_u,
            "M_v_customer": M_v_c, "M_p_customer": M_p_c, "M_q_customer": M_q_c,
        }

    def _compute_static_combined_matrices(self) -> Dict[str, np.ndarray]:
        """
        Precompute (once, at compile() time) every t-INDEPENDENT matrix used
        by the per-scenario constraint builders: the combined P+Q / P-Q
        sensitivity matrices, and the robust (max/min-with-zero) positive/
        negative variants of every customer-side matrix.

        These depend only on self.matrices / self.matrices_qinj / self.is_robust
        -- never on the scenario index t -- so computing them inside a
        per-scenario loop (as an earlier version of this file did, in both
        `_build_scenario_constraints_cvxpy` and `_find_violated_scenarios`)
        redundantly reallocates O(num_scenarios) copies of ~140MB worth of
        (3153 x n_customer)-sized arrays per Clarkson iteration -- enough to
        exhaust tens of GB of RAM at full-feeder scale (confirmed via an OOM
        kill in this environment). `PVHCProblemWithoutBattery._find_violated_constraints`
        (src/analyzer.py:1320-1351) already hoists this exact computation out
        of its scenario loop; this mirrors that pattern.
        """
        M = self.matrices
        Mq = self.matrices_qinj

        M_pq_utility = M["M_p_utility"] + M["M_q_utility"]
        M_pq_customer = M["M_p_customer"] + M["M_q_customer"]
        M_pmq_utility = M["M_p_utility"] - M["M_q_utility"]
        M_pmq_customer = M["M_p_customer"] - M["M_q_customer"]
        M_pq_qinj = Mq["M_P_qinj"] + Mq["M_Q_qinj"]
        M_pmq_qinj = Mq["M_P_qinj"] - Mq["M_Q_qinj"]

        def pos_neg(M_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if not self.is_robust:
                return M_c, M_c
            return np.maximum(M_c, 0), np.minimum(M_c, 0)

        v_pos, v_neg = pos_neg(M["M_v_customer"])
        p_pos, p_neg = pos_neg(M["M_p_customer"])
        q_pos, q_neg = pos_neg(M["M_q_customer"])
        pq_pos, pq_neg = pos_neg(M_pq_customer)
        pmq_pos, pmq_neg = pos_neg(M_pmq_customer)

        return {
            "M_pq_utility": M_pq_utility, "M_pmq_utility": M_pmq_utility,
            "M_pq_qinj": M_pq_qinj, "M_pmq_qinj": M_pmq_qinj,
            "M_v_customer_pos": v_pos, "M_v_customer_neg": v_neg,
            "M_p_customer_pos": p_pos, "M_p_customer_neg": p_neg,
            "M_q_customer_pos": q_pos, "M_q_customer_neg": q_neg,
            "M_pq_customer_pos": pq_pos, "M_pq_customer_neg": pq_neg,
            "M_pmq_customer_pos": pmq_pos, "M_pmq_customer_neg": pmq_neg,
        }

    def _compute_qinj_impact_matrices(self, dev_info: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Impact of a unit REACTIVE power injection at each device bus (i.e. the
        "q_from_q"/"p_from_q" terms of the delta-to-wye transform). For a
        grounded (wye) connection there is no p/q cross-coupling at the bus:
        injecting Q affects voltages via X only, and P branch flow only below a
        delta winding, where C_tilde_S is complex. For a delta-connected device,
        injecting Q also couples into P branch flow.
        This is exactly the "q dispatch" half of
        `PVHCProblemWithBattery._compute_battery_impact_matrices`.
        """
        n_dev = len(dev_info)
        n_buses = self.network_data["n_buses"]
        M_v_q = np.zeros((n_buses * 3, n_dev))
        M_P_q = np.zeros((n_buses * 3, n_dev))
        M_Q_q = np.zeros((n_buses * 3, n_dev))

        if n_dev == 0:
            return {"M_v_qinj": M_v_q, "M_P_qinj": M_P_q, "M_Q_qinj": M_Q_q}

        R = self.network_data["R"]
        X = self.network_data["X"]
        C_tilde_S = self.network_data["C_tilde_S"]
        node2idx = self.network_data["node2idx"]

        for i, info in enumerate(dev_info):
            bus_name = info["bus"]
            phases = info["phases"]
            if bus_name not in node2idx:
                raise KeyError(f"Bus '{bus_name}' not found in node2idx")
            bus_idx = node2idx[bus_name]

            if phases[1] == 0:
                idx = (bus_idx - 1) * 3 + phases[0] - 1
                M_v_q[:, i] = np.real(1e-3 * X[:, idx])
                # P + jQ = C_tilde_S (j q): the cross term is nonzero only below a delta
                # winding, where C_tilde_S is complex.
                M_P_q[:, i] = -np.imag(C_tilde_S[:, idx])
                M_Q_q[:, i] = np.real(C_tilde_S[:, idx])
            else:
                phase_idx = ldf.STD_DELTA_ORDER.index(tuple(sorted(phases)))
                trans_vec = self._get_transformation_vector(info.get("connected_transformer", {}), phase_idx)
                idx0 = (bus_idx - 1) * 3

                a = np.real(trans_vec).reshape(-1)
                b = np.imag(trans_vec).reshape(-1)
                # s_wye = trans_vec * (p + jq)  =>  p_wye = a*p - b*q, q_wye = b*p + a*q
                p_from_q = -b
                q_from_q = a

                M_v_q[:, i] = 1e-3 * (R[:, idx0:idx0 + 3] @ p_from_q + X[:, idx0:idx0 + 3] @ q_from_q)
                # P + jQ = (C_tilde_S trans_vec) (j q)
                branch_pwr = C_tilde_S[:, idx0:idx0 + 3] @ trans_vec
                M_P_q[:, i] = -np.imag(branch_pwr)
                M_Q_q[:, i] = np.real(branch_pwr)

        return {
            "M_v_qinj": np.real(M_v_q),
            "M_P_qinj": np.real(M_P_q),
            "M_Q_qinj": np.real(M_Q_q),
        }

    def _get_nominal_voltage_squared(self) -> np.ndarray:
        N = self.network_data["n_buses"]
        C_tilde_0 = self.network_data["C_tilde_0"]

        v0_ref_squared = ldf.compute_v0_ref(self.v_root_nom * np.ones(3)).swapaxes(0, 1).reshape(9, 1)
        nominal_squared = (C_tilde_0.reshape(N, 3, 9) @ v0_ref_squared).reshape(N * 3)

        assert np.imag(nominal_squared).max() < 1e-6, "Nominal voltages should be real"
        return np.real(nominal_squared)

    def _compute_network_limits(self) -> Dict[str, np.ndarray]:
        nominal_voltage_sq = self._get_nominal_voltage_squared()
        voltage_upper = nominal_voltage_sq * (1 + self.vol_tol) ** 2
        voltage_lower = nominal_voltage_sq * (1 - self.lower_voltage_limit_ratio * self.vol_tol) ** 2

        n_buses = self.network_data["n_buses"]
        raw_ratings = self.network_data["branch_pwr_ratings"]
        processed_ratings = [r if (r is not None and r != 0) else 1e6 for r in raw_ratings[:n_buses]]
        S_ratings = np.repeat(processed_ratings, 3) * self.relaxation_factor

        return {
            "voltage_upper": voltage_upper,
            "voltage_lower": voltage_lower,
            "S_ratings": S_ratings,
            "sqrt2_S_ratings": np.sqrt(2) * S_ratings,
        }

    # --------------------------------------------------------------------------
    # Base-case (zero PV, zero q_vv) feasibility + adaptive relaxation
    # --------------------------------------------------------------------------
    def _check_basic_feasibility(self) -> bool:
        if self.base_case is None or self.limits is None:
            raise RuntimeError("compile() must populate base_case/limits before feasibility check.")

        violations = self._find_base_case_violations()
        if not violations:
            return True

        print(f"[Base feasibility] {len(violations)} violations detected at zero PV/Q_vv (load-only base case).")
        if self.adaptive_flag:
            self._relax_limits_adaptively(violations)
            print("[Adaptive relaxation] Limits updated. Retrying feasibility check...")
            violations = self._find_base_case_violations()
            if not violations:
                print("[Adaptive relaxation] Network is now feasible after limit relaxation.")
                return True
            print("[Adaptive relaxation] Still has violations after relaxation.")
        return False

    def _find_base_case_violations(self) -> List[Tuple[int, str, int]]:
        """At zero PV/q_vv the only nonzero terms are the load-only base case,
        so this only ever checks the 10 network constraint types."""
        violations: List[Tuple[int, str, int]] = []
        v_upper, v_lower = self.limits["voltage_upper"], self.limits["voltage_lower"]
        S, sqrt2S = self.limits["S_ratings"], self.limits["sqrt2_S_ratings"]

        for t in range(self._num_scenarios):
            base_v = self.base_case["voltage"][t]
            base_P = self.base_case["active_power"][t]
            base_Q = self.base_case["reactive_power"][t]

            for elem in np.where(base_v > v_upper + 1e-6)[0]:
                violations.append((t, "voltage_upper", int(elem)))
            for elem in np.where(base_v < v_lower - 1e-6)[0]:
                violations.append((t, "voltage_lower", int(elem)))
            for elem in np.where(base_P > S + 1e-6)[0]:
                violations.append((t, "active_flow_upper", int(elem)))
            for elem in np.where(base_P < -S - 1e-6)[0]:
                violations.append((t, "active_flow_lower", int(elem)))
            for elem in np.where(base_Q > S + 1e-6)[0]:
                violations.append((t, "reactive_flow_upper", int(elem)))
            for elem in np.where(base_Q < -S - 1e-6)[0]:
                violations.append((t, "reactive_flow_lower", int(elem)))

            base_pq = base_P + base_Q
            for elem in np.where(base_pq > sqrt2S + 1e-6)[0]:
                violations.append((t, "appr_flow_upper", int(elem)))
            for elem in np.where(base_pq < -sqrt2S - 1e-6)[0]:
                violations.append((t, "appr_flow_lower", int(elem)))

            base_pmq = base_P - base_Q
            for elem in np.where(base_pmq > sqrt2S + 1e-6)[0]:
                violations.append((t, "appr_diff_flow_upper", int(elem)))
            for elem in np.where(base_pmq < -sqrt2S - 1e-6)[0]:
                violations.append((t, "appr_diff_flow_lower", int(elem)))

        return violations

    def _relax_limits_adaptively(self, violations: List[Tuple[int, str, int]]) -> None:
        relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        voltage_upper_ratio: Dict[int, float] = {}
        voltage_lower_ratio: Dict[int, float] = {}
        s_ratings_ratio: Dict[int, float] = {}

        for (t, ctype, elem) in violations:
            base_v = self.base_case["voltage"][t]
            base_P = self.base_case["active_power"][t]
            base_Q = self.base_case["reactive_power"][t]
            is_upper = "upper" in ctype

            if "voltage" in ctype:
                lhs_val = base_v[elem]
                limit = self.limits["voltage_upper"][elem] if is_upper else self.limits["voltage_lower"][elem]
            elif "active_flow" in ctype:
                lhs_val = base_P[elem]
                limit = self.limits["S_ratings"][elem] if is_upper else -self.limits["S_ratings"][elem]
            elif "reactive_flow" in ctype:
                lhs_val = base_Q[elem]
                limit = self.limits["S_ratings"][elem] if is_upper else -self.limits["S_ratings"][elem]
            elif "appr_flow" in ctype and "diff" not in ctype:
                lhs_val = base_P[elem] + base_Q[elem]
                limit = self.limits["sqrt2_S_ratings"][elem] if is_upper else -self.limits["sqrt2_S_ratings"][elem]
            else:
                lhs_val = base_P[elem] - base_Q[elem]
                limit = self.limits["sqrt2_S_ratings"][elem] if is_upper else -self.limits["sqrt2_S_ratings"][elem]

            margin = lhs_val - limit if is_upper else limit - lhs_val
            ratio = 1 + relaxation_factor * margin / abs(limit)

            if "voltage" in ctype:
                if is_upper:
                    voltage_upper_ratio[elem] = max(voltage_upper_ratio.get(elem, 1.0), ratio)
                else:
                    voltage_lower_ratio[elem] = max(voltage_lower_ratio.get(elem, 1.0), ratio)
            else:
                s_ratings_ratio[elem] = max(s_ratings_ratio.get(elem, 1.0), ratio)

        for elem, ratio in voltage_upper_ratio.items():
            bus_idx = elem // 3
            for ph in range(3):
                idx = bus_idx * 3 + ph
                self.limits["voltage_upper"][idx] *= ratio

        for elem, ratio in voltage_lower_ratio.items():
            bus_idx = elem // 3
            for ph in range(3):
                idx = bus_idx * 3 + ph
                self.limits["voltage_lower"][idx] *= ratio

        for elem, ratio in s_ratings_ratio.items():
            bus_idx = elem // 3
            for ph in range(3):
                idx = bus_idx * 3 + ph
                self.limits["S_ratings"][idx] *= ratio
                self.limits["sqrt2_S_ratings"][idx] = np.sqrt(2) * self.limits["S_ratings"][idx]

    # --------------------------------------------------------------------------
    # Limit cache (same file format/location as src/analyzer.py classes)
    # --------------------------------------------------------------------------
    def _get_limits_cache_path(self) -> str:
        feeder_name = self.simulator.feeder_name.replace("/", "_")
        return os.path.join(self.limits_cache_dir, f"{feeder_name}_limits.pkl")

    def _save_limits_to_cache(self) -> None:
        if self.limits is None:
            return
        os.makedirs(self.limits_cache_dir, exist_ok=True)
        cache_data = {
            "limits": self.limits,
            "feeder_name": self.simulator.feeder_name,
            "vol_tol": self.vol_tol,
            "lower_voltage_limit_ratio": self.lower_voltage_limit_ratio,
            "relaxation_factor": self.relaxation_factor,
            "v_root_nom": self.v_root_nom,
            "n_buses": self._n_buses,
        }
        with open(self._get_limits_cache_path(), "wb") as f:
            pickle.dump(cache_data, f)

    def _load_limits_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        cache_path = self._get_limits_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            if (cache_data.get("feeder_name") != self.simulator.feeder_name or
                    cache_data.get("n_buses") != self._n_buses or
                    abs(cache_data.get("vol_tol", -1) - self.vol_tol) > 1e-9 or
                    abs(cache_data.get("lower_voltage_limit_ratio", -1) - self.lower_voltage_limit_ratio) > 1e-9 or
                    abs(cache_data.get("relaxation_factor", -1) - self.relaxation_factor) > 1e-9 or
                    abs(cache_data.get("v_root_nom", -1) - self.v_root_nom) > 1e-9):
                print("[Limit Cache] Cached limits configuration mismatch, recomputing")
                return None
            return cache_data.get("limits")
        except Exception as e:
            print(f"[Limit Cache] Error loading cached limits: {e}")
            return None
