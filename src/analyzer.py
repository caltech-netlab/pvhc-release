"""
PVHC Optimization Framework — Paper Section III (RSP formulation) and Algorithm 1

Provides the convex scenario program (RSP) for PV Hosting Capacity (PVHC)
optimization and the Clarkson-style iterative solver that realizes Algorithm 1.

Paper mapping:
  Eq. (10)-(12)  — RSP: maximize Σζ̄_c s.t. voltage & branch-flow constraints
                    hold for all ζ_c ∈ [0, ζ̄_c] and all sampled scenarios ω_k
  Proposition III.1 — LP reformulation: replace sup_{ζ_c} with A_{c,+}(ω)ζ̄_c
                    using the support function of the box [0, ζ̄_c]
                    → `_compute_robust_lhs_cvxpy` (np.maximum/minimum trick)
  Eq. (7)        — Voltage constraint: v_min ≤ v̂^(k) ≤ v_max
  Eq. (9a,b)     — Branch active/reactive power limits: |P^(k)|, |Q^(k)| ≤ S_max
  Eq. (9c,d)     — Linearized apparent power limits: |P±Q| ≤ √2·S_max
                    → `_build_scenario_constraints_cvxpy` (10 constraint types)
  Eq. (16)       — Fairness objective: Σζ̄_c + γ·Σlog(ζ̄_c + η)
                    → `_create_objective`
  Algorithm 1    — Iterative scenario addition (Clarkson-style):
                    solve subproblem → find violated → grow active set → repeat
                    → `clarkson_solver`
  Definition III.1 — Support constraints: scenarios with non-zero dual variable
                    → `_extract_support_from_duals`
  Appendix C     — Battery extension: `PVHCProblemWithBattery` (further below)

Classes:
  ClarksonProblem          — Abstract interface (compile / solve_subproblem /
                             find_violated_indices)
  PVHCProblemWithoutBattery — Concrete PVHC without storage (paper Sections III-IV)
  PVHCProblemWithBattery   — Battery extension (paper Appendix C, Table II)

Top-level function:
  clarkson_solver(problem, ...) — Runs Algorithm 1 and returns optimal capacities
                                  and support index set used by
                                  `update_num_scenarios` (src/utils.py).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import pickle
import os

from .simulator import Simulator
from .constants import *
from .solver_utils import solve_with_fallback
import three_phase_lindistflow as ldf


# A scenario is a support scenario (Definition III.1) when the duals of its constraints are not
# zero. Interior-point solvers (Clarabel, MOSEK) leave every inactive constraint a small residual
# dual (about 1e-7 to 1e-6 here), so "not zero" has to be judged against the duals that are.
# Each scenario is scored by its largest |dual| times the norm of that constraint's row, which
# puts voltage (kW/kV^2) and flow (kW/kW) constraints on one scale and cancels the fairness
# weight. It is support if one of its constraints is active (slack within SUPPORT_SLACK_RTOL of
# the constraint's size) and its score is at least SUPPORT_DUAL_RTOL times the largest score of
# such scenarios.
SUPPORT_DUAL_RTOL = 1e-6
SUPPORT_SLACK_RTOL = 1e-4


def support_from_duals(blocks: Dict[int, List[Tuple[Any, np.ndarray]]],
                       rtol: float = SUPPORT_DUAL_RTOL,
                       slack_rtol: float = SUPPORT_SLACK_RTOL) -> Optional[Set[int]]:
    """
    Support scenarios from the duals of their constraints (see SUPPORT_DUAL_RTOL).

    Args:
        blocks: {scenario index: [(CVXPY inequality, norms of its coefficient rows), ...]}
            for every scenario in the subproblem, after it has been solved.

    Returns:
        The support scenarios (possibly none, e.g. when every site sits at its capacity
        limit), or None if the solver returned no duals at all.
    """
    scores, slacks = {}, {}
    for scenario, items in blocks.items():
        best, min_slack, has_duals = 0.0, np.inf, False
        for constraint, row_norms in items:
            dual = constraint.dual_value
            if dual is None:
                continue
            has_duals = True
            best = max(best, float(np.max(np.abs(dual) * row_norms)))
            lhs, rhs = constraint.args
            size = np.maximum(np.maximum(np.abs(lhs.value), np.abs(rhs.value)), 1e-9)
            min_slack = min(min_slack, float(np.min(-np.asarray(constraint.expr.value) / size)))
        if has_duals:
            scores[scenario], slacks[scenario] = best, min_slack
    if not scores:
        return None
    # Duals are judged against those of scenarios with an active constraint: the duals of
    # inactive constraints are residuals (complementary slackness).
    active = [s for s in scores if slacks[s] <= slack_rtol]
    top = max((scores[s] for s in active), default=0.0)
    if top == 0:
        return set()  # no active scenario constraint carries a dual, e.g. every site at its cap
    threshold = rtol * top
    support = {s for s in active if scores[s] >= threshold}
    # Say so when the call is close: an active scenario just below the threshold, or a
    # non-negligible dual on a constraint that is not active (an inaccurate solution).
    below = [scores[s] for s in active if s not in support and scores[s] > 0]
    if support and below and min(scores[s] for s in support) < 1e3 * max(below):
        print(f"  [support] {len(support)} support scenario(s) with dual scores down to "
              f"{min(scores[s] for s in support):.2e}, but an active non-support one scores "
              f"{max(below):.2e} (threshold {threshold:.2e}): the support count may be unreliable.")
    inactive = [s for s in scores if s not in active and scores[s] >= threshold]
    if inactive:
        print(f"  [support] {len(inactive)} scenario(s) have non-negligible duals but no active "
              f"constraint (smallest relative slack {min(slacks[s] for s in inactive):.1e}); "
              f"not counted as support. The solution may be inaccurate.")
    return support



# ==============================================================================
# ABSTRACT BASE CLASS
# ==============================================================================

class ClarksonProblem(ABC):
    """
    Abstract base class for problems that can be solved by Clarkson-like algorithms.
    
    This class defines the interface for convex scenario programs where the goal is
    to find an optimal solution that satisfies constraints from all scenarios.
    
    In scenario-based mode: each scenario i corresponds to a set of constraints
        A_i @ x <= b_i, where A_i can be a matrix.
    In constraint-based mode: we treat each row of each A_i as a separate constraint.
    """
    
    @property
    @abstractmethod
    def num_scenarios(self) -> int:
        """Total number of scenarios in the problem."""
        pass
    
    @property
    @abstractmethod
    def num_constraints_per_scenario(self) -> int:
        """Number of constraint rows per scenario (for constraint-based mode)."""
        pass
    
    @property
    @abstractmethod
    def num_variables(self) -> int:
        """Number of decision variables."""
        pass
    
    @abstractmethod
    def compile(self) -> None:
        """
        Compile/precompute all data and matrices needed for the optimization problem.
        Must be called before solve_subproblem or find_violated_indices.
        """
        pass
    
    @abstractmethod
    def solve_subproblem(
        self,
        indices: List[int],
        is_scenario_based: bool = True,
        solver: Optional[Any] = cp.HIGHS,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        """
        Solve optimization restricted to a subset of scenarios/constraints.
        
        Args:
            indices: List of scenario indices (if scenario-based) or 
                     constraint indices (if not scenario-based) to include
            is_scenario_based: If True, treat indices as scenario indices.
                              If False, treat indices as individual constraint indices.
            solver: CVXPY solver to use
            verbose: Print solver output
            
        Returns:
            Tuple of:
                - optimal_solution: Optimal x* or None if infeasible
                - objective_value: Optimal objective value
                - support_indices: Set of support scenario/constraint indices
        """
        pass
    
    @abstractmethod
    def find_violated_indices(
        self,
        solution: np.ndarray,
        tolerance: float = 1e-6,
        is_scenario_based: bool = True,
        exclude_indices: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Find scenarios/constraints violated by the given solution.
        
        Args:
            solution: Solution vector x*
            tolerance: Violation tolerance
            is_scenario_based: If True, return violated scenario indices.
                              If False, return violated constraint indices.
            exclude_indices: Set of indices to exclude from checking
            
        Returns:
            Set of violated scenario/constraint indices
        """
        pass


# ==============================================================================
# PVHC PROBLEM WITHOUT BATTERY
# ==============================================================================

class PVHCProblemWithoutBattery(ClarksonProblem):
    """
    PV Hosting Capacity problem without battery storage.
    
    Maximizes total PV capacity subject to voltage and power flow constraints
    across multiple scenarios (time periods with different load/generation).
    
    The optimization problem is:
        maximize    sum(x)   (or with fairness: sum(x) + factor*sum(log(x+eps)))
        subject to  x <= capacity_limits
                    voltage_constraints(x, scenario_i) for all i
                    power_flow_constraints(x, scenario_i) for all i
    
    Each scenario corresponds to one time period with specific load and PV profiles.
    """
    
    # Class constants
    # Symmetric +-VOL_TOL voltage band (e.g. ANSI C84.1 Range A is +-5%) --
    # matches the upper bound; no evidence a wider lower-side allowance is
    # needed (tested empirically: the real feeder's base-case voltage never
    # approaches even the symmetric band, let alone a wider one).
    DEFAULT_LOWER_VOLTAGE_RATIO = 1.0
    DEFAULT_RELAXATION_FACTOR = 1.5
    DEFAULT_FAIRNESS_FACTOR = 100
    NUM_CONSTRAINT_TYPES = 10  # 2 voltage + 4 power + 4 approximated
    
    def __init__(
        self,
        simulator: Simulator,
        utility_pv_info: List[Dict[str, Any]],
        customer_pv_info: List[Dict[str, Any]],
        load_info: List[Dict[str, Any]],
        is_robust: bool = True,
        with_fairness: bool = False,
        fairness_factor: float = DEFAULT_FAIRNESS_FACTOR,
        load_cached_limits: bool = False
    ):
        """
        Initialize PVHC problem.

        Args:
            simulator: Simulator instance for power flow calculations
            utility_pv_info: List of dicts with keys "bus", "phases", "capacity"
            customer_pv_info: List of dicts with keys "bus", "phases", "capacity"
            load_info: List of dicts with keys "bus" and "phases"
            is_robust: Use robust optimization formulation
            with_fairness: Include fairness term in objective
            fairness_factor: Weight for fairness term
            load_cached_limits: If True, load previously saved limits instead of recalculating
        """
        self.simulator = simulator
        self.utility_pv_info = utility_pv_info or []
        self.customer_pv_info = customer_pv_info or []
        self.load_info = load_info
        self.is_robust = is_robust
        self.with_fairness = with_fairness
        self.fairness_factor = fairness_factor
        
        # Configuration
        self.vol_tol = VOL_TOL
        self.lower_voltage_limit_ratio = self.DEFAULT_LOWER_VOLTAGE_RATIO
        self.relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        self.v_root_nom = self._get_feeder_nominal_voltage()
        
        # PV counts
        self.num_utility_pv = len(self.utility_pv_info)
        self.num_customer_pv = len(self.customer_pv_info)
        self._num_variables = self.num_utility_pv + self.num_customer_pv

        # Get network data
        self.network_data = self.simulator.get_network_data()
        self._n_buses = self.network_data["n_buses"]
        self._num_elements = self._n_buses * 3

        # Data storage - will be populated by compile()
        self._is_compiled = False
        self.matrices: Optional[Dict] = None
        self.limits: Optional[Dict] = None
        self.base_case: Optional[Dict] = None
        self.pv_profiles_processed: Optional[np.ndarray] = None
        self.capacity_limits: Optional[np.ndarray] = None

        # Will be set when scenarios are added
        self._num_scenarios = 0
        self._filter_zero_pv = True  # decision made by add_scenarios(); compile() must
        # respect it rather than unconditionally re-filtering (see add_scenarios/compile)

        # Adaptive limit relaxation flag
        self.adaptive_flag = True

        # Limit caching configuration
        self.load_cached_limits = load_cached_limits
        self.limits_cache_dir = "temp/cache/limits"

    def _get_feeder_nominal_voltage(self) -> float:
        """Get nominal voltage for the current feeder."""
        if self.simulator.feeder_name not in FEEDERS:
            raise ValueError(f"Feeder name {self.simulator.feeder_name} not recognized.")
        return FEEDERS[self.simulator.feeder_name]["v_root_nom"]
    
    # --------------------------------------------------------------------------
    # ClarksonProblem interface implementation
    # --------------------------------------------------------------------------
    
    @property
    def num_scenarios(self) -> int:
        return self._num_scenarios
    
    @property
    def num_constraints_per_scenario(self) -> int:
        # Each scenario has NUM_CONSTRAINT_TYPES constraint vectors, 
        # each of size n_buses * 3
        return self.NUM_CONSTRAINT_TYPES * self._num_elements
    
    @property
    def num_variables(self) -> int:
        return self._num_variables
    
    def compile(
        self,
    ) -> None:
        """
        Precompute all matrices and data needed for optimization.
        
        Args:
            load_profiles: Tuple of (active_power, reactive_power) arrays, shape (T, N_loads)
            pv_profiles: PV generation profiles, shape (T,) or (T, N_pv)
            V0_profiles: Root bus voltage profiles, shape (T, 3)
        """
        if self._load_profiles is None or self._pv_profiles is None or self._V0_profiles is None:
            raise RuntimeError("Scenarios not added. Call add_scenarios() before compile().")
        
        if not self._scenario_is_processed:
            # Filter scenarios with zero PV generation -- but only if add_scenarios()
            # was asked to (self._filter_zero_pv); otherwise this would silently
            # override a caller's explicit filter_zero_pv=False (e.g. CVaR baselines
            # in scripts/run_method_comparison.py, which need the full scenario
            # population -- including zero-PV ones -- for a correctly-calibrated
            # sample-based risk average).
            if self._filter_zero_pv:
                valid_idx = self._get_nonzero_pv_indices(self._pv_profiles)
                self._pv_profiles = self._pv_profiles[valid_idx]
                self._V0_profiles = self._V0_profiles[valid_idx]
                self._load_profiles = (self._load_profiles[0][valid_idx], self._load_profiles[1][valid_idx])
                self._num_scenarios = self._pv_profiles.shape[0]
            self._scenario_is_processed = True

            # Compute base case (without PV)
            base_v, base_P, base_Q, _, _, _ = self.simulator.simulate(
                load_profiles=self._load_profiles,
                pv_profiles=self._pv_profiles,
                V0_profiles=self._V0_profiles,
                load_info=self.load_info,
                PV_info=None
            )
            self.base_case = {
                'voltage': np.square(base_v),
                'active_power': base_P,
                'reactive_power': base_Q
            }

            # Process PV profiles
            self.pv_profiles_processed = self._preprocess_pv_profiles(self._pv_profiles)
        
        # Compute PV impact matrices
        self.matrices = self._compute_pv_impact_matrices()
        
        # Compute or load network limits
        if self.load_cached_limits:
            loaded_limits = self._load_limits_from_cache()
            if loaded_limits is not None:
                self.limits = loaded_limits
                print(f"[Limit Cache] Loaded cached limits for {self.simulator.feeder_name}")
                feasible = self._check_basic_feasibility()
            else:
                print(f"[Limit Cache] No cached limits found, computing new limits")
                self.limits = self._compute_network_limits()
                # Check basic feasibility and potentially relax
                feasible = self._check_basic_feasibility()
                # Save the (possibly relaxed) limits
                self._save_limits_to_cache()
        else:
            self.limits = self._compute_network_limits()
            # Check basic feasibility and potentially relax
            feasible = self._check_basic_feasibility()
            self._save_limits_to_cache()

        if not feasible:
            raise RuntimeError(
                "Network remains infeasible under existing customer load (zero PV) even "
                "after adaptive limit relaxation. This means some equipment is loaded far "
                "beyond what a modest safety margin can account for -- inspect the "
                "[Base feasibility] messages above rather than solving with limits that "
                "don't reflect a real operating point."
            )
        
        # Capacity limits
        self.capacity_limits = np.array([
            info["capacity"] for info in (self.utility_pv_info + self.customer_pv_info)
        ])
             
        self._is_compiled = True
    
    def solve_subproblem(
        self,
        indices: List[int],
        is_scenario_based: bool = True,
        solver: Optional[Any] = cp.HIGHS,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        """
        Solve optimization restricted to subset of scenarios/constraints.
        """
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")
        
        if is_scenario_based:
            return self._solve_scenario_based(indices, solver, verbose)
        else:
            return self._solve_constraint_based(indices, solver, verbose)
    
    def find_violated_indices(
        self,
        solution: np.ndarray,
        tolerance: float = 1e-6,
        is_scenario_based: bool = True,
        exclude_indices: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Find scenarios/constraints violated by the solution.
        """
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")
        
        exclude = exclude_indices or set()
        
        if is_scenario_based:
            return self._find_violated_scenarios(solution, tolerance, exclude)
        else:
            return self._find_violated_constraints(solution, tolerance, exclude)
        
    # --------------------------------------------------------------------------
    # Public Modification Methods
    # --------------------------------------------------------------------------
    def add_scenarios(
        self,
        load_profiles: Tuple[np.ndarray, np.ndarray],
        pv_profiles: np.ndarray,
        V0_profiles: np.ndarray,
        filter_zero_pv: bool = True
    ) -> None:
        """
        Add scenarios to the problem. This must be called before compile(). This will overwrite the existing scenarios if called multiple times.
        
        Args:
            load_profiles: Tuple of (active_power, reactive_power) arrays, shape (T, N_loads)
            pv_profiles: PV generation profiles, shape (T,) or (T, N_pv)
            V0_profiles: Root bus voltage profiles, shape (T, 3)
            filter_zero_pv: If True, filter out scenarios with zero PV generation
        """
        if filter_zero_pv:
            valid_idx = self._get_nonzero_pv_indices(pv_profiles)
            pv_profiles = pv_profiles[valid_idx]
            V0_profiles = V0_profiles[valid_idx]
            load_profiles = (load_profiles[0][valid_idx], load_profiles[1][valid_idx])

        self._num_scenarios = pv_profiles.shape[0]
        self._filter_zero_pv = filter_zero_pv  # compile() must not override this choice

        # Store profiles for later compilation
        self._load_profiles = load_profiles
        self._pv_profiles = pv_profiles
        self._V0_profiles = V0_profiles

        self._scenario_is_processed = False

    def modify_problem_parameters(
        self,
        utility_pv_info: List[Dict[str, Any]] = None,
        customer_pv_info: List[Dict[str, Any]] = None,
        load_info: List[Dict[str, Any]] = None,
        is_robust: bool = None,
        with_fairness: bool = None,
        fairness_factor: float = None
    ) -> None:
        """
        Modify problem parameters. Only recompiles if structural parameters change.

        Args:
            utility_pv_info: New utility PV information (requires recompile)
            customer_pv_info: New customer PV information (requires recompile)
            load_info: New load information (requires recompile)
            is_robust: New robust optimization flag (requires recompile)
            with_fairness: New fairness flag (no recompile needed)
            fairness_factor: New fairness weight (no recompile needed)
        """
        needs_recompile = False

        # Check and update parameters that require recompilation
        if utility_pv_info is not None and utility_pv_info != self.utility_pv_info:
            self.utility_pv_info = utility_pv_info
            self.num_utility_pv = len(self.utility_pv_info)
            self._num_variables = self.num_utility_pv + self.num_customer_pv
            needs_recompile = True

        if customer_pv_info is not None and customer_pv_info != self.customer_pv_info:
            self.customer_pv_info = customer_pv_info
            self.num_customer_pv = len(self.customer_pv_info)
            self._num_variables = self.num_utility_pv + self.num_customer_pv
            needs_recompile = True

        if load_info is not None and load_info != self.load_info:
            self.load_info = load_info
            needs_recompile = True

        if is_robust is not None and is_robust != self.is_robust:
            self.is_robust = is_robust
            needs_recompile = True
        
        # Update parameters that don't require recompilation
        if with_fairness is not None:
            self.with_fairness = with_fairness
        
        if fairness_factor is not None:
            self.fairness_factor = fairness_factor
        
        # Mark as needing recompilation if necessary
        if needs_recompile and self._is_compiled:
            self._is_compiled = False
            # Clear cached data
            self.network_data = None
            self.matrices = None
            self.limits = None
            self.base_case = None
            self.capacity_limits = None

    # --------------------------------------------------------------------------
    # Public Accessor Methods
    # --------------------------------------------------------------------------
    def get_nominal_voltage(self, squared: bool = False) -> np.ndarray:
        if squared:
            return self._get_nominal_voltage_squared()
        else:
            return np.sqrt(self._get_nominal_voltage_squared())
        
    def get_network_limits(self) -> Dict[str, np.ndarray]:
        if not self._is_compiled:
            return self._compute_network_limits()
        return self.limits.copy()
    
    def _get_limits_cache_path(self) -> str:
        """Get the file path for cached limits."""
        feeder_name = self.simulator.feeder_name.replace('/', '_')
        filename = f"{feeder_name}_limits.pkl"
        return os.path.join(self.limits_cache_dir, filename)
    
    def _save_limits_to_cache(self) -> None:
        """Save computed limits to cache file."""
        if self.limits is None:
            return
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.limits_cache_dir, exist_ok=True)
        
        cache_path = self._get_limits_cache_path()
        
        # Save limits with metadata
        cache_data = {
            'limits': self.limits,
            'feeder_name': self.simulator.feeder_name,
            'vol_tol': self.vol_tol,
            'lower_voltage_limit_ratio': self.lower_voltage_limit_ratio,
            'relaxation_factor': self.relaxation_factor,
            'v_root_nom': self.v_root_nom,
            'n_buses': self._n_buses
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"[Limit Cache] Saved limits to {cache_path}")

    def _load_limits_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        """Load limits from cache file if it exists and matches current configuration."""
        cache_path = self._get_limits_cache_path()

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate that cached data matches current configuration
            if (cache_data['feeder_name'] != self.simulator.feeder_name or
                cache_data['n_buses'] != self._n_buses or
                abs(cache_data['vol_tol'] - self.vol_tol) > 1e-9 or
                abs(cache_data['lower_voltage_limit_ratio'] - self.lower_voltage_limit_ratio) > 1e-9 or
                abs(cache_data['relaxation_factor'] - self.relaxation_factor) > 1e-9 or
                abs(cache_data['v_root_nom'] - self.v_root_nom) > 1e-9):
                print(f"[Limit Cache] Cached limits configuration mismatch, recomputing")
                return None
            
            return cache_data['limits']
        
        except Exception as e:
            print(f"[Limit Cache] Error loading cached limits: {e}")
            return None
    

    # --------------------------------------------------------------------------
    # Debugging Line Capacity
    # --------------------------------------------------------------------------
    # Use the self.base_case to check feasibility when no PV is installed
    def _check_basic_feasibility(self, max_rounds: int = 10) -> bool:
        """
        Validate that the network is feasible with existing customer load alone
        (zero PV installed) -- i.e. that native peak demand doesn't already
        exceed real equipment ratings before any PV-hosting decision is made.

        This CAN legitimately happen on real feeders (a specific transformer or
        line already tight or over its nameplate rating under peak native load,
        e.g. due to load growth since it was sized) -- it is not something a
        PV-hosting-capacity study should try to "fix"; that pre-existing
        condition exists independent of any PV decision. When it does happen,
        `_relax_limits_adaptively` raises the *effective* operating limit for
        just the affected element(s) to the observed native peak (plus a small
        margin) so the PV decision is evaluated on top of that reality, rather
        than making the whole optimization problem infeasible over something
        PV additions don't cause and can't fix.

        Iterates relax-and-recheck (each round can only orphan a strictly
        smaller violated set, since relaxing a limit never re-introduces a
        violation elsewhere) until fully feasible or `max_rounds` is hit.
        """
        if self.base_case is None or self.limits is None or self.matrices is None:
            raise RuntimeError("Problem data incomplete; compile() must populate base_case and limits before feasibility check.")

        zero_solution = np.zeros(self._num_variables, dtype=float)

        for round_idx in range(max_rounds):
            violated_constraints = self._find_violated_constraints(
                zero_solution, tolerance=1e-6, exclude=set()
            )
            if not violated_constraints:
                if round_idx > 0:
                    print(f"[Adaptive relaxation] Network is now feasible after {round_idx} round(s) of relaxation.")
                return True

            print(f"[Base feasibility] Round {round_idx + 1}/{max_rounds}: "
                  f"{len(violated_constraints)} violation(s) detected with zero PV "
                  f"(native customer load alone). First few (of possibly many, scenario x row pairs):")
            for idx in sorted(violated_constraints)[:5]:
                t, ctype, elem = self._decode_constraint_index(idx)
                desc = self._describe_violation_at_zero_pv(t, ctype, elem)
                direction = "upper" if desc["is_upper"] else "lower"
                print(
                    f"  scenario={t}, constraint={ctype}, row={elem}, "
                    f"value={desc['lhs']:.6f}, limit={desc['limit']:.6f}, "
                    f"margin={desc['margin']:.6f} (relax {direction} bound at row {elem})"
                )
            if len(violated_constraints) > 5:
                print(f"  ... and {len(violated_constraints) - 5} more.")

            if not self.adaptive_flag:
                return False

            self._relax_limits_adaptively(violated_constraints)

        print(f"[Adaptive relaxation] Still has violations after {max_rounds} round(s); giving up.")
        return False

    def _describe_violation_at_zero_pv(self, t: int, ctype: str, elem: int) -> Dict[str, Any]:
        """Return lhs, limit, margin, and direction for a violated constraint at zero PV."""
        # Base values with zero PV injections
        base_v = self.base_case['voltage'][t]
        base_P = self.base_case['active_power'][t]
        base_Q = self.base_case['reactive_power'][t]

        if 'voltage' in ctype:
            lhs_val = base_v[elem]
            limit = self.limits['voltage_upper'][elem] if 'upper' in ctype else self.limits['voltage_lower'][elem]
            is_upper = 'upper' in ctype
        elif 'active_flow' in ctype:
            lhs_val = base_P[elem]
            limit = self.limits['S_ratings'][elem] if 'upper' in ctype else -self.limits['S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'reactive_flow' in ctype:
            lhs_val = base_Q[elem]
            limit = self.limits['S_ratings'][elem] if 'upper' in ctype else -self.limits['S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'appr_flow' in ctype and 'diff' not in ctype:
            lhs_val = base_P[elem] + base_Q[elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if 'upper' in ctype else -self.limits['sqrt2_S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'appr_diff_flow' in ctype:
            lhs_val = base_P[elem] - base_Q[elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if 'upper' in ctype else -self.limits['sqrt2_S_ratings'][elem]
            is_upper = 'upper' in ctype
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")

        margin = lhs_val - limit if is_upper else limit - lhs_val

        return {
            'lhs': float(lhs_val),
            'limit': float(limit),
            'margin': float(margin),
            'is_upper': is_upper,
        }

    def _relax_limits_adaptively(self, violated_constraints: Set[int]) -> None:
        """
        Relax limits adaptively based on maximum relaxation factors.
        
        For each constraint type and element, compute the relaxation factor needed
        to satisfy the violation: ratio = (1 + factor * margin / old_limit).
        Then take the maximum relaxation factor across all constraint types for each element.
        Since this is a three-phase system, the relaxation applies to all three phases 
        of each bus (elements sharing the same bus).
        
        Importantly: S_ratings and sqrt2_S_ratings are related (sqrt2_S_ratings = sqrt(2)*S_ratings).
        By using relaxation factors instead of margins, we ensure that the same relaxation
        ratio is applied to S_ratings, maintaining the sqrt(2) relationship automatically.
        
        Args:
            violated_constraints: Set of violated constraint indices
        """
        relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        
        # Record maximum relaxation ratio for each element by limit type
        voltage_upper_ratio = {}  # {element: max_ratio}
        voltage_lower_ratio = {}  # {element: max_ratio}
        s_ratings_ratio = {}      # {element: max_ratio}
        
        for idx in violated_constraints:
            t, ctype, elem = self._decode_constraint_index(idx)
            desc = self._describe_violation_at_zero_pv(t, ctype, elem)
            margin = desc['margin']
            limit = desc['limit']

            # Compute relaxation ratio for this violation
            ratio = 1 + relaxation_factor * margin / abs(limit)

            # Key by bus (not the raw per-phase element index): a violation on
            # any one phase relaxes all 3 phases of that bus together, so two
            # violations on different phases of the SAME bus must land in the
            # SAME dict entry (via max()) rather than each independently
            # triggering their own full 3-phase relaxation pass below -- the
            # latter would silently double- (or triple-) relax that bus.
            bus_idx = elem // 3

            if 'voltage' in ctype:
                if 'upper' in ctype:
                    voltage_upper_ratio[bus_idx] = max(voltage_upper_ratio.get(bus_idx, 1.0), ratio)
                else:
                    voltage_lower_ratio[bus_idx] = max(voltage_lower_ratio.get(bus_idx, 1.0), ratio)
            else:
                # All flow-based constraints use S_ratings or sqrt2_S_ratings
                # Record the max ratio - will apply to S_ratings
                s_ratings_ratio[bus_idx] = max(s_ratings_ratio.get(bus_idx, 1.0), ratio)
        
        # Apply voltage relaxations (print only a few examples -- with thousands of
        # scenarios this can touch every bus, and printing all of them just floods
        # the log without adding information beyond the summary count).
        _MAX_RELAX_PRINTS = 5

        n_printed = 0
        for bus_idx, ratio in voltage_upper_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_limit = self.limits['voltage_upper'][idx]
                # Upper bound: multiplying by ratio > 1 raises it, i.e. relaxes it.
                new_limit = old_limit * ratio
                self.limits['voltage_upper'][idx] = new_limit
                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed voltage_upper[{idx}]: {old_limit:.6f} -> {new_limit:.6f} (ratio={ratio:.4f})")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(voltage_upper_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... voltage_upper relaxed for {len(voltage_upper_ratio) * 3} element(s) total.")

        n_printed = 0
        for bus_idx, ratio in voltage_lower_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_limit = self.limits['voltage_lower'][idx]
                # Lower bound: relaxing it means making it MORE permissive, i.e.
                # LOWER, not higher -- dividing by ratio > 1 decreases it. (Multiplying,
                # as with the upper bound / S_ratings above, would tighten a lower
                # bound instead of relaxing it, and could never converge to feasible.)
                new_limit = old_limit / ratio
                self.limits['voltage_lower'][idx] = new_limit
                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed voltage_lower[{idx}]: {old_limit:.6f} -> {new_limit:.6f} (ratio={ratio:.4f})")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(voltage_lower_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... voltage_lower relaxed for {len(voltage_lower_ratio) * 3} element(s) total.")

        # Apply S_ratings relaxations (and maintain sqrt2_S_ratings = sqrt(2) * S_ratings)
        n_printed = 0
        for bus_idx, ratio in s_ratings_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_s_limit = self.limits['S_ratings'][idx]
                new_s_limit = old_s_limit * ratio
                self.limits['S_ratings'][idx] = new_s_limit

                # Maintain sqrt2_S_ratings = sqrt(2) * S_ratings
                old_sqrt2_limit = self.limits['sqrt2_S_ratings'][idx]
                new_sqrt2_limit = np.sqrt(2) * new_s_limit
                self.limits['sqrt2_S_ratings'][idx] = new_sqrt2_limit

                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed S_ratings[{idx}]: {old_s_limit:.6f} -> {new_s_limit:.6f} (ratio={ratio:.4f})")
                    print(f"    Relaxed sqrt2_S_ratings[{idx}]: {old_sqrt2_limit:.6f} -> {new_sqrt2_limit:.6f}")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(s_ratings_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... S_ratings relaxed for {len(s_ratings_ratio) * 3} element(s) total.")

    
    # --------------------------------------------------------------------------
    # Scenario-based solving
    # --------------------------------------------------------------------------
    
    def _solve_scenario_based(
        self,
        scenario_indices: List[int],
        solver: Any,
        verbose: bool
    ) -> Tuple[Optional[np.ndarray], float, Set[int], Optional[np.ndarray]]:
        """
        Solve with each scenario as one constraint block.
        Allows extraction of support via dual variables.
        """
        # Decision variables
        x = cp.Variable(self._num_variables, nonneg=True)
        
        # Objective
        objective = self._create_objective(x)
        
        # Constraints
        constraints = []
        
        # Capacity bounds
        constraints.append(x <= self.capacity_limits)
        
        # Scenario constraints - one constraint object per type per scenario
        for t in scenario_indices:
            scenario_constraints = self._build_scenario_constraints_cvxpy(x, t)
            constraints.extend(scenario_constraints)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        solve_with_fallback(problem, solver, verbose,
                            label=f"RSP subproblem ({len(scenario_indices)} scenarios)")

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"  [WARN] Subproblem status={problem.status} (infeasible/unbounded) "
                  f"for {len(scenario_indices)} active scenario(s).")
            return None, float('-inf'), None, None, None

        # Extract support indices via dual variables
        support_indices = self._extract_support_from_duals(constraints, scenario_indices)
        
        # Extract dual variable values
        # Shape: (num_scenarios, num_constraint_types, num_elements)
        dual_var = np.zeros((len(scenario_indices), self.NUM_CONSTRAINT_TYPES, self._num_elements))
        
        # First constraint is capacity bound, skip it
        num_non_scenario = 1
        for i, scenario_idx in enumerate(scenario_indices):
            for j in range(self.NUM_CONSTRAINT_TYPES):
                constraint_idx = num_non_scenario + i * self.NUM_CONSTRAINT_TYPES + j
                if constraint_idx < len(constraints):
                    dual_val = constraints[constraint_idx].dual_value
                    if dual_val is not None:
                        dual_var[i, j, :] = dual_val
                    else:
                        print(f"Warning: Dual value for scenario {scenario_idx}, constraint type {j} is None.")
        
        return x.value, problem.value, support_indices, dual_var, scenario_indices
    
    def _build_scenario_constraints_cvxpy(
        self,
        x: cp.Variable,
        t: int
    ) -> List:
        """
        Build CVXPY constraints for scenario t — implements Eq. (7) and (9).

        Returns exactly NUM_CONSTRAINT_TYPES=10 CVXPY constraint objects:
          [0] voltage upper   v̂^(k) ≤ v_max²             Eq. (7)
          [1] voltage lower   v̂^(k) ≥ v_min²             Eq. (7)
          [2] P upper         P^(k)  ≤  S_max             Eq. (9a)
          [3] P lower         P^(k)  ≥ -S_max             Eq. (9a)
          [4] Q upper         Q^(k)  ≤  S_max             Eq. (9b)
          [5] Q lower         Q^(k)  ≥ -S_max             Eq. (9b)
          [6] P+Q upper       P+Q    ≤  √2·S_max          Eq. (9c)
          [7] P+Q lower       P+Q    ≥ -√2·S_max          Eq. (9c)
          [8] P-Q upper       P-Q    ≤  √2·S_max          Eq. (9d)
          [9] P-Q lower       P-Q    ≥ -√2·S_max          Eq. (9d)

        Each LHS is the robust worst-case over ζ_c ∈ [0, ζ̄_c] (Proposition III.1),
        computed via `_compute_robust_lhs_cvxpy`.
        Dual variables on these 10 constraints are used to identify support
        scenarios (Definition III.1) and to compute dual sensitivities
        (Appendix B, used in `dual_analysis.py`).
        """
        pv_t = self.pv_profiles_processed[t]
        
        # Scale decision variables by PV profile
        # Handle empty arrays (when num_utility_pv=0 or num_customer_pv=0)
        if self.num_utility_pv > 0:
            utility_scaled = cp.multiply(x[:self.num_utility_pv], pv_t[:self.num_utility_pv])
        else:
            utility_scaled = np.array([])
            
        if self.num_customer_pv > 0:
            customer_scaled = cp.multiply(x[self.num_utility_pv:], pv_t[self.num_utility_pv:])
        else:
            customer_scaled = np.array([])
        
        constraints = []
        
        # 1. Voltage upper
        base_v = self.base_case['voltage'][t]
        lhs = self._compute_robust_lhs_cvxpy(
            base_v, self.matrices['M_v_utility'], utility_scaled,
            self.matrices['M_v_customer'], customer_scaled, use_max=True
        )
        constraints.append(lhs <= self.limits['voltage_upper'])
        
        # 2. Voltage lower
        lhs = self._compute_robust_lhs_cvxpy(
            base_v, self.matrices['M_v_utility'], utility_scaled,
            self.matrices['M_v_customer'], customer_scaled, use_max=False
        )
        constraints.append(lhs >= self.limits['voltage_lower'])
        
        # 3-4. Active power upper/lower
        base_P = self.base_case['active_power'][t]
        lhs = self._compute_robust_lhs_cvxpy(
            base_P, self.matrices['M_p_utility'], utility_scaled,
            self.matrices['M_p_customer'], customer_scaled, use_max=True
        )
        constraints.append(lhs <= self.limits['S_ratings'])
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_P, self.matrices['M_p_utility'], utility_scaled,
            self.matrices['M_p_customer'], customer_scaled, use_max=False
        )
        constraints.append(lhs >= -self.limits['S_ratings'])
        
        # 5-6. Reactive power upper/lower
        base_Q = self.base_case['reactive_power'][t]
        lhs = self._compute_robust_lhs_cvxpy(
            base_Q, self.matrices['M_q_utility'], utility_scaled,
            self.matrices['M_q_customer'], customer_scaled, use_max=True
        )
        constraints.append(lhs <= self.limits['S_ratings'])
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_Q, self.matrices['M_q_utility'], utility_scaled,
            self.matrices['M_q_customer'], customer_scaled, use_max=False
        )
        constraints.append(lhs >= -self.limits['S_ratings'])
        
        # 7-8. Approximated P+Q
        base_PQ = base_P + base_Q
        M_pq_util = self.matrices['M_p_utility'] + self.matrices['M_q_utility']
        M_pq_cust = self.matrices['M_p_customer'] + self.matrices['M_q_customer']
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_PQ, M_pq_util, utility_scaled, M_pq_cust, customer_scaled, use_max=True
        )
        constraints.append(lhs <= self.limits['sqrt2_S_ratings'])
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_PQ, M_pq_util, utility_scaled, M_pq_cust, customer_scaled, use_max=False
        )
        constraints.append(lhs >= -self.limits['sqrt2_S_ratings'])
        
        # 9-10. Approximated P-Q
        base_PmQ = base_P - base_Q
        M_pmq_util = self.matrices['M_p_utility'] - self.matrices['M_q_utility']
        M_pmq_cust = self.matrices['M_p_customer'] - self.matrices['M_q_customer']
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_PmQ, M_pmq_util, utility_scaled, M_pmq_cust, customer_scaled, use_max=True
        )
        constraints.append(lhs <= self.limits['sqrt2_S_ratings'])
        
        lhs = self._compute_robust_lhs_cvxpy(
            base_PmQ, M_pmq_util, utility_scaled, M_pmq_cust, customer_scaled, use_max=False
        )
        constraints.append(lhs >= -self.limits['sqrt2_S_ratings'])
        
        return constraints
    
    def _compute_robust_lhs_cvxpy(
        self,
        base: np.ndarray,
        M_util: np.ndarray,
        utility_scaled: cp.Expression,
        M_cust: np.ndarray,
        customer_scaled: cp.Expression,
        use_max: bool
    ) -> cp.Expression:
        """Compute robust worst-case LHS — implements Proposition III.1.

        For an upper-bound constraint (use_max=True), the worst-case customer PV
        is ζ_c = ζ̄_c when the impact coefficient A_{c,+}(ω) > 0, and ζ_c = 0
        otherwise.  This is the support function of the box [0, ζ̄_c]:
            sup_{ζ_c ∈ [0,ζ̄_c]} A_c(ω)ζ_c = max(A_c(ω), 0) · ζ̄_c

        The np.maximum(M_cust, 0) / np.minimum(M_cust, 0) trick selects the
        column-wise worst-case sign, replacing the explicit sup with a linear
        expression in ζ̄_c (Proposition III.1).

        Utility PV is treated as a fixed known quantity (no robustness over
        utility capacity), so M_util is used directly.
        """
        if self.is_robust:
            # Retain only the "worst-case" column sign for customer PV (Prop. III.1)
            M_cust_robust = np.maximum(M_cust, 0) if use_max else np.minimum(M_cust, 0)
        else:
            M_cust_robust = M_cust

        return base + M_util @ utility_scaled + M_cust_robust @ customer_scaled
    
    def _extract_support_from_duals(
        self,
        constraints: List,
        scenario_indices: List[int]
    ) -> Set[int]:
        """Extract support scenarios from dual variables (see `support_from_duals`)."""
        # First constraint is capacity bound, skip it
        num_non_scenario = 1
        squared = self._squared_constraint_matrices()
        blocks = {}
        for i, scenario_idx in enumerate(scenario_indices):
            start_idx = num_non_scenario + i * self.NUM_CONSTRAINT_TYPES
            row_norms = self._scenario_row_norms(scenario_idx, squared)
            blocks[scenario_idx] = list(zip(
                constraints[start_idx:start_idx + self.NUM_CONSTRAINT_TYPES], row_norms))
        support = support_from_duals(blocks)
        # No duals at all: keep every active scenario
        return set(scenario_indices) if support is None else support

    def _squared_constraint_matrices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Squared coefficient matrices of the 10 scenario constraint types, in the order of
        `_build_scenario_constraints_cvxpy`: (utility, customer) per type, the customer one
        with the robust sign selection of `_compute_robust_lhs_cvxpy`.
        """
        M = self.matrices
        pairs = [
            (M['M_v_utility'], M['M_v_customer']),
            (M['M_p_utility'], M['M_p_customer']),
            (M['M_q_utility'], M['M_q_customer']),
            (M['M_p_utility'] + M['M_q_utility'], M['M_p_customer'] + M['M_q_customer']),
            (M['M_p_utility'] - M['M_q_utility'], M['M_p_customer'] - M['M_q_customer']),
        ]
        squared = []
        for M_util, M_cust in pairs:
            for use_max in (True, False):
                if self.is_robust:
                    M_cust_robust = np.maximum(M_cust, 0) if use_max else np.minimum(M_cust, 0)
                else:
                    M_cust_robust = M_cust
                squared.append((np.asarray(M_util) ** 2, np.asarray(M_cust_robust) ** 2))
        return squared

    def _scenario_row_norms(self, t: int, squared: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """Norms of the coefficient rows (w.r.t. x) of scenario t's 10 constraint types."""
        pv_t = np.asarray(self.pv_profiles_processed[t], dtype=float)
        pv_util_sq = pv_t[:self.num_utility_pv] ** 2
        pv_cust_sq = pv_t[self.num_utility_pv:] ** 2
        return [np.sqrt(M_util_sq @ pv_util_sq + M_cust_sq @ pv_cust_sq) for M_util_sq, M_cust_sq in squared]
    
    # --------------------------------------------------------------------------
    # Constraint-based solving (for large-scale problems)
    # --------------------------------------------------------------------------
    
    def _solve_constraint_based(
        self,
        constraint_indices: List[int],
        solver: Any,
        verbose: bool
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        """
        Solve with selected individual constraints.
        Uses matrix form for efficiency.
        """
        # Build the combined constraint matrix for selected constraints
        A_list, b_list, constraint_types = self._build_constraint_matrices(constraint_indices)
        
        if len(A_list) == 0:
            # No constraints, just capacity bounds
            x = cp.Variable(self._num_variables, nonneg=True)
            objective = self._create_objective(x)
            constraints = [x <= self.capacity_limits]
            problem = cp.Problem(objective, constraints)
            solve_with_fallback(problem, solver, verbose, label="RSP subproblem (no constraints)")

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"  [WARN] Subproblem status={problem.status}.")
                return None, float('-inf'), None
            return x.value, problem.value, set()
        
        # Stack matrices
        A = np.vstack(A_list)
        b = np.hstack(b_list)
        
        # Decision variables
        x = cp.Variable(self._num_variables, nonneg=True)
        
        # Objective
        objective = self._create_objective(x)
        
        # Constraints
        constraints = [x <= self.capacity_limits]
        
        # Add selected constraints as single matrix constraint
        # Separate by type (upper vs lower bounds)
        upper_mask = np.array(['upper' in ct for ct in constraint_types])
        lower_mask = ~upper_mask
        
        if np.any(upper_mask):
            constraints.append(A[upper_mask] @ x <= b[upper_mask])
        if np.any(lower_mask):
            constraints.append(A[lower_mask] @ x >= b[lower_mask])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        solve_with_fallback(problem, solver, verbose,
                            label=f"RSP subproblem ({len(constraint_indices)} constraints)")

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"  [WARN] Subproblem status={problem.status}.")
            return None, float('-inf'), None

        # Find support by direct evaluation (check which constraints are active)
        support_indices = self._find_active_constraints(
            x.value, constraint_indices, tolerance=1e-6
        )
        
        return x.value, problem.value, support_indices
    
    def _build_constraint_matrices(
        self,
        constraint_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Build constraint matrices A and vectors b for selected constraints.
        
        Each constraint is: A_row @ x <= b (or >= for lower bounds)
        """
        A_list = []
        b_list = []
        constraint_types = []
        
        for idx in constraint_indices:
            t, ctype, elem = self._decode_constraint_index(idx)
            
            # Get the appropriate matrices and limits
            A_row, b_val, is_upper = self._get_constraint_row(t, ctype, elem)
            
            A_list.append(A_row)
            b_list.append(b_val)
            constraint_types.append(ctype)
        
        return A_list, b_list, constraint_types
    
    def _decode_constraint_index(self, idx: int) -> Tuple[int, str, int]:
        """
        Decode global constraint index to (scenario, type, element).
        """
        n_per_scenario = self.num_constraints_per_scenario
        t = idx // n_per_scenario
        remainder = idx % n_per_scenario
        
        type_idx = remainder // self._num_elements
        elem = remainder % self._num_elements
        
        constraint_types = [
            'voltage_upper', 'voltage_lower',
            'active_flow_upper', 'active_flow_lower',
            'reactive_flow_upper', 'reactive_flow_lower',
            'appr_flow_upper', 'appr_flow_lower',
            'appr_diff_flow_upper', 'appr_diff_flow_lower'
        ]
        
        return t, constraint_types[type_idx], elem
    
    def _get_constraint_row(
        self,
        t: int,
        ctype: str,
        elem: int
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Get constraint row and RHS for a specific constraint.
        
        Returns: (A_row, b_value, is_upper_bound)
        """
        pv_t = self.pv_profiles_processed[t]
        is_upper = 'upper' in ctype
        
        # Build A_row based on constraint type
        A_row = np.zeros(self._num_variables)
        
        if 'voltage' in ctype:
            M_util = self.matrices['M_v_utility']
            M_cust = self.matrices['M_v_customer']
            base_val = self.base_case['voltage'][t, elem]
            limit = self.limits['voltage_upper'][elem] if is_upper else self.limits['voltage_lower'][elem]
            
        elif 'active_flow' in ctype:
            M_util = self.matrices['M_p_utility']
            M_cust = self.matrices['M_p_customer']
            base_val = self.base_case['active_power'][t, elem]
            limit = self.limits['S_ratings'][elem] if is_upper else -self.limits['S_ratings'][elem]
            
        elif 'reactive_flow' in ctype:
            M_util = self.matrices['M_q_utility']
            M_cust = self.matrices['M_q_customer']
            base_val = self.base_case['reactive_power'][t, elem]
            limit = self.limits['S_ratings'][elem] if is_upper else -self.limits['S_ratings'][elem]
            
        elif 'appr_flow' in ctype and 'diff' not in ctype:
            M_util = self.matrices['M_p_utility'] + self.matrices['M_q_utility']
            M_cust = self.matrices['M_p_customer'] + self.matrices['M_q_customer']
            base_val = self.base_case['active_power'][t, elem] + self.base_case['reactive_power'][t, elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if is_upper else -self.limits['sqrt2_S_ratings'][elem]
            
        elif 'appr_diff_flow' in ctype:
            M_util = self.matrices['M_p_utility'] - self.matrices['M_q_utility']
            M_cust = self.matrices['M_p_customer'] - self.matrices['M_q_customer']
            base_val = self.base_case['active_power'][t, elem] - self.base_case['reactive_power'][t, elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if is_upper else -self.limits['sqrt2_S_ratings'][elem]
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")
        
        # Apply robust formulation to customer matrix
        if self.is_robust:
            M_cust_robust = np.maximum(M_cust, 0) if is_upper else np.minimum(M_cust, 0)
        else:
            M_cust_robust = M_cust
        
        # Build A_row: coefficient of x in (base + A_row @ x) <= limit
        # => A_row @ x <= limit - base
        A_row[:self.num_utility_pv] = M_util[elem, :] * pv_t[:self.num_utility_pv]
        A_row[self.num_utility_pv:] = M_cust_robust[elem, :] * pv_t[self.num_utility_pv:]
        
        b_val = limit - base_val
        
        return A_row, b_val, is_upper
    
    def _find_active_constraints(
        self,
        solution: np.ndarray,
        constraint_indices: List[int],
        tolerance: float
    ) -> Set[int]:
        """Find which constraints are active (binding) at the solution."""
        active = set()
        
        for idx in constraint_indices:
            t, ctype, elem = self._decode_constraint_index(idx)
            A_row, b_val, is_upper = self._get_constraint_row(t, ctype, elem)
            
            lhs = A_row @ solution
            
            if is_upper:
                if abs(lhs - b_val) < tolerance:
                    active.add(idx)
            else:
                if abs(lhs - b_val) < tolerance:
                    active.add(idx)
        
        return active
    
    # --------------------------------------------------------------------------
    # Violation checking
    # --------------------------------------------------------------------------
    
    def _find_violated_scenarios(
        self,
        solution: np.ndarray,
        tolerance: float,
        exclude: Set[int]
    ) -> Set[int]:
        """Find scenarios with constraint violations."""
        violated = set()
        
        utility_cap = solution[:self.num_utility_pv]
        customer_cap = solution[self.num_utility_pv:]
        
        for t in range(self._num_scenarios):
            if t in exclude:
                continue
            
            pv_t = self.pv_profiles_processed[t]
            utility_scaled = utility_cap * pv_t[:self.num_utility_pv]
            customer_scaled = customer_cap * pv_t[self.num_utility_pv:]
            
            if self._check_scenario_violations(t, utility_scaled, customer_scaled, tolerance):
                violated.add(t)
        
        return violated
    
    def _check_scenario_violations(
        self,
        t: int,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        tolerance: float
    ) -> bool:
        """Check if any constraint is violated for scenario t."""
        # Check voltage constraints
        if not self._check_voltage_feasibility(t, utility_scaled, customer_scaled, tolerance):
            return True
        
        # Check power flow constraints
        if not self._check_power_feasibility(t, utility_scaled, customer_scaled, tolerance):
            return True
        
        # Check approximated constraints
        if not self._check_approx_feasibility(t, utility_scaled, customer_scaled, tolerance):
            return True
        
        return False
    
    def _check_voltage_feasibility(
        self,
        t: int,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        tol: float
    ) -> bool:
        """Check voltage constraint feasibility."""
        base_v = self.base_case['voltage'][t]
        M_v_util = self.matrices['M_v_utility']
        M_v_cust = self.matrices['M_v_customer']
        
        # Upper
        if self.is_robust:
            lhs = base_v + M_v_util @ utility_scaled + np.maximum(M_v_cust, 0) @ customer_scaled
        else:
            lhs = base_v + M_v_util @ utility_scaled + M_v_cust @ customer_scaled
        
        if not np.all(lhs <= self.limits['voltage_upper'] + tol):
            return False
        
        # Lower
        if self.is_robust:
            lhs = base_v + M_v_util @ utility_scaled + np.minimum(M_v_cust, 0) @ customer_scaled
        else:
            lhs = base_v + M_v_util @ utility_scaled + M_v_cust @ customer_scaled
        
        if not np.all(lhs >= self.limits['voltage_lower'] - tol):
            return False
        
        return True
    
    def _check_power_feasibility(
        self,
        t: int,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        tol: float
    ) -> bool:
        """Check active and reactive power constraint feasibility."""
        # Active power
        if not self._check_power_component_feasibility(
            self.base_case['active_power'][t],
            utility_scaled, customer_scaled,
            self.matrices['M_p_utility'], self.matrices['M_p_customer'],
            self.limits['S_ratings'], tol
        ):
            return False
        
        # Reactive power
        if not self._check_power_component_feasibility(
            self.base_case['reactive_power'][t],
            utility_scaled, customer_scaled,
            self.matrices['M_q_utility'], self.matrices['M_q_customer'],
            self.limits['S_ratings'], tol
        ):
            return False
        
        return True
    
    def _check_power_component_feasibility(
        self,
        base_power: np.ndarray,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        M_util: np.ndarray,
        M_cust: np.ndarray,
        S_ratings: np.ndarray,
        tol: float
    ) -> bool:
        """Check single power component feasibility."""
        # Upper
        if self.is_robust:
            lhs = base_power + M_util @ utility_scaled + np.maximum(M_cust, 0) @ customer_scaled
        else:
            lhs = base_power + M_util @ utility_scaled + M_cust @ customer_scaled
        
        if not np.all(lhs <= S_ratings + tol):
            return False
        
        # Lower
        if self.is_robust:
            lhs = base_power + M_util @ utility_scaled + np.minimum(M_cust, 0) @ customer_scaled
        else:
            lhs = base_power + M_util @ utility_scaled + M_cust @ customer_scaled
        
        if not np.all(lhs >= -S_ratings - tol):
            return False
        
        return True
    
    def _check_approx_feasibility(
        self,
        t: int,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        tol: float
    ) -> bool:
        """Check approximated (P+Q, P-Q) constraint feasibility."""
        base_P = self.base_case['active_power'][t]
        base_Q = self.base_case['reactive_power'][t]
        sqrt2_S = self.limits['sqrt2_S_ratings']
        
        # P+Q
        if not self._check_combined_power_feasibility(
            base_P + base_Q, utility_scaled, customer_scaled,
            self.matrices['M_p_utility'] + self.matrices['M_q_utility'],
            self.matrices['M_p_customer'] + self.matrices['M_q_customer'],
            sqrt2_S, tol
        ):
            return False
        
        # P-Q
        if not self._check_combined_power_feasibility(
            base_P - base_Q, utility_scaled, customer_scaled,
            self.matrices['M_p_utility'] - self.matrices['M_q_utility'],
            self.matrices['M_p_customer'] - self.matrices['M_q_customer'],
            sqrt2_S, tol
        ):
            return False
        
        return True
    
    def _check_combined_power_feasibility(
        self,
        base_combined: np.ndarray,
        utility_scaled: np.ndarray,
        customer_scaled: np.ndarray,
        M_util_combined: np.ndarray,
        M_cust_combined: np.ndarray,
        sqrt2_S: np.ndarray,
        tol: float
    ) -> bool:
        """Check combined power constraint feasibility."""
        # Upper
        if self.is_robust:
            lhs = base_combined + M_util_combined @ utility_scaled + \
                  np.maximum(M_cust_combined, 0) @ customer_scaled
        else:
            lhs = base_combined + M_util_combined @ utility_scaled + M_cust_combined @ customer_scaled
        
        if not np.all(lhs <= sqrt2_S + tol):
            return False
        
        # Lower
        if self.is_robust:
            lhs = base_combined + M_util_combined @ utility_scaled + \
                  np.minimum(M_cust_combined, 0) @ customer_scaled
        else:
            lhs = base_combined + M_util_combined @ utility_scaled + M_cust_combined @ customer_scaled
        
        if not np.all(lhs >= -sqrt2_S - tol):
            return False
        
        return True
    
    def _find_violated_constraints(
        self,
        solution: np.ndarray,
        tolerance: float,
        exclude: Set[int]
    ) -> Set[int]:
        """Find individual constraints with violations (vectorized)."""
        violated = set()
        
        utility_cap = solution[:self.num_utility_pv]
        customer_cap = solution[self.num_utility_pv:]
        
        # Precompute matrices for robust formulation
        M_v_util = self.matrices['M_v_utility']
        M_v_cust = self.matrices['M_v_customer']
        M_p_util = self.matrices['M_p_utility']
        M_p_cust = self.matrices['M_p_customer']
        M_q_util = self.matrices['M_q_utility']
        M_q_cust = self.matrices['M_q_customer']
        
        # Combined matrices for approximated constraints
        M_pq_util = M_p_util + M_q_util
        M_pq_cust = M_p_cust + M_q_cust
        M_pmq_util = M_p_util - M_q_util
        M_pmq_cust = M_p_cust - M_q_cust
        
        # Robust versions (precompute once)
        if self.is_robust:
            M_v_cust_pos = np.maximum(M_v_cust, 0)
            M_v_cust_neg = np.minimum(M_v_cust, 0)
            M_p_cust_pos = np.maximum(M_p_cust, 0)
            M_p_cust_neg = np.minimum(M_p_cust, 0)
            M_q_cust_pos = np.maximum(M_q_cust, 0)
            M_q_cust_neg = np.minimum(M_q_cust, 0)
            M_pq_cust_pos = np.maximum(M_pq_cust, 0)
            M_pq_cust_neg = np.minimum(M_pq_cust, 0)
            M_pmq_cust_pos = np.maximum(M_pmq_cust, 0)
            M_pmq_cust_neg = np.minimum(M_pmq_cust, 0)
        else:
            M_v_cust_pos = M_v_cust_neg = M_v_cust
            M_p_cust_pos = M_p_cust_neg = M_p_cust
            M_q_cust_pos = M_q_cust_neg = M_q_cust
            M_pq_cust_pos = M_pq_cust_neg = M_pq_cust
            M_pmq_cust_pos = M_pmq_cust_neg = M_pmq_cust
        
        # Limits
        v_upper = self.limits['voltage_upper']
        v_lower = self.limits['voltage_lower']
        S_ratings = self.limits['S_ratings']
        sqrt2_S = self.limits['sqrt2_S_ratings']
        
        n_elem = self._num_elements
        
        # Process each scenario
        for t in range(self._num_scenarios):
            pv_t = self.pv_profiles_processed[t]
            utility_scaled = utility_cap * pv_t[:self.num_utility_pv]
            customer_scaled = customer_cap * pv_t[self.num_utility_pv:]
            
            base_offset = t * self.num_constraints_per_scenario
            
            # Voltage upper (ctype_idx = 0)
            lhs = (self.base_case['voltage'][t] + 
                   M_v_util @ utility_scaled + M_v_cust_pos @ customer_scaled)
            violations_mask = lhs > v_upper + tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 0 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Voltage lower (ctype_idx = 1)
            lhs = (self.base_case['voltage'][t] + 
                   M_v_util @ utility_scaled + M_v_cust_neg @ customer_scaled)
            violations_mask = lhs < v_lower - tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 1 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Active power upper (ctype_idx = 2)
            lhs = (self.base_case['active_power'][t] + 
                   M_p_util @ utility_scaled + M_p_cust_pos @ customer_scaled)
            violations_mask = lhs > S_ratings + tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 2 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Active power lower (ctype_idx = 3)
            lhs = (self.base_case['active_power'][t] + 
                   M_p_util @ utility_scaled + M_p_cust_neg @ customer_scaled)
            violations_mask = lhs < -S_ratings - tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 3 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Reactive power upper (ctype_idx = 4)
            lhs = (self.base_case['reactive_power'][t] + 
                   M_q_util @ utility_scaled + M_q_cust_pos @ customer_scaled)
            violations_mask = lhs > S_ratings + tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 4 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Reactive power lower (ctype_idx = 5)
            lhs = (self.base_case['reactive_power'][t] + 
                   M_q_util @ utility_scaled + M_q_cust_neg @ customer_scaled)
            violations_mask = lhs < -S_ratings - tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 5 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Approximated P+Q upper (ctype_idx = 6)
            base_pq = self.base_case['active_power'][t] + self.base_case['reactive_power'][t]
            lhs = base_pq + M_pq_util @ utility_scaled + M_pq_cust_pos @ customer_scaled
            violations_mask = lhs > sqrt2_S + tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 6 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Approximated P+Q lower (ctype_idx = 7)
            lhs = base_pq + M_pq_util @ utility_scaled + M_pq_cust_neg @ customer_scaled
            violations_mask = lhs < -sqrt2_S - tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 7 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Approximated P-Q upper (ctype_idx = 8)
            base_pmq = self.base_case['active_power'][t] - self.base_case['reactive_power'][t]
            lhs = base_pmq + M_pmq_util @ utility_scaled + M_pmq_cust_pos @ customer_scaled
            violations_mask = lhs > sqrt2_S + tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 8 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
            
            # Approximated P-Q lower (ctype_idx = 9)
            lhs = base_pmq + M_pmq_util @ utility_scaled + M_pmq_cust_neg @ customer_scaled
            violations_mask = lhs < -sqrt2_S - tolerance
            if np.any(violations_mask):
                violated_elems = np.where(violations_mask)[0]
                for elem in violated_elems:
                    idx = base_offset + 9 * n_elem + elem
                    if idx not in exclude:
                        violated.add(idx)
        
        return violated
    
    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    
    def _create_objective(self, x: cp.Variable) -> cp.Maximize:
        """Create optimization objective.

        Without fairness: maximize Σζ̄_c (total PVHC), Eq. (10).
        With fairness: maximize Σζ̄_c + γ·Σlog(ζ̄_c + η), Eq. (16).
        The log term incentivizes equal allocation across customers;
        η (default 1e-5) is a numerical floor keeping log well-defined at zero.
        """
        total_capacity = cp.sum(x)

        if self.with_fairness:
            fairness_term = self.fairness_factor * cp.sum(cp.log(x + 1e-5))
            return cp.Maximize(total_capacity + fairness_term)

        return cp.Maximize(total_capacity)
    
    def _compute_pv_impact_matrices(self) -> Dict[str, np.ndarray]:
        """Compute matrices relating PV capacity to voltage/power impacts."""
        M_v_util, M_p_util, M_q_util = self._construct_pv_matrix(self.utility_pv_info)
        M_v_cust, M_p_cust, M_q_cust = self._construct_pv_matrix(self.customer_pv_info)
        
        return {
            'M_v_utility': M_v_util,
            'M_p_utility': M_p_util,
            'M_q_utility': M_q_util,
            'M_v_customer': M_v_cust,
            'M_p_customer': M_p_cust,
            'M_q_customer': M_q_cust
        }
    
    def _construct_pv_matrix(
        self,
        pv_info: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct M_v, M_p, M_q matrices — implements Eq. (8).

        For each PV unit c at bus b with PV profile σ_c(ω):
            Δv̂_c = R̃[:, b] · σ_c(ω) · ζ̄_c   →  column of M_v
            ΔP_c  = Re(C̃_S[:, b]) · σ_c(ω) · ζ̄_c  →  column of M_p
            ΔQ_c  = Im(C̃_S[:, b]) · σ_c(ω) · ζ̄_c  →  column of M_q

        PV runs at unity power factor, but C̃_S is complex: through a delta winding,
        a phase's power reaches the other phases with complex weights, so real PV
        power also drives per-phase reactive branch flows (they sum to zero over
        the phases).

        For delta-connected PV, the injection is first converted to equivalent
        wye injection via the D2Y transformation (Appendix A).

        The factor 1e-3 in M_v converts kW → MW to match the voltage units.
        """
        R = self.network_data["R"]
        X = self.network_data["X"]
        C_tilde_S = self.network_data["C_tilde_S"]
        node2idx = self.network_data["node2idx"]
        n_buses = self.network_data["n_buses"]
        
        n_pv = len(pv_info)
        M_v_pv = np.zeros((n_buses * 3, n_pv))
        M_p_pv = np.zeros((n_buses * 3, n_pv))
        M_q_pv = np.zeros((n_buses * 3, n_pv))
        
        for i, info in enumerate(pv_info):
            bus_name = info["bus"]
            phases = info["phases"]
            
            try:
                bus_idx = node2idx[bus_name]
            except KeyError:
                raise KeyError(f"Bus '{bus_name}' not found in node2idx")
            
            if phases[1] == 0:  # Grounded connection
                self._add_grounded_pv_impact(
                    i, bus_idx, phases[0], R, X, C_tilde_S,
                    M_v_pv, M_p_pv, M_q_pv
                )
            else:  # Delta connection
                self._add_delta_pv_impact(
                    i, bus_idx, phases, info.get('connected_transformer', {}),
                    R, X, C_tilde_S, M_v_pv, M_p_pv, M_q_pv
                )
        
        return M_v_pv, M_p_pv, M_q_pv
    
    def _add_grounded_pv_impact(
        self, pv_idx, bus_idx, phase, R, X, C_tilde_S,
        M_v_pv, M_p_pv, M_q_pv
    ):
        """Add impact of grounded PV connection."""
        idx = (bus_idx - 1) * 3 + phase - 1
        M_v_pv[:, pv_idx] = np.real(1e-3 * R[:, idx])
        M_p_pv[:, pv_idx] = np.real(C_tilde_S[:, idx])
        M_q_pv[:, pv_idx] = np.imag(C_tilde_S[:, idx])

    def _add_delta_pv_impact(
        self, pv_idx, bus_idx, phases, conn_trans, R, X, C_tilde_S,
        M_v_pv, M_p_pv, M_q_pv
    ):
        """Add impact of delta-connected PV."""
        phase_idx = ldf.STD_DELTA_ORDER.index(tuple(sorted(phases)))
        transformation_vec = self._get_transformation_vector(conn_trans, phase_idx)
        
        idx = (bus_idx - 1) * 3
        
        M_v_pv[:, pv_idx] = (
            1e-3 * R[:, idx:idx+3] @ np.real(transformation_vec) +
            1e-3 * X[:, idx:idx+3] @ np.imag(transformation_vec)
        )
        branch_pwr = C_tilde_S[:, idx:idx+3] @ transformation_vec
        M_p_pv[:, pv_idx] = np.real(branch_pwr)
        M_q_pv[:, pv_idx] = np.imag(branch_pwr)
    
    def _get_transformation_vector(self, conn_trans: Dict, phase_idx: int) -> np.ndarray:
        """Get delta-to-wye transformation vector."""
        if not conn_trans:
            return ldf.D2Y[:, phase_idx]
        
        element_type = conn_trans.get('element_type', 'line')
        
        if element_type in {'line', 'YY', 'DD', 'YD', 'DY', 'YD_alt',
                           'DY_alt', 'openD_openD', 'openY_openY'}:
            return ldf.D2Y[:, phase_idx]
        
        if element_type in {'openY_openD_split', 'openD_openD_split'}:
            assert conn_trans.get('split_phase') in conn_trans.get('winding_pairing', [])
            _, permutation = ldf.build_permutation_matrix_open_split(conn_trans)
            return (permutation.T @ ldf.openD_split2Y @ permutation)[:, phase_idx]
        
        raise ValueError(f"Unknown transformer type: {element_type}")
    
    def _compute_network_limits(self) -> Dict[str, np.ndarray]:
        """Compute voltage and power flow limits."""
        nominal_voltage_sq = self._get_nominal_voltage_squared()
        
        voltage_upper = nominal_voltage_sq * (1 + self.vol_tol) ** 2
        voltage_lower = nominal_voltage_sq * (1 - self.lower_voltage_limit_ratio * self.vol_tol) ** 2
        
        n_buses = self.network_data["n_buses"]
        raw_ratings = self.network_data["branch_pwr_ratings"]
        processed_ratings = [
            r if (r is not None and r != 0) else 1e6
            for r in raw_ratings[:n_buses]
        ]
        # Use the equipment's real OpenDSS-derived thermal rating as-is -- no blanket
        # safety margin. `relaxation_factor` is reserved for the *adaptive*, per-branch
        # relaxation in _relax_limits_adaptively(), which only stretches a rating when
        # the base (zero-PV) case is actually found infeasible against it.
        S_ratings = np.repeat(processed_ratings, 3)
        
        return {
            'voltage_upper': voltage_upper,
            'voltage_lower': voltage_lower,
            'S_ratings': S_ratings,
            'sqrt2_S_ratings': np.sqrt(2) * S_ratings
        }
    
    def _get_nominal_voltage_squared(self) -> np.ndarray:
        """Get nominal voltage squared vector."""
        N = self.network_data["n_buses"]
        C_tilde_0 = self.network_data["C_tilde_0"]
        
        v0_ref_squared = ldf.compute_v0_ref(
            self.v_root_nom * np.ones(3)
        ).swapaxes(0, 1).reshape(9, 1)
        nominal_squared = (
            C_tilde_0.reshape(N, 3, 9) @ v0_ref_squared
        ).reshape(N * 3)

        assert np.imag(nominal_squared).max() < 1e-6, "Nominal voltages should be real"
        return np.real(nominal_squared)
    
    def _preprocess_pv_profiles(self, pv_profiles: np.ndarray) -> np.ndarray:
        """Reshape PV profiles to match decision variables."""
        if pv_profiles.ndim == 1:
            return np.repeat(
                pv_profiles.reshape(-1, 1),
                self._num_variables,
                axis=1
            )
        return pv_profiles
    
    def _get_nonzero_pv_indices(self, pv_profiles: np.ndarray) -> np.ndarray:
        """Get indices of scenarios with non-zero PV generation."""
        if pv_profiles.ndim == 1:
            return np.where(pv_profiles > 0)[0]
        return np.where(np.any(pv_profiles > 0, axis=1))[0]


# ==============================================================================
# PVHC PROBLEM WITH BATTERY
# ==============================================================================

class PVHCProblemWithBattery(ClarksonProblem):
    """
    PV Hosting Capacity Problem with open-loop battery storage control.

    This extends :class:`PVHCProblemWithoutBattery` by adding:
      - Battery investment variables zeta_b (binary or continuous in [0,1])
      - Battery dispatch variables p_b(t), q_b(t)
      - Battery SoC dynamics and bounds

    The network constraints follow the robust reformulation in math_formulation.tex.
    """
    DEFAULT_LOWER_VOLTAGE_RATIO = PVHCProblemWithoutBattery.DEFAULT_LOWER_VOLTAGE_RATIO
    DEFAULT_RELAXATION_FACTOR = PVHCProblemWithoutBattery.DEFAULT_RELAXATION_FACTOR
    DEFAULT_FAIRNESS_FACTOR = PVHCProblemWithoutBattery.DEFAULT_FAIRNESS_FACTOR
    NUM_CONSTRAINT_TYPES = PVHCProblemWithoutBattery.NUM_CONSTRAINT_TYPES  # network constraints per scenario

    def __init__(
        self,
        simulator: Simulator,
        utility_pv_info: List[Dict[str, Any]],
        customer_pv_info: List[Dict[str, Any]],
        utility_battery_info: List[Dict[str, Any]],
        load_info: List[Dict[str, Any]],
        is_robust: bool = True,
        with_fairness: bool = False,
        fairness_factor: float = DEFAULT_FAIRNESS_FACTOR,
        battery_cost: float = 0.0,
        zeta_b_binary: bool = True,
        delta_t_hours: float = 0.5,
        total_capacity_limit: Optional[float] = None,
        load_cached_limits: bool = False
    ):
        self.simulator = simulator
        self.utility_pv_info = utility_pv_info or []
        self.customer_pv_info = customer_pv_info or []
        self.utility_battery_info = utility_battery_info or []
        self.load_info = load_info
        self.is_robust = is_robust
        self.with_fairness = with_fairness
        self.fairness_factor = fairness_factor
        self.battery_cost = float(battery_cost)
        self.zeta_b_binary = bool(zeta_b_binary)
        self.delta_t_hours = float(delta_t_hours)
        self.total_capacity_limit = total_capacity_limit

        # Configuration consistent with PVHCProblemWithoutBattery
        self.vol_tol = VOL_TOL
        self.lower_voltage_limit_ratio = self.DEFAULT_LOWER_VOLTAGE_RATIO
        self.relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        self.v_root_nom = self._get_feeder_nominal_voltage()

        # Counts
        self.num_utility_pv = len(self.utility_pv_info)
        self.num_customer_pv = len(self.customer_pv_info)
        self.num_battery = len(self.utility_battery_info)

        # Network
        self.network_data = self.simulator.get_network_data()
        self._n_buses = self.network_data["n_buses"]
        self._num_elements = self._n_buses * 3

        # Scenario storage
        self._num_scenarios = 0
        self._load_profiles: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pv_profiles: Optional[np.ndarray] = None
        self._V0_profiles: Optional[np.ndarray] = None
        self._scenario_is_processed = False

        self._battery_period = 48

        # Data populated by compile()
        self._is_compiled = False
        self.base_case: Optional[Dict[str, np.ndarray]] = None
        self.pv_profiles_processed: Optional[np.ndarray] = None
        self.limits: Optional[Dict[str, np.ndarray]] = None
        self.matrices_pv: Optional[Dict[str, np.ndarray]] = None
        self.matrices_batt: Optional[Dict[str, np.ndarray]] = None

        # Capacity limits (first stage)
        self.capacity_limits_pv: Optional[np.ndarray] = None  # utility+customer PV
        self.capacity_limits_customer: Optional[np.ndarray] = None
        self.capacity_limits_utility: Optional[np.ndarray] = None

        # Battery bounds (per installation candidate)
        self.batt_bounds: Optional[Dict[str, np.ndarray]] = None

        # Variable packing (set in compile)
        self._num_variables = 0
        self._offsets: Dict[str, int] = {}
        self._T: int = 0  # number of time steps / scenarios
        
        # Adaptive limit relaxation flag
        self.adaptive_flag = True
        
        # Limit caching configuration
        self.load_cached_limits = load_cached_limits
        self.limits_cache_dir = "temp/cache/limits"

    # --------------------------------------------------------------------------
    # ClarksonProblem interface
    # --------------------------------------------------------------------------
    @property
    def num_scenarios(self) -> int:
        return self._num_scenarios

    @property
    def num_constraints_per_scenario(self) -> int:
        # Only the network constraints are scenario-indexed for Clarkson.
        return self.NUM_CONSTRAINT_TYPES * self._num_elements

    @property
    def num_variables(self) -> int:
        return self._num_variables

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------
    def add_scenarios(
        self,
        load_profiles: Tuple[np.ndarray, np.ndarray],
        pv_profiles: np.ndarray,
        V0_profiles: np.ndarray,
        filter_zero_pv: bool = True
    ) -> None:
        """
        Add scenarios to the problem. Since this problem contains batteries, the 
        scenarios are required to be days by days (24-hour profiles).

        Parameters
        ----------
        load_profiles : Tuple[np.ndarray, np.ndarray]
            Tuple of active and reactive load profiles with shape (num_scenarios, num_loads).
        pv_profiles : np.ndarray
            PV generation profiles with shape (num_scenarios, num_pv).
        V0_profiles : np.ndarray
            Substation voltage profiles with shape (num_scenarios, 3).
        filter_zero_pv : bool, optional
            If True, filter out scenarios with zero PV generation, by default True.
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
        if self._load_profiles is None or self._pv_profiles is None or self._V0_profiles is None:
            raise RuntimeError("Scenarios not added. Call add_scenarios() before compile().")

        if not self._scenario_is_processed:
            self._num_scenarios = self._pv_profiles.shape[0]
            self._scenario_is_processed = True

            # Base case: loads only (PV_info=None)
            base_v, base_P, base_Q, _, _, _ = self.simulator.simulate(
                load_profiles=self._load_profiles,
                pv_profiles=self._pv_profiles,
                V0_profiles=self._V0_profiles,
                load_info=self.load_info,
                PV_info=None
            )
            self.base_case = {
                "voltage": np.square(base_v),
                "active_power": base_P,
                "reactive_power": base_Q
            }

            self.pv_profiles_processed = self._preprocess_pv_profiles(self._pv_profiles)

        # Matrices
        self.matrices_pv = self._compute_pv_impact_matrices()
        self.matrices_batt = self._compute_battery_impact_matrices()

        # Compute or load network limits
        if self.load_cached_limits:
            loaded_limits = self._load_limits_from_cache()
            if loaded_limits is not None:
                self.limits = loaded_limits
                print(f"Loaded network limits from cache.")
            else:
                print(f"Cache miss or incompatible. Computing limits...")
                self.limits = self._compute_network_limits()
                self._save_limits_to_cache()
        else:
            self.limits = self._compute_network_limits()
            self._save_limits_to_cache()        

        # PV capacity limits
        self.capacity_limits_utility = np.array([info["capacity"] for info in self.utility_pv_info], dtype=float) \
            if self.num_utility_pv > 0 else np.zeros(0)
        self.capacity_limits_customer = np.array([info["capacity"] for info in self.customer_pv_info], dtype=float) \
            if self.num_customer_pv > 0 else np.zeros(0)
        self.capacity_limits_pv = np.concatenate([self.capacity_limits_utility, self.capacity_limits_customer], axis=0)

        # Battery bounds
        self.batt_bounds = self._compute_battery_bounds()

        # Variable packing
        self._T = int(self._num_scenarios)
        self._build_variable_offsets()

        # Check basic feasibility with zero PV and battery
        if not self._check_basic_feasibility():
            raise RuntimeError(
                "Network remains infeasible under existing customer load (zero PV/battery) "
                "even after adaptive limit relaxation. This means some equipment is loaded "
                "far beyond what a modest safety margin can account for -- inspect the "
                "[Base feasibility] messages above rather than solving with limits that "
                "don't reflect a real operating point."
            )

        self._is_compiled = True

    def solve_subproblem(
        self,
        indices: List[int],
        is_scenario_based: bool = True,
        solver: Optional[Any] = cp.HIGHS,
        verbose: bool = False
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")

        if is_scenario_based:
            return self._solve_scenario_based(indices, solver, verbose)
        else:
            return self._solve_constraint_based(indices, solver, verbose)

    def find_violated_indices(
        self,
        solution: np.ndarray,
        tolerance: float = 1e-6,
        is_scenario_based: bool = True,
        exclude_indices: Optional[Set[int]] = None
    ) -> Set[int]:
        if not self._is_compiled:
            raise RuntimeError("Problem not compiled. Call compile() first.")

        exclude = exclude_indices or set()
        if is_scenario_based:
            return self._find_violated_scenarios(solution, tolerance, exclude)
        else:
            return self._find_violated_constraints(solution, tolerance, exclude)

    # --------------------------------------------------------------------------
    # Scenario-based solve (recommended)
    # --------------------------------------------------------------------------
    def _solve_scenario_based(
        self,
        scenario_indices: List[int],
        solver: Any,
        verbose: bool
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        T = self._battery_period
        n_u, n_c, n_b = self.num_utility_pv, self.num_customer_pv, self.num_battery

        # Variables
        zeta_u = cp.Variable(n_u, nonneg=True) if n_u > 0 else None
        zeta_cbar = cp.Variable(n_c, nonneg=True) if n_c > 0 else None

        if n_b > 0:
            if self.zeta_b_binary:
                zeta_b = cp.Variable(n_b, boolean=True)
            else:
                zeta_b = cp.Variable(n_b)
            p_b = cp.Variable((T, n_b))
            q_b = cp.Variable((T, n_b))
            soc = cp.Variable((T, n_b))
        else:
            zeta_b = None
            p_b = q_b = soc = None

        # Objective: PVHC benefit minus battery investment cost
        obj_expr = 0
        if zeta_u is not None:
            obj_expr += cp.sum(zeta_u)
        if zeta_cbar is not None:
            obj_expr += cp.sum(zeta_cbar)
        if zeta_b is not None and self.battery_cost != 0.0:
            obj_expr -= self.battery_cost * cp.sum(zeta_b)

        if self.with_fairness:
            # Fairness only on PV investment variables (avoid log on binaries)
            fairness_terms = []
            if zeta_u is not None:
                fairness_terms.append(cp.sum(cp.log(zeta_u + 1e-5)))
            if zeta_cbar is not None:
                fairness_terms.append(cp.sum(cp.log(zeta_cbar + 1e-5)))
            if len(fairness_terms) > 0:
                obj_expr += self.fairness_factor * cp.sum(fairness_terms)

        objective = cp.Maximize(obj_expr)

        constraints: List = []

        # PV bounds
        if zeta_u is not None:
            constraints.append(zeta_u <= self.capacity_limits_utility)
        if zeta_cbar is not None:
            constraints.append(zeta_cbar <= self.capacity_limits_customer)

        # Battery bounds and dynamics
        if n_b > 0:
            if not self.zeta_b_binary:
                constraints += [zeta_b >= 0, zeta_b <= 1]

            pmin = self.batt_bounds["p_min"]
            pmax = self.batt_bounds["p_max"]
            qmin = self.batt_bounds["q_min"]
            qmax = self.batt_bounds["q_max"]
            socmin = self.batt_bounds["soc_min"]
            socmax = self.batt_bounds["soc_max"]

            # Dispatch bounds (vectorized)
            constraints += [
                p_b >= cp.multiply(np.ones((T, 1)) @ pmin.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
                p_b <= cp.multiply(np.ones((T, 1)) @ pmax.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
                q_b >= cp.multiply(np.ones((T, 1)) @ qmin.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
                q_b <= cp.multiply(np.ones((T, 1)) @ qmax.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
            ]

            # SoC dynamics: soc[t] = soc[t-1] - p_b[t] * dt  (t=1..T-1)
            if T > 1:
                constraints.append(soc[1:, :] == soc[:-1, :] - self.delta_t_hours * p_b[1:, :])
            # Cycle constraint
            if T >= 1:
                constraints.append(soc[0, :] == soc[-1, :] - self.delta_t_hours * p_b[0, :])

            # SoC bounds
            constraints += [
                soc >= cp.multiply(np.ones((T, 1)) @ socmin.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
                soc <= cp.multiply(np.ones((T, 1)) @ socmax.reshape(1, -1), cp.reshape(zeta_b, (1, n_b), order='C')),
            ]

        # Network constraints (only for selected scenarios)
        for t in scenario_indices:
            constraints.extend(self._build_scenario_constraints_cvxpy(zeta_u, zeta_cbar, zeta_b, p_b, q_b, t))

        problem = cp.Problem(objective, constraints)
        solve_with_fallback(problem, solver, verbose,
                            label=f"battery subproblem ({len(scenario_indices)} scenarios)")

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"  [WARN] Subproblem status={problem.status}.")
            return None, float("-inf"), None

        # Pack solution for violation checking
        sol = self._pack_solution(zeta_u, zeta_cbar, zeta_b, p_b, q_b, soc)

        # Support: for MILP duals are generally unavailable; use active-constraint heuristic
        support = self._extract_support_by_activity(sol, scenario_indices, tol=1e-6)
        return sol, float(problem.value), support

    def _build_scenario_constraints_cvxpy(
        self,
        zeta_u: Optional[cp.Variable],
        zeta_cbar: Optional[cp.Variable],
        zeta_b: Optional[cp.Variable],
        p_b: Optional[cp.Variable],
        q_b: Optional[cp.Variable],
        t: int
    ) -> List:
        pv_t = self.pv_profiles_processed[t]

        # PV scaling
        if self.num_utility_pv > 0:
            util_scaled = cp.multiply(zeta_u, pv_t[:self.num_utility_pv])
        else:
            util_scaled = None
        if self.num_customer_pv > 0:
            cust_scaled = cp.multiply(zeta_cbar, pv_t[self.num_utility_pv:self.num_utility_pv+self.num_customer_pv])
        else:
            cust_scaled = None

        # Battery dispatch at time t
        if self.num_battery > 0:
            # Notice the battery only has self._battery_period time steps
            t_batt = t % self._battery_period
            p_bt = p_b[t_batt, :]
            q_bt = q_b[t_batt, :]
        else:
            p_bt = q_bt = None

        constraints = []

        # --- Voltage upper/lower ---
        base_v = self.base_case["voltage"][t]
        lhs_v = self._lhs_network_cvxpy(
            base_v,
            self.matrices_pv["M_v_utility"], util_scaled,
            self.matrices_pv["M_v_customer"], cust_scaled,
            self.matrices_batt["M_v_p"], p_bt,
            self.matrices_batt["M_v_q"], q_bt,
            use_max=True
        )
        constraints.append(lhs_v <= self.limits["voltage_upper"])

        lhs_v = self._lhs_network_cvxpy(
            base_v,
            self.matrices_pv["M_v_utility"], util_scaled,
            self.matrices_pv["M_v_customer"], cust_scaled,
            self.matrices_batt["M_v_p"], p_bt,
            self.matrices_batt["M_v_q"], q_bt,
            use_max=False
        )
        constraints.append(lhs_v >= self.limits["voltage_lower"])

        # --- Active flow upper/lower ---
        base_P = self.base_case["active_power"][t]
        lhs_P = self._lhs_network_cvxpy(
            base_P,
            self.matrices_pv["M_p_utility"], util_scaled,
            self.matrices_pv["M_p_customer"], cust_scaled,
            self.matrices_batt["M_P_p"], p_bt,
            self.matrices_batt["M_P_q"], q_bt,
            use_max=True
        )
        constraints.append(lhs_P <= self.limits["S_ratings"])

        lhs_P = self._lhs_network_cvxpy(
            base_P,
            self.matrices_pv["M_p_utility"], util_scaled,
            self.matrices_pv["M_p_customer"], cust_scaled,
            self.matrices_batt["M_P_p"], p_bt,
            self.matrices_batt["M_P_q"], q_bt,
            use_max=False
        )
        constraints.append(lhs_P >= -self.limits["S_ratings"])

        # --- Reactive flow upper/lower ---
        base_Q = self.base_case["reactive_power"][t]
        lhs_Q = self._lhs_network_cvxpy(
            base_Q,
            self.matrices_pv["M_q_utility"], util_scaled,
            self.matrices_pv["M_q_customer"], cust_scaled,
            self.matrices_batt["M_Q_p"], p_bt,
            self.matrices_batt["M_Q_q"], q_bt,
            use_max=True
        )
        constraints.append(lhs_Q <= self.limits["S_ratings"])

        lhs_Q = self._lhs_network_cvxpy(
            base_Q,
            self.matrices_pv["M_q_utility"], util_scaled,
            self.matrices_pv["M_q_customer"], cust_scaled,
            self.matrices_batt["M_Q_p"], p_bt,
            self.matrices_batt["M_Q_q"], q_bt,
            use_max=False
        )
        constraints.append(lhs_Q >= -self.limits["S_ratings"])

        # --- Approximated P+Q and P-Q (same as PVHCProblemWithoutBattery structure) ---
        base_PQ = base_P + base_Q
        M_pq_util = self.matrices_pv["M_p_utility"] + self.matrices_pv["M_q_utility"]
        M_pq_cust = self.matrices_pv["M_p_customer"] + self.matrices_pv["M_q_customer"]
        M_pq_b_p = self.matrices_batt["M_P_p"] + self.matrices_batt["M_Q_p"]
        M_pq_b_q = self.matrices_batt["M_P_q"] + self.matrices_batt["M_Q_q"]

        lhs = self._lhs_network_cvxpy(
            base_PQ,
            M_pq_util, util_scaled,
            M_pq_cust, cust_scaled,
            M_pq_b_p, p_bt,
            M_pq_b_q, q_bt,
            use_max=True
        )
        constraints.append(lhs <= self.limits["sqrt2_S_ratings"])

        lhs = self._lhs_network_cvxpy(
            base_PQ,
            M_pq_util, util_scaled,
            M_pq_cust, cust_scaled,
            M_pq_b_p, p_bt,
            M_pq_b_q, q_bt,
            use_max=False
        )
        constraints.append(lhs >= -self.limits["sqrt2_S_ratings"])

        base_PmQ = base_P - base_Q
        M_pmq_util = self.matrices_pv["M_p_utility"] - self.matrices_pv["M_q_utility"]
        M_pmq_cust = self.matrices_pv["M_p_customer"] - self.matrices_pv["M_q_customer"]
        M_pmq_b_p = self.matrices_batt["M_P_p"] - self.matrices_batt["M_Q_p"]
        M_pmq_b_q = self.matrices_batt["M_P_q"] - self.matrices_batt["M_Q_q"]

        lhs = self._lhs_network_cvxpy(
            base_PmQ,
            M_pmq_util, util_scaled,
            M_pmq_cust, cust_scaled,
            M_pmq_b_p, p_bt,
            M_pmq_b_q, q_bt,
            use_max=True
        )
        constraints.append(lhs <= self.limits["sqrt2_S_ratings"])

        lhs = self._lhs_network_cvxpy(
            base_PmQ,
            M_pmq_util, util_scaled,
            M_pmq_cust, cust_scaled,
            M_pmq_b_p, p_bt,
            M_pmq_b_q, q_bt,
            use_max=False
        )
        constraints.append(lhs >= -self.limits["sqrt2_S_ratings"])

        return constraints

    def _lhs_network_cvxpy(
        self,
        base: np.ndarray,
        M_util: np.ndarray,
        util_scaled: Optional[cp.Expression],
        M_cust: np.ndarray,
        cust_scaled: Optional[cp.Expression],
        M_b_p: np.ndarray,
        p_bt: Optional[cp.Expression],
        M_b_q: np.ndarray,
        q_bt: Optional[cp.Expression],
        use_max: bool
    ) -> cp.Expression:
        expr = base

        if util_scaled is not None and M_util.size > 0:
            expr = expr + M_util @ util_scaled

        if cust_scaled is not None and M_cust.size > 0:
            if self.is_robust:
                M_cust_robust = np.maximum(M_cust, 0) if use_max else np.minimum(M_cust, 0)
            else:
                M_cust_robust = M_cust
            expr = expr + M_cust_robust @ cust_scaled

        if p_bt is not None and M_b_p.size > 0:
            expr = expr + M_b_p @ p_bt
        if q_bt is not None and M_b_q.size > 0:
            expr = expr + M_b_q @ q_bt

        return expr

    # --------------------------------------------------------------------------
    # Constraint-based mode (network constraints only; battery constraints are global)
    # --------------------------------------------------------------------------
    def _solve_constraint_based(
        self,
        constraint_indices: List[int],
        solver: Any,
        verbose: bool
    ) -> Tuple[Optional[np.ndarray], float, Set[int]]:
        # Build A, b for the selected NETWORK constraint rows only.
        A_list, b_list, ctype_list = self._build_constraint_matrices(constraint_indices)

        # Variables (vector form)
        x = cp.Variable(self._num_variables)

        constraints: List = []
        constraints += self._global_constraints_from_x(x)

        # Add selected constraints as matrix inequalities
        if len(A_list) > 0:
            A = np.vstack(A_list)
            b = np.hstack(b_list)
            upper_mask = np.array(["upper" in ct for ct in ctype_list])
            lower_mask = ~upper_mask
            if np.any(upper_mask):
                constraints.append(A[upper_mask] @ x <= b[upper_mask])
            if np.any(lower_mask):
                constraints.append(A[lower_mask] @ x >= b[lower_mask])

        objective = cp.Maximize(self._objective_from_x(x))
        prob = cp.Problem(objective, constraints)
        solve_with_fallback(prob, solver, verbose,
                            label=f"battery subproblem ({len(constraint_indices)} constraints)")

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"  [WARN] Subproblem status={prob.status}.")
            return None, float("-inf"), None

        sol = np.array(x.value).reshape(-1)
        support = self._find_active_constraints(sol, constraint_indices, tolerance=1e-6)
        return sol, float(prob.value), support

    # --------------------------------------------------------------------------
    # Violation checking
    # --------------------------------------------------------------------------
    def _find_violated_scenarios(
        self,
        solution: np.ndarray,
        tolerance: float,
        exclude: Set[int]
    ) -> Set[int]:
        violated = set()
        parts = self._unpack_solution(solution)

        for t in range(self._T):
            if t in exclude:
                continue
            if self._check_scenario_violations(t, parts, tolerance):
                violated.add(t)
        return violated

    def _check_scenario_violations(
        self,
        t: int,
        parts: Dict[str, np.ndarray],
        tol: float
    ) -> bool:
        # Only network constraints drive Clarkson violations.
        pv_t = self.pv_profiles_processed[t]
        zeta_u = parts["zeta_u"]
        zeta_cbar = parts["zeta_cbar"]
        tbatt = t % self._battery_period
        p_bt = parts["p_b"][tbatt] if self.num_battery > 0 else None
        q_bt = parts["q_b"][tbatt] if self.num_battery > 0 else None

        util_scaled = zeta_u * pv_t[:self.num_utility_pv] if self.num_utility_pv > 0 else np.zeros(0)
        cust_scaled = zeta_cbar * pv_t[self.num_utility_pv:self.num_utility_pv+self.num_customer_pv] \
            if self.num_customer_pv > 0 else np.zeros(0)

        # Precompute robust matrices
        def robust(M, upper: bool):
            if not self.is_robust:
                return M
            return np.maximum(M, 0) if upper else np.minimum(M, 0)

        # Voltage
        base_v = self.base_case["voltage"][t]
        Mv_u = self.matrices_pv["M_v_utility"]
        Mv_c = self.matrices_pv["M_v_customer"]
        lhs = base_v
        if self.num_utility_pv > 0:
            lhs = lhs + Mv_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mv_c, True) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_v_p"] @ p_bt + self.matrices_batt["M_v_q"] @ q_bt
        if np.any(lhs > self.limits["voltage_upper"] + tol):
            return True

        lhs = base_v
        if self.num_utility_pv > 0:
            lhs = lhs + Mv_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mv_c, False) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_v_p"] @ p_bt + self.matrices_batt["M_v_q"] @ q_bt
        if np.any(lhs < self.limits["voltage_lower"] - tol):
            return True

        # Active flow
        base_P = self.base_case["active_power"][t]
        Mp_u = self.matrices_pv["M_p_utility"]
        Mp_c = self.matrices_pv["M_p_customer"]
        lhs = base_P
        if self.num_utility_pv > 0:
            lhs = lhs + Mp_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mp_c, True) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_P_p"] @ p_bt + self.matrices_batt["M_P_q"] @ q_bt
        if np.any(lhs > self.limits["S_ratings"] + tol):
            return True

        lhs = base_P
        if self.num_utility_pv > 0:
            lhs = lhs + Mp_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mp_c, False) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_P_p"] @ p_bt + self.matrices_batt["M_P_q"] @ q_bt
        if np.any(lhs < -self.limits["S_ratings"] - tol):
            return True

        # Reactive flow
        base_Q = self.base_case["reactive_power"][t]
        Mq_u = self.matrices_pv["M_q_utility"]
        Mq_c = self.matrices_pv["M_q_customer"]
        lhs = base_Q
        if self.num_utility_pv > 0:
            lhs = lhs + Mq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mq_c, True) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_Q_p"] @ p_bt + self.matrices_batt["M_Q_q"] @ q_bt
        if np.any(lhs > self.limits["S_ratings"] + tol):
            return True

        lhs = base_Q
        if self.num_utility_pv > 0:
            lhs = lhs + Mq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mq_c, False) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + self.matrices_batt["M_Q_p"] @ p_bt + self.matrices_batt["M_Q_q"] @ q_bt
        if np.any(lhs < -self.limits["S_ratings"] - tol):
            return True

        # Approx constraints
        sqrt2S = self.limits["sqrt2_S_ratings"]
        base_PQ = base_P + base_Q
        Mpq_u = Mp_u + Mq_u
        Mpq_c = Mp_c + Mq_c
        lhs = base_PQ
        if self.num_utility_pv > 0:
            lhs = lhs + Mpq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mpq_c, True) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + (self.matrices_batt["M_P_p"] + self.matrices_batt["M_Q_p"]) @ p_bt + \
                  (self.matrices_batt["M_P_q"] + self.matrices_batt["M_Q_q"]) @ q_bt
        if np.any(lhs > sqrt2S + tol):
            return True
        lhs = base_PQ
        if self.num_utility_pv > 0:
            lhs = lhs + Mpq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mpq_c, False) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + (self.matrices_batt["M_P_p"] + self.matrices_batt["M_Q_p"]) @ p_bt + \
                  (self.matrices_batt["M_P_q"] + self.matrices_batt["M_Q_q"]) @ q_bt
        if np.any(lhs < -sqrt2S - tol):
            return True

        base_PmQ = base_P - base_Q
        Mpmq_u = Mp_u - Mq_u
        Mpmq_c = Mp_c - Mq_c
        lhs = base_PmQ
        if self.num_utility_pv > 0:
            lhs = lhs + Mpmq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mpmq_c, True) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + (self.matrices_batt["M_P_p"] - self.matrices_batt["M_Q_p"]) @ p_bt + \
                  (self.matrices_batt["M_P_q"] - self.matrices_batt["M_Q_q"]) @ q_bt
        if np.any(lhs > sqrt2S + tol):
            return True
        lhs = base_PmQ
        if self.num_utility_pv > 0:
            lhs = lhs + Mpmq_u @ util_scaled
        if self.num_customer_pv > 0:
            lhs = lhs + robust(Mpmq_c, False) @ cust_scaled
        if self.num_battery > 0:
            lhs = lhs + (self.matrices_batt["M_P_p"] - self.matrices_batt["M_Q_p"]) @ p_bt + \
                  (self.matrices_batt["M_P_q"] - self.matrices_batt["M_Q_q"]) @ q_bt
        if np.any(lhs < -sqrt2S - tol):
            return True

        return False

    def _find_violated_constraints(
        self,
        solution: np.ndarray,
        tolerance: float,
        exclude: Set[int]
    ) -> Set[int]:
        violated = set()
        parts = self._unpack_solution(solution)

        n_elem = self._num_elements
        for t in range(self._T):
            pv_t = self.pv_profiles_processed[t]
            zeta_u = parts["zeta_u"]
            zeta_cbar = parts["zeta_cbar"]
            if self.num_battery > 0:
                tbatt = t % self._battery_period
                p_bt = parts["p_b"][tbatt]
                q_bt = parts["q_b"][tbatt]
            else:
                p_bt = None
                q_bt = None

            util_scaled = zeta_u * pv_t[:self.num_utility_pv] if self.num_utility_pv > 0 else np.zeros(0)
            cust_scaled = zeta_cbar * pv_t[self.num_utility_pv:self.num_utility_pv+self.num_customer_pv] \
                if self.num_customer_pv > 0 else np.zeros(0)

            base_offset = t * self.num_constraints_per_scenario

            def robust(M, upper: bool):
                if not self.is_robust:
                    return M
                return np.maximum(M, 0) if upper else np.minimum(M, 0)

            # Helpers to assemble vector LHS quickly
            def assemble(base, M_u, M_c, upper, M_bp, M_bq):
                lhs = base
                if self.num_utility_pv > 0:
                    lhs = lhs + M_u @ util_scaled
                if self.num_customer_pv > 0:
                    lhs = lhs + robust(M_c, upper) @ cust_scaled
                if self.num_battery > 0:
                    lhs = lhs + M_bp @ p_bt + M_bq @ q_bt
                return lhs

            # Voltage upper/lower
            lhs = assemble(self.base_case["voltage"][t], self.matrices_pv["M_v_utility"], self.matrices_pv["M_v_customer"],
                           True, self.matrices_batt["M_v_p"], self.matrices_batt["M_v_q"])
            mask = lhs > self.limits["voltage_upper"] + tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 0 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            lhs = assemble(self.base_case["voltage"][t], self.matrices_pv["M_v_utility"], self.matrices_pv["M_v_customer"],
                           False, self.matrices_batt["M_v_p"], self.matrices_batt["M_v_q"])
            mask = lhs < self.limits["voltage_lower"] - tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 1 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            # Active upper/lower
            lhs = assemble(self.base_case["active_power"][t], self.matrices_pv["M_p_utility"], self.matrices_pv["M_p_customer"],
                           True, self.matrices_batt["M_P_p"], self.matrices_batt["M_P_q"])
            mask = lhs > self.limits["S_ratings"] + tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 2 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            lhs = assemble(self.base_case["active_power"][t], self.matrices_pv["M_p_utility"], self.matrices_pv["M_p_customer"],
                           False, self.matrices_batt["M_P_p"], self.matrices_batt["M_P_q"])
            mask = lhs < -self.limits["S_ratings"] - tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 3 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            # Reactive upper/lower
            lhs = assemble(self.base_case["reactive_power"][t], self.matrices_pv["M_q_utility"], self.matrices_pv["M_q_customer"],
                           True, self.matrices_batt["M_Q_p"], self.matrices_batt["M_Q_q"])
            mask = lhs > self.limits["S_ratings"] + tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 4 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            lhs = assemble(self.base_case["reactive_power"][t], self.matrices_pv["M_q_utility"], self.matrices_pv["M_q_customer"],
                           False, self.matrices_batt["M_Q_p"], self.matrices_batt["M_Q_q"])
            mask = lhs < -self.limits["S_ratings"] - tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 5 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            # Approx P+Q upper/lower
            base_pq = self.base_case["active_power"][t] + self.base_case["reactive_power"][t]
            M_u = self.matrices_pv["M_p_utility"] + self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_p_customer"] + self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_P_p"] + self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_P_q"] + self.matrices_batt["M_Q_q"]
            lhs = assemble(base_pq, M_u, M_c, True, M_bp, M_bq)
            mask = lhs > self.limits["sqrt2_S_ratings"] + tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 6 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)
            lhs = assemble(base_pq, M_u, M_c, False, M_bp, M_bq)
            mask = lhs < -self.limits["sqrt2_S_ratings"] - tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 7 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

            # Approx P-Q upper/lower
            base_pmq = self.base_case["active_power"][t] - self.base_case["reactive_power"][t]
            M_u = self.matrices_pv["M_p_utility"] - self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_p_customer"] - self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_P_p"] - self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_P_q"] - self.matrices_batt["M_Q_q"]
            lhs = assemble(base_pmq, M_u, M_c, True, M_bp, M_bq)
            mask = lhs > self.limits["sqrt2_S_ratings"] + tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 8 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)
            lhs = assemble(base_pmq, M_u, M_c, False, M_bp, M_bq)
            mask = lhs < -self.limits["sqrt2_S_ratings"] - tolerance
            for elem in np.where(mask)[0]:
                idx = base_offset + 9 * n_elem + elem
                if idx not in exclude:
                    violated.add(idx)

        return violated

    # --------------------------------------------------------------------------
    # Constraint-based helpers (network rows)
    # --------------------------------------------------------------------------
    def _decode_constraint_index(self, idx: int) -> Tuple[int, str, int]:
        n_per = self.num_constraints_per_scenario
        t = idx // n_per
        rem = idx % n_per
        type_idx = rem // self._num_elements
        elem = rem % self._num_elements

        constraint_types = [
            "voltage_upper", "voltage_lower",
            "active_flow_upper", "active_flow_lower",
            "reactive_flow_upper", "reactive_flow_lower",
            "appr_flow_upper", "appr_flow_lower",
            "appr_diff_flow_upper", "appr_diff_flow_lower",
        ]
        return t, constraint_types[type_idx], elem

    def _build_constraint_matrices(
        self,
        constraint_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        A_list: List[np.ndarray] = []
        b_list: List[np.ndarray] = []
        ctype_list: List[str] = []
        for idx in constraint_indices:
            t, ctype, elem = self._decode_constraint_index(idx)
            A_row, b_val, _ = self._get_constraint_row(t, ctype, elem)
            A_list.append(A_row)
            b_list.append(np.array([b_val]))
            ctype_list.append(ctype)
        return A_list, b_list, ctype_list

    def _get_constraint_row(self, t: int, ctype: str, elem: int) -> Tuple[np.ndarray, float, bool]:
        pv_t = self.pv_profiles_processed[t]
        is_upper = "upper" in ctype

        A_row = np.zeros(self._num_variables)

        # Choose base, matrices, and limit
        if "voltage" in ctype:
            base_val = self.base_case["voltage"][t, elem]
            M_u = self.matrices_pv["M_v_utility"]
            M_c = self.matrices_pv["M_v_customer"]
            M_bp = self.matrices_batt["M_v_p"]
            M_bq = self.matrices_batt["M_v_q"]
            limit = self.limits["voltage_upper"][elem] if is_upper else self.limits["voltage_lower"][elem]
        elif "active_flow" in ctype:
            base_val = self.base_case["active_power"][t, elem]
            M_u = self.matrices_pv["M_p_utility"]
            M_c = self.matrices_pv["M_p_customer"]
            M_bp = self.matrices_batt["M_P_p"]
            M_bq = self.matrices_batt["M_P_q"]
            limit = self.limits["S_ratings"][elem] if is_upper else -self.limits["S_ratings"][elem]
        elif "reactive_flow" in ctype:
            base_val = self.base_case["reactive_power"][t, elem]
            M_u = self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_Q_q"]
            limit = self.limits["S_ratings"][elem] if is_upper else -self.limits["S_ratings"][elem]
        elif "appr_flow" in ctype and "diff" not in ctype:
            base_val = self.base_case["active_power"][t, elem] + self.base_case["reactive_power"][t, elem]
            M_u = self.matrices_pv["M_p_utility"] + self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_p_customer"] + self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_P_p"] + self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_P_q"] + self.matrices_batt["M_Q_q"]
            limit = self.limits["sqrt2_S_ratings"][elem] if is_upper else -self.limits["sqrt2_S_ratings"][elem]
        elif "appr_diff_flow" in ctype:
            base_val = self.base_case["active_power"][t, elem] - self.base_case["reactive_power"][t, elem]
            M_u = self.matrices_pv["M_p_utility"] - self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_p_customer"] - self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_P_p"] - self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_P_q"] - self.matrices_batt["M_Q_q"]
            limit = self.limits["sqrt2_S_ratings"][elem] if is_upper else -self.limits["sqrt2_S_ratings"][elem]
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")

        # Robust customer matrix
        if self.num_customer_pv > 0:
            if self.is_robust:
                M_c_robust = np.maximum(M_c, 0) if is_upper else np.minimum(M_c, 0)
            else:
                M_c_robust = M_c
        else:
            M_c_robust = np.zeros((self._num_elements, 0))

        # Fill coefficients into A_row for this element
        # zeta_u block
        if self.num_utility_pv > 0:
            A_row[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv] = \
                M_u[elem, :] * pv_t[:self.num_utility_pv]

        # zeta_cbar block
        if self.num_customer_pv > 0:
            A_row[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv] = \
                M_c_robust[elem, :] * pv_t[self.num_utility_pv:self.num_utility_pv+self.num_customer_pv]

        # battery dispatch blocks
        if self.num_battery > 0:
            tbatt = t % self._battery_period
            p_off = self._offsets["p_b"] + tbatt * self.num_battery
            q_off = self._offsets["q_b"] + tbatt * self.num_battery
            A_row[p_off:p_off + self.num_battery] = M_bp[elem, :]
            A_row[q_off:q_off + self.num_battery] = M_bq[elem, :]

        # b such that A_row @ x <= b for upper, >= b for lower.
        b_val = limit - base_val
        return A_row, float(b_val), is_upper

    def _find_active_constraints(
        self,
        solution: np.ndarray,
        constraint_indices: List[int],
        tolerance: float
    ) -> Set[int]:
        active = set()
        for idx in constraint_indices:
            t, ctype, elem = self._decode_constraint_index(idx)
            A_row, b_val, _ = self._get_constraint_row(t, ctype, elem)
            lhs = float(A_row @ solution)
            if abs(lhs - b_val) < tolerance:
                active.add(idx)
        return active

    # --------------------------------------------------------------------------
    # Packing/unpacking and global constraints for vector form
    # --------------------------------------------------------------------------
    def _build_variable_offsets(self) -> None:
        """Define a fixed ordering for the solution vector."""
        off = 0
        self._offsets = {}

        self._offsets["zeta_u"] = off
        off += self.num_utility_pv

        self._offsets["zeta_cbar"] = off
        off += self.num_customer_pv

        self._offsets["zeta_b"] = off
        off += self.num_battery

        self._offsets["p_b"] = off
        off += self._battery_period * self.num_battery

        self._offsets["q_b"] = off
        off += self._battery_period * self.num_battery

        self._offsets["soc"] = off
        off += self._battery_period * self.num_battery

        self._num_variables = off

    def _pack_solution(
        self,
        zeta_u: Optional[cp.Variable],
        zeta_cbar: Optional[cp.Variable],
        zeta_b: Optional[cp.Variable],
        p_b: Optional[cp.Variable],
        q_b: Optional[cp.Variable],
        soc: Optional[cp.Variable]
    ) -> np.ndarray:
        x = np.zeros(self._num_variables, dtype=float)

        if self.num_utility_pv > 0:
            x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv] = np.array(zeta_u.value).reshape(-1)

        if self.num_customer_pv > 0:
            x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv] = np.array(zeta_cbar.value).reshape(-1)

        if self.num_battery > 0:
            x[self._offsets["zeta_b"]:self._offsets["zeta_b"] + self.num_battery] = np.array(zeta_b.value).reshape(-1)
            x[self._offsets["p_b"]:self._offsets["p_b"] + self._battery_period * self.num_battery] = np.array(p_b.value).reshape(-1)
            x[self._offsets["q_b"]:self._offsets["q_b"] + self._battery_period * self.num_battery] = np.array(q_b.value).reshape(-1)
            x[self._offsets["soc"]:self._offsets["soc"] + self._battery_period * self.num_battery] = np.array(soc.value).reshape(-1)

        return x

    def _unpack_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.array(x).reshape(-1)
        out: Dict[str, np.ndarray] = {}
        out["zeta_u"] = x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv] \
            if self.num_utility_pv > 0 else np.zeros(0)
        out["zeta_cbar"] = x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv] \
            if self.num_customer_pv > 0 else np.zeros(0)
        out["zeta_b"] = x[self._offsets["zeta_b"]:self._offsets["zeta_b"] + self.num_battery] \
            if self.num_battery > 0 else np.zeros(0)

        if self.num_battery > 0:
            out["p_b"] = x[self._offsets["p_b"]:self._offsets["p_b"] + self._battery_period * self.num_battery].reshape(self._battery_period, self.num_battery)
            out["q_b"] = x[self._offsets["q_b"]:self._offsets["q_b"] + self._battery_period * self.num_battery].reshape(self._battery_period, self.num_battery)
            out["soc"] = x[self._offsets["soc"]:self._offsets["soc"] + self._battery_period * self.num_battery].reshape(self._battery_period, self.num_battery)
        else:
            out["p_b"] = np.zeros((self._battery_period, 0))
            out["q_b"] = np.zeros((self._battery_period, 0))
            out["soc"] = np.zeros((self._battery_period, 0))
        return out

    def _objective_from_x(self, x: cp.Variable) -> cp.Expression:
        expr = 0
        if self.num_utility_pv > 0:
            expr += cp.sum(x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv])
        if self.num_customer_pv > 0:
            expr += cp.sum(x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv])
        if self.num_battery > 0 and self.battery_cost != 0.0:
            expr -= self.battery_cost * cp.sum(x[self._offsets["zeta_b"]:self._offsets["zeta_b"] + self.num_battery])

        if self.with_fairness:
            terms = []
            if self.num_utility_pv > 0:
                terms.append(cp.sum(cp.log(x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv] + 1e-5)))
            if self.num_customer_pv > 0:
                terms.append(cp.sum(cp.log(x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv] + 1e-5)))
            if len(terms) > 0:
                expr += self.fairness_factor * cp.sum(terms)
        return expr

    def _global_constraints_from_x(self, x: cp.Variable) -> List:
        """Global constraints that are always included (PV & battery bounds + SoC)."""
        cons: List = []

        # PV nonneg
        if self.num_utility_pv > 0:
            z_u = x[self._offsets["zeta_u"]:self._offsets["zeta_u"] + self.num_utility_pv]
            cons += [z_u >= 0, z_u <= self.capacity_limits_utility]
        if self.num_customer_pv > 0:
            z_c = x[self._offsets["zeta_cbar"]:self._offsets["zeta_cbar"] + self.num_customer_pv]
            cons += [z_c >= 0, z_c <= self.capacity_limits_customer]

        # Battery constraints
        if self.num_battery > 0:
            z_b = x[self._offsets["zeta_b"]:self._offsets["zeta_b"] + self.num_battery]
            if self.zeta_b_binary:
                # In vector form we cannot enforce integrality; scenario-based is recommended for MILP.
                # We still keep bounds to make continuous relaxation usable.
                cons += [z_b >= 0, z_b <= 1]
            else:
                cons += [z_b >= 0, z_b <= 1]

            # Battery decision variables are defined over one daily period (self._battery_period),
            # and are reused across multiple-day scenario indices via t % self._battery_period.
            Tb = self._battery_period
            p_b = cp.reshape(x[self._offsets["p_b"]:self._offsets["p_b"] + Tb * self.num_battery], (Tb, self.num_battery), order='C')
            q_b = cp.reshape(x[self._offsets["q_b"]:self._offsets["q_b"] + Tb * self.num_battery], (Tb, self.num_battery), order='C')
            soc = cp.reshape(x[self._offsets["soc"]:self._offsets["soc"] + Tb * self.num_battery], (Tb, self.num_battery), order='C')

            pmin = self.batt_bounds["p_min"]
            pmax = self.batt_bounds["p_max"]
            qmin = self.batt_bounds["q_min"]
            qmax = self.batt_bounds["q_max"]
            socmin = self.batt_bounds["soc_min"]
            socmax = self.batt_bounds["soc_max"]

            cons += [
                p_b >= cp.multiply(np.ones((Tb, 1)) @ pmin.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
                p_b <= cp.multiply(np.ones((Tb, 1)) @ pmax.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
                q_b >= cp.multiply(np.ones((Tb, 1)) @ qmin.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
                q_b <= cp.multiply(np.ones((Tb, 1)) @ qmax.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
            ]

            if Tb > 1:
                cons.append(soc[1:, :] == soc[:-1, :] - self.delta_t_hours * p_b[1:, :])
            cons.append(soc[0, :] == soc[-1, :] - self.delta_t_hours * p_b[0, :])

            cons += [
                soc >= cp.multiply(np.ones((Tb, 1)) @ socmin.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
                soc <= cp.multiply(np.ones((Tb, 1)) @ socmax.reshape(1, -1), cp.reshape(z_b, (1, self.num_battery), order='C')),
            ]
        
            cons += [cp.sum(z_b) <= self.total_capacity_limit]

        return cons

    def _extract_support_by_activity(self, solution: np.ndarray, scenario_indices: List[int], tol: float = 1e-6) -> Set[int]:
        """Heuristic support extraction via constraint activity (MILP-safe)."""
        parts = self._unpack_solution(solution)
        support = set()
        for t in scenario_indices:
            if self._scenario_is_active(t, parts, tol):
                support.add(t)
        if len(support) == 0:
            support = set(scenario_indices)
        return support

    def _scenario_is_active(self, t: int, parts: Dict[str, np.ndarray], tol: float) -> bool:
        # If any network constraint is within tol of its bound, consider it support.
        pv_t = self.pv_profiles_processed[t]
        zeta_u = parts["zeta_u"]
        zeta_cbar = parts["zeta_cbar"]
        if self.num_battery > 0:
            t_batt = t % self._battery_period
            p_bt = parts["p_b"][t_batt]
            q_bt = parts["q_b"][t_batt]
        else:
            p_bt = None
            q_bt = None

        util_scaled = zeta_u * pv_t[:self.num_utility_pv] if self.num_utility_pv > 0 else np.zeros(0)
        cust_scaled = zeta_cbar * pv_t[self.num_utility_pv:self.num_utility_pv+self.num_customer_pv] \
            if self.num_customer_pv > 0 else np.zeros(0)

        def robust(M, upper: bool):
            if not self.is_robust:
                return M
            return np.maximum(M, 0) if upper else np.minimum(M, 0)

        def lhs(base, M_u, M_c, upper, M_bp, M_bq, lim, is_upper):
            val = base
            if self.num_utility_pv > 0:
                val = val + M_u @ util_scaled
            if self.num_customer_pv > 0:
                val = val + robust(M_c, upper) @ cust_scaled
            if self.num_battery > 0:
                val = val + M_bp @ p_bt + M_bq @ q_bt
            # check activity
            if is_upper:
                return np.any(np.abs(val - lim) < tol)
            return np.any(np.abs(val - lim) < tol)

        # voltage bounds
        if lhs(self.base_case["voltage"][t], self.matrices_pv["M_v_utility"], self.matrices_pv["M_v_customer"],
               True, self.matrices_batt["M_v_p"], self.matrices_batt["M_v_q"], self.limits["voltage_upper"], True):
            return True
        if lhs(self.base_case["voltage"][t], self.matrices_pv["M_v_utility"], self.matrices_pv["M_v_customer"],
               False, self.matrices_batt["M_v_p"], self.matrices_batt["M_v_q"], self.limits["voltage_lower"], False):
            return True

        # active
        if lhs(self.base_case["active_power"][t], self.matrices_pv["M_p_utility"], self.matrices_pv["M_p_customer"],
               True, self.matrices_batt["M_P_p"], self.matrices_batt["M_P_q"], self.limits["S_ratings"], True):
            return True
        if lhs(self.base_case["active_power"][t], self.matrices_pv["M_p_utility"], self.matrices_pv["M_p_customer"],
               False, self.matrices_batt["M_P_p"], self.matrices_batt["M_P_q"], -self.limits["S_ratings"], False):
            return True

        # reactive
        if lhs(self.base_case["reactive_power"][t], self.matrices_pv["M_q_utility"], self.matrices_pv["M_q_customer"],
               True, self.matrices_batt["M_Q_p"], self.matrices_batt["M_Q_q"], self.limits["S_ratings"], True):
            return True
        if lhs(self.base_case["reactive_power"][t], self.matrices_pv["M_q_utility"], self.matrices_pv["M_q_customer"],
               False, self.matrices_batt["M_Q_p"], self.matrices_batt["M_Q_q"], -self.limits["S_ratings"], False):
            return True

        # linearized apparent power, P+Q and P-Q (constraint types 6-9)
        for sign in (1, -1):
            base = self.base_case["active_power"][t] + sign * self.base_case["reactive_power"][t]
            M_u = self.matrices_pv["M_p_utility"] + sign * self.matrices_pv["M_q_utility"]
            M_c = self.matrices_pv["M_p_customer"] + sign * self.matrices_pv["M_q_customer"]
            M_bp = self.matrices_batt["M_P_p"] + sign * self.matrices_batt["M_Q_p"]
            M_bq = self.matrices_batt["M_P_q"] + sign * self.matrices_batt["M_Q_q"]
            if lhs(base, M_u, M_c, True, M_bp, M_bq, self.limits["sqrt2_S_ratings"], True):
                return True
            if lhs(base, M_u, M_c, False, M_bp, M_bq, -self.limits["sqrt2_S_ratings"], False):
                return True

        return False

    # --------------------------------------------------------------------------
    # Debugging Line Capacity
    # --------------------------------------------------------------------------
    def _check_basic_feasibility(self, max_rounds: int = 10) -> bool:
        """
        Validate that the network is feasible with existing customer load alone
        (zero PV, zero battery) -- see `PVHCProblemWithoutBattery._check_basic_feasibility`
        for why this can legitimately happen on a real feeder and how the
        adaptive relaxation below is meant to be interpreted.
        """
        if self.base_case is None or self.limits is None:
            raise RuntimeError("Problem data incomplete; compile() must populate base_case and limits before feasibility check.")

        zero_solution = np.zeros(self._num_variables, dtype=float)

        for round_idx in range(max_rounds):
            violated_constraints = self._find_violated_constraints(
                zero_solution, tolerance=1e-6, exclude=set()
            )
            if not violated_constraints:
                if round_idx > 0:
                    print(f"[Adaptive relaxation] Network is now feasible after {round_idx} round(s) of relaxation.")
                return True

            print(f"[Base feasibility] Round {round_idx + 1}/{max_rounds}: "
                  f"{len(violated_constraints)} violation(s) detected with zero PV and battery "
                  f"(native customer load alone). First few (of possibly many, scenario x row pairs):")
            for idx in sorted(violated_constraints)[:5]:
                t, ctype, elem = self._decode_constraint_index(idx)
                desc = self._describe_violation_at_zero_pv(t, ctype, elem)
                direction = "upper" if desc["is_upper"] else "lower"
                print(
                    f"  scenario={t}, constraint={ctype}, row={elem}, "
                    f"value={desc['lhs']:.6f}, limit={desc['limit']:.6f}, "
                    f"margin={desc['margin']:.6f} (relax {direction} bound at row {elem})"
                )
            if len(violated_constraints) > 5:
                print(f"  ... and {len(violated_constraints) - 5} more.")

            if not self.adaptive_flag:
                return False

            self._relax_limits_adaptively(violated_constraints)

        print(f"[Adaptive relaxation] Still has violations after {max_rounds} round(s); giving up.")
        return False

    def _describe_violation_at_zero_pv(self, t: int, ctype: str, elem: int) -> Dict[str, Any]:
        """Return lhs, limit, margin, and direction for a violated constraint at zero PV and battery."""
        # Base values with zero PV injections and no batteries
        base_v = self.base_case['voltage'][t]
        base_P = self.base_case['active_power'][t]
        base_Q = self.base_case['reactive_power'][t]

        if 'voltage' in ctype:
            lhs_val = base_v[elem]
            limit = self.limits['voltage_upper'][elem] if 'upper' in ctype else self.limits['voltage_lower'][elem]
            is_upper = 'upper' in ctype
        elif 'active_flow' in ctype:
            lhs_val = base_P[elem]
            limit = self.limits['S_ratings'][elem] if 'upper' in ctype else -self.limits['S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'reactive_flow' in ctype:
            lhs_val = base_Q[elem]
            limit = self.limits['S_ratings'][elem] if 'upper' in ctype else -self.limits['S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'appr_flow' in ctype and 'diff' not in ctype:
            lhs_val = base_P[elem] + base_Q[elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if 'upper' in ctype else -self.limits['sqrt2_S_ratings'][elem]
            is_upper = 'upper' in ctype
        elif 'appr_diff_flow' in ctype:
            lhs_val = base_P[elem] - base_Q[elem]
            limit = self.limits['sqrt2_S_ratings'][elem] if 'upper' in ctype else -self.limits['sqrt2_S_ratings'][elem]
            is_upper = 'upper' in ctype
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")

        margin = lhs_val - limit if is_upper else limit - lhs_val

        return {
            'lhs': float(lhs_val),
            'limit': float(limit),
            'margin': float(margin),
            'is_upper': is_upper,
        }

    def _relax_limits_adaptively(self, violated_constraints: Set[int]) -> None:
        """
        Relax limits adaptively based on maximum relaxation factors.
        
        For each constraint type and element, compute the relaxation factor needed
        to satisfy the violation: ratio = (1 + factor * margin / old_limit).
        Then take the maximum relaxation factor across all constraint types for each element.
        Since this is a three-phase system, the relaxation applies to all three phases 
        of each bus (elements sharing the same bus).
        
        Importantly: S_ratings and sqrt2_S_ratings are related (sqrt2_S_ratings = sqrt(2)*S_ratings).
        By using relaxation factors instead of margins, we ensure that the same relaxation
        ratio is applied to S_ratings, maintaining the sqrt(2) relationship automatically.
        
        Args:
            violated_constraints: Set of violated constraint indices
        """
        relaxation_factor = self.DEFAULT_RELAXATION_FACTOR
        
        # Record maximum relaxation ratio for each element by limit type
        voltage_upper_ratio = {}  # {element: max_ratio}
        voltage_lower_ratio = {}  # {element: max_ratio}
        s_ratings_ratio = {}      # {element: max_ratio}
        
        for idx in violated_constraints:
            t, ctype, elem = self._decode_constraint_index(idx)
            desc = self._describe_violation_at_zero_pv(t, ctype, elem)
            margin = desc['margin']
            limit = desc['limit']

            # Compute relaxation ratio for this violation
            ratio = 1 + relaxation_factor * margin / abs(limit)

            # Key by bus (not the raw per-phase element index): a violation on
            # any one phase relaxes all 3 phases of that bus together, so two
            # violations on different phases of the SAME bus must land in the
            # SAME dict entry (via max()) rather than each independently
            # triggering their own full 3-phase relaxation pass below -- the
            # latter would silently double- (or triple-) relax that bus.
            bus_idx = elem // 3

            if 'voltage' in ctype:
                if 'upper' in ctype:
                    voltage_upper_ratio[bus_idx] = max(voltage_upper_ratio.get(bus_idx, 1.0), ratio)
                else:
                    voltage_lower_ratio[bus_idx] = max(voltage_lower_ratio.get(bus_idx, 1.0), ratio)
            else:
                # All flow-based constraints use S_ratings or sqrt2_S_ratings
                # Record the max ratio - will apply to S_ratings
                s_ratings_ratio[bus_idx] = max(s_ratings_ratio.get(bus_idx, 1.0), ratio)
        
        # Apply voltage relaxations (print only a few examples -- with thousands of
        # scenarios this can touch every bus, and printing all of them just floods
        # the log without adding information beyond the summary count).
        _MAX_RELAX_PRINTS = 5

        n_printed = 0
        for bus_idx, ratio in voltage_upper_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_limit = self.limits['voltage_upper'][idx]
                # Upper bound: multiplying by ratio > 1 raises it, i.e. relaxes it.
                new_limit = old_limit * ratio
                self.limits['voltage_upper'][idx] = new_limit
                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed voltage_upper[{idx}]: {old_limit:.6f} -> {new_limit:.6f} (ratio={ratio:.4f})")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(voltage_upper_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... voltage_upper relaxed for {len(voltage_upper_ratio) * 3} element(s) total.")

        n_printed = 0
        for bus_idx, ratio in voltage_lower_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_limit = self.limits['voltage_lower'][idx]
                # Lower bound: relaxing it means making it MORE permissive, i.e.
                # LOWER, not higher -- dividing by ratio > 1 decreases it. (Multiplying,
                # as with the upper bound / S_ratings above, would tighten a lower
                # bound instead of relaxing it, and could never converge to feasible.)
                new_limit = old_limit / ratio
                self.limits['voltage_lower'][idx] = new_limit
                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed voltage_lower[{idx}]: {old_limit:.6f} -> {new_limit:.6f} (ratio={ratio:.4f})")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(voltage_lower_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... voltage_lower relaxed for {len(voltage_lower_ratio) * 3} element(s) total.")

        # Apply S_ratings relaxations (and maintain sqrt2_S_ratings = sqrt(2) * S_ratings)
        n_printed = 0
        for bus_idx, ratio in s_ratings_ratio.items():
            for ph in range(3):
                idx = bus_idx * 3 + ph
                old_s_limit = self.limits['S_ratings'][idx]
                new_s_limit = old_s_limit * ratio
                self.limits['S_ratings'][idx] = new_s_limit

                # Maintain sqrt2_S_ratings = sqrt(2) * S_ratings
                old_sqrt2_limit = self.limits['sqrt2_S_ratings'][idx]
                new_sqrt2_limit = np.sqrt(2) * new_s_limit
                self.limits['sqrt2_S_ratings'][idx] = new_sqrt2_limit

                if n_printed < _MAX_RELAX_PRINTS:
                    print(f"    Relaxed S_ratings[{idx}]: {old_s_limit:.6f} -> {new_s_limit:.6f} (ratio={ratio:.4f})")
                    print(f"    Relaxed sqrt2_S_ratings[{idx}]: {old_sqrt2_limit:.6f} -> {new_sqrt2_limit:.6f}")
                    n_printed += 1
        if n_printed >= _MAX_RELAX_PRINTS and len(s_ratings_ratio) * 3 > _MAX_RELAX_PRINTS:
            print(f"    ... S_ratings relaxed for {len(s_ratings_ratio) * 3} element(s) total.")

    # --------------------------------------------------------------------------
    # Matrix construction helpers (copied/adapted from PVHCProblemWithoutBattery)
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

    def _compute_network_limits(self) -> Dict[str, np.ndarray]:
        nominal_voltage_sq = self._get_nominal_voltage_squared()
        voltage_upper = nominal_voltage_sq * (1 + self.vol_tol) ** 2
        voltage_lower = nominal_voltage_sq * (1 - self.lower_voltage_limit_ratio * self.vol_tol) ** 2

        n_buses = self.network_data["n_buses"]
        raw_ratings = self.network_data["branch_pwr_ratings"]
        processed_ratings = [r if (r is not None and r != 0) else 1e6 for r in raw_ratings[:n_buses]]
        # Use the equipment's real OpenDSS-derived thermal rating as-is -- no blanket
        # safety margin. `relaxation_factor` is reserved for the *adaptive*, per-branch
        # relaxation in _relax_limits_adaptively(), which only stretches a rating when
        # the base (zero-PV) case is actually found infeasible against it.
        S_ratings = np.repeat(processed_ratings, 3)

        return {
            "voltage_upper": voltage_upper,
            "voltage_lower": voltage_lower,
            "S_ratings": S_ratings,
            "sqrt2_S_ratings": np.sqrt(2) * S_ratings
        }

    def _get_nominal_voltage_squared(self) -> np.ndarray:
        N = self.network_data["n_buses"]
        C_tilde_0 = self.network_data["C_tilde_0"]

        v0_ref_squared = ldf.compute_v0_ref(self.v_root_nom * np.ones(3)).swapaxes(0, 1).reshape(9, 1)
        nominal_squared = (C_tilde_0.reshape(N, 3, 9) @ v0_ref_squared).reshape(N * 3)

        assert np.imag(nominal_squared).max() < 1e-6, "Nominal voltages should be real"
        return np.real(nominal_squared)

    def _compute_pv_impact_matrices(self) -> Dict[str, np.ndarray]:
        M_v_u, M_p_u, M_q_u = self._construct_power_injection_matrices(self.utility_pv_info)
        M_v_c, M_p_c, M_q_c = self._construct_power_injection_matrices(self.customer_pv_info)
        return {
            "M_v_utility": M_v_u,
            "M_p_utility": M_p_u,
            "M_q_utility": M_q_u,
            "M_v_customer": M_v_c,
            "M_p_customer": M_p_c,
            "M_q_customer": M_q_c,
        }

    def _construct_power_injection_matrices(
        self,
        dev_info: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct matrices mapping a scalar device injection variable to voltage/P/Q impacts.
        This is identical to PVHCProblemWithoutBattery's construct_pv_matrix and supports
        both grounded and delta connections.
        """
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
                    1e-3 * R[:, idx0:idx0+3] @ np.real(trans_vec) +
                    1e-3 * X[:, idx0:idx0+3] @ np.imag(trans_vec)
                )
                branch_pwr = C_tilde_S[:, idx0:idx0+3] @ trans_vec
                M_p[:, i] = np.real(branch_pwr)
                M_q[:, i] = np.imag(branch_pwr)

        return M_v, M_p, M_q

    def _compute_battery_impact_matrices(self) -> Dict[str, np.ndarray]:
        """
        Construct matrices mapping battery p_b and q_b dispatch to voltage and branch P/Q.
        For delta-connected batteries, we model the complex mapping s_delta = p + j q with
        s_wye = trans_vec * s_delta, giving coupling between p and q in the wye frame.
        """
        n_b = self.num_battery
        if n_b == 0:
            z = np.zeros((self._num_elements, 0))
            return {
                "M_v_p": z, "M_v_q": z,
                "M_P_p": z, "M_P_q": z,
                "M_Q_p": z, "M_Q_q": z,
            }

        R = self.network_data["R"]
        X = self.network_data["X"]
        C_tilde_S = self.network_data["C_tilde_S"]
        node2idx = self.network_data["node2idx"]
        n_buses = self.network_data["n_buses"]

        M_v_p = np.zeros((n_buses * 3, n_b))
        M_v_q = np.zeros((n_buses * 3, n_b))
        M_P_p = np.zeros((n_buses * 3, n_b))
        M_P_q = np.zeros((n_buses * 3, n_b))
        M_Q_p = np.zeros((n_buses * 3, n_b))
        M_Q_q = np.zeros((n_buses * 3, n_b))

        for i, info in enumerate(self.utility_battery_info):
            bus_name = info["bus"]
            phases = info["phases"]
            if bus_name not in node2idx:
                raise KeyError(f"Bus '{bus_name}' not found in node2idx")
            bus_idx = node2idx[bus_name]

            if phases[1] == 0:
                # grounded phase connection: no coupling at the bus
                idx = (bus_idx - 1) * 3 + phases[0] - 1
                # voltage: v = ... + R*p + X*q  (scaled)
                M_v_p[:, i] = np.real(1e-3 * R[:, idx])
                M_v_q[:, i] = np.real(1e-3 * X[:, idx])
                # branch flows: P + jQ = C_tilde_S (p + j q). The cross terms are nonzero
                # only below a delta winding, where C_tilde_S is complex.
                M_P_p[:, i] = np.real(C_tilde_S[:, idx])
                M_P_q[:, i] = -np.imag(C_tilde_S[:, idx])
                M_Q_p[:, i] = np.imag(C_tilde_S[:, idx])
                M_Q_q[:, i] = np.real(C_tilde_S[:, idx])
            else:
                phase_idx = ldf.STD_DELTA_ORDER.index(tuple(sorted(phases)))
                trans_vec = self._get_transformation_vector(info.get("connected_transformer", {}), phase_idx)
                idx0 = (bus_idx - 1) * 3

                # s_wye = trans_vec * (p + j q)
                a = np.real(trans_vec).reshape(-1)  # 3
                b = np.imag(trans_vec).reshape(-1)  # 3
                # nodal p and q coefficients in wye frame
                # p_wye = a*p - b*q
                # q_wye = b*p + a*q
                p_from_p = a
                p_from_q = -b
                q_from_p = b
                q_from_q = a

                # Voltage: R*p_wye + X*q_wye
                M_v_p[:, i] = 1e-3 * (R[:, idx0:idx0+3] @ p_from_p + X[:, idx0:idx0+3] @ q_from_p)
                M_v_q[:, i] = 1e-3 * (R[:, idx0:idx0+3] @ p_from_q + X[:, idx0:idx0+3] @ q_from_q)

                # Branch flows: P + jQ = C_tilde_S s_wye = (C_tilde_S trans_vec) (p + j q)
                branch_pwr = C_tilde_S[:, idx0:idx0+3] @ trans_vec
                M_P_p[:, i] = np.real(branch_pwr)
                M_P_q[:, i] = -np.imag(branch_pwr)
                M_Q_p[:, i] = np.imag(branch_pwr)
                M_Q_q[:, i] = np.real(branch_pwr)

        return {
            "M_v_p": np.real(M_v_p),
            "M_v_q": np.real(M_v_q),
            "M_P_p": np.real(M_P_p),
            "M_P_q": np.real(M_P_q),
            "M_Q_p": np.real(M_Q_p),
            "M_Q_q": np.real(M_Q_q),
        }

    def _compute_battery_bounds(self) -> Dict[str, np.ndarray]:
        """
        Extract per-candidate bounds from utility_battery_info.

        Required keys: S, p_b_min, p_b_max, q_b_min, q_b_max (or variants).
        Optional keys: soc_min, soc_max.
        Defaults: soc in [0, S].
        """
        n_b = self.num_battery
        if n_b == 0:
            return {
                "S": np.zeros(0),
                "p_min": np.zeros(0), "p_max": np.zeros(0),
                "q_min": np.zeros(0), "q_max": np.zeros(0),
                "soc_min": np.zeros(0), "soc_max": np.zeros(0),
            }

        def get_key(d, keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        S = np.array([get_key(info, ["S", "capacity", "energy_capacity"], 0.0) for info in self.utility_battery_info], dtype=float)

        p_min = np.array([get_key(info, ["p_b_min", "p_min", "pmin"], 0.0) for info in self.utility_battery_info], dtype=float)
        p_max = np.array([get_key(info, ["p_b_max", "p_max", "pmax"], 0.0) for info in self.utility_battery_info], dtype=float)
        q_min = np.array([get_key(info, ["q_b_min", "q_min", "qmin"], 0.0) for info in self.utility_battery_info], dtype=float)
        q_max = np.array([get_key(info, ["q_b_max", "q_max", "qmax"], 0.0) for info in self.utility_battery_info], dtype=float)

        soc_min = np.array([get_key(info, ["soc_min", "SoC_min", "SoCmin"], 0.0) for info in self.utility_battery_info], dtype=float)
        soc_max = np.array([get_key(info, ["soc_max", "SoC_max", "SoCmax"], None) for info in self.utility_battery_info], dtype=float)
        # Fill defaults for soc_max when missing
        soc_max = np.where(np.isnan(soc_max), S, soc_max)
        soc_max = np.where(soc_max == 0.0, S, soc_max)

        return {
            "S": S,
            "p_min": p_min,
            "p_max": p_max,
            "q_min": q_min,
            "q_max": q_max,
            "soc_min": soc_min,
            "soc_max": soc_max,
        }

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
    
    def _get_limits_cache_path(self) -> str:
        """Get the file path for cached limits."""
        feeder_name = self.simulator.feeder_name.replace('/', '_')
        filename = f"{feeder_name}_limits.pkl"
        return os.path.join(self.limits_cache_dir, filename)
    
    def _save_limits_to_cache(self) -> None:
        """Save computed limits to cache file."""
        if self.limits is None:
            return
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.limits_cache_dir, exist_ok=True)
        
        cache_path = self._get_limits_cache_path()
        
        # Save limits with metadata
        cache_data = {
            'limits': self.limits,
            'feeder_name': self.simulator.feeder_name,
            'vol_tol': self.vol_tol,
            'lower_voltage_limit_ratio': self.lower_voltage_limit_ratio,
            'relaxation_factor': self.relaxation_factor,
            'v_root_nom': self.v_root_nom,
            'n_buses': self._n_buses
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_limits_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        """Load limits from cache file if it exists and is valid."""
        cache_path = self._get_limits_cache_path()

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate metadata
            if (cache_data.get('feeder_name') != self.simulator.feeder_name or
                cache_data.get('n_buses') != self._n_buses):
                return None
            
            return cache_data.get('limits')
        except Exception:
            return None


# ==============================================================================
# CLARKSON SOLVER
# ==============================================================================

def clarkson_solver(
    problem: ClarksonProblem = None,
    is_scenario_based: bool = True,
    initial_sample_size: int = 25,
    max_iterations: int = 10,
    maximum_subproblem_size: int = 10,
    solver: Optional[Any] = cp.HIGHS,
    verbose: bool = True,
    seed: int = 42,
    solve_all: bool = False,
    max_subproblem_growth: float = 2.0,
) -> Dict[str, Any]:
    """
    Iterative scenario/constraint addition solver — implements step 2 of Algorithm 1.

    This is a Clarkson-style randomized LP algorithm:
      1. Sample a small subset of scenarios (initial_sample_size).
      2. Solve the restricted LP (RSP subproblem, Eq. 10-12).
      3. Find violated scenarios not in the active set.
      4. Double the sampling weight of each violated scenario (importance sampling).
      5. Add violated scenarios to the active set; repeat until no new violations.

    The returned `support_indices` are the support scenarios (Definition III.1)
    identified via non-zero dual variables (constraints [0..9] per scenario; see
    `support_from_duals`). The caller (`run_iterative_pvhc_analysis`) passes
    |support_indices| to `update_num_scenarios` (utils.py) to check the
    φ_{ℓ,n,β}(1-ε) ≥ 0 criterion (Theorem III.2 / Algorithm 1 step 3). That is only
    valid for a converged result: callers must check 'converged'.

    Args:
        is_scenario_based: If True, one subproblem constraint per scenario;
                           if False, individual constraint rows (more granular).
        initial_sample_size: |S_0|, initial active scenario count.
        max_iterations: Cap on Clarkson iterations within one outer call.
        maximum_subproblem_size: Cap on the active set once violated scenarios have to be
            sampled. When the support alone fills it, the active set grows past it (the
            support plus up to half the cap of violated scenarios) instead of stopping,
            up to `max_subproblem_growth` times the cap.
        solver: CVXPY solver, (solver, options) tuple, or a list of them tried in order on
            each subproblem, falling back on any failure (`solver_utils.solve_with_fallback`).
            The default HIGHS cannot handle the fairness/log-barrier term; use
            `solver_utils.FAIRNESS_SOLVERS` for it.
        verbose: Print iteration progress.
        seed: Random seed for reproducibility.
        solve_all: Skip iterative approach; solve full LP with all scenarios.
        max_subproblem_growth: See maximum_subproblem_size.

    Returns:
        dict with keys: 'optimal_capacities', 'total_capacity', 'support_indices',
                        'violated_scenarios', 'converged', 'termination_reason'
                        ('converged', 'max_iterations', 'subproblem_failed' or
                        'subproblem_too_large'), 'dual_var', 'scenario_indices'.
        'dual_var' and 'scenario_indices' come from the same subproblem as the capacities.
    """
    np.random.seed(seed)

    # Total number of elements (scenarios or constraints)
    if is_scenario_based:
        N = problem.num_scenarios
    else:
        N = problem.num_scenarios * problem.num_constraints_per_scenario
    largest_subproblem = int(max_subproblem_growth * maximum_subproblem_size)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Clarkson Iterative {'Scenario' if is_scenario_based else 'Constraint'} Addition")
        print(f"{'='*70}")
        print(f"Total {'scenarios' if is_scenario_based else 'constraints'}: {N}")
        print(f"Initial sample size: {initial_sample_size}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")

    # Initialize
    iteration = 0
    weights = np.ones(N)
    k_current = min(initial_sample_size, N) if not solve_all else N

    # Sample initial set
    active_indices = np.random.choice(
        N, size=k_current, replace=False, p=(weights / np.sum(weights))
    ) if not solve_all else np.arange(N)
    active_indices = set(active_indices.tolist())

    # Main loop
    x_optimal = None
    obj_val = None
    support_indices: Optional[Set[int]] = None
    dual_var, scenario_indices = None, None
    converged = False
    termination_reason = None
    violated_indices: Set[int] = set()

    while iteration <= max_iterations:
        if verbose:
            print(f"\nIteration {iteration}:")
            print(f"  Active {'scenarios' if is_scenario_based else 'constraints'}: {len(active_indices)}")

        active_list = sorted(list(active_indices))
        results = problem.solve_subproblem(active_list, is_scenario_based, solver, verbose=False)
        if len(results) == 3:
            x_new, obj_new, support_new = results
            dual_new, scenarios_new = None, None
        elif len(results) == 5:
            x_new, obj_new, support_new, dual_new, scenarios_new = results
        else:
            raise ValueError("Unexpected number of return values from solve_subproblem")

        if x_new is None:
            termination_reason = 'subproblem_failed'
            print(f"  [clarkson] The subproblem with {len(active_list)} active "
                  f"{'scenarios' if is_scenario_based else 'constraints'} has no solution; "
                  f"stopping without convergence.")
            # Then the sampled program has no solution either: return none, not the previous one
            x_optimal = obj_val = support_indices = dual_var = scenario_indices = None
            break
        x_optimal, obj_val, support_indices = x_new, obj_new, set(support_new)
        dual_var, scenario_indices = dual_new, scenarios_new

        if verbose:
            print(f"  Optimal value: {obj_val:.2f}")
            print(f"  Support size: {len(support_indices)}")

        # Find violations
        violated_indices = problem.find_violated_indices(
            x_optimal, tolerance=1e-6, is_scenario_based=is_scenario_based,
            exclude_indices=active_indices
        )

        # Update weights
        for idx in violated_indices:
            weights[idx] *= 2.0

        if verbose:
            print(f"  Violated: {len(violated_indices)}")

        # Termination check
        if len(violated_indices) == 0:
            converged = True
            termination_reason = 'converged'
            if verbose:
                print(f"\n{'='*70}")
                print(f"Converged! No violations found.")
                print(f"{'='*70}")
            break

        # Update active set
        if len(active_indices) + len(violated_indices) > maximum_subproblem_size:
            num_to_sample = maximum_subproblem_size - len(support_indices)
            if num_to_sample <= 0:
                # The support alone fills the cap. Stopping here would return a solution
                # that violates sampled scenarios, so grow the subproblem instead.
                num_to_sample = min(max(1, maximum_subproblem_size // 2), len(violated_indices),
                                    largest_subproblem - len(support_indices))
                if num_to_sample <= 0:
                    termination_reason = 'subproblem_too_large'
                    print(f"  [clarkson] {len(support_indices)} support scenarios: growing the subproblem "
                          f"past {largest_subproblem} (max_subproblem_growth x maximum_subproblem_size) "
                          f"would be needed; stopping without convergence.")
                    break
                if verbose:
                    print(f"  Support size {len(support_indices)} fills the subproblem cap "
                          f"{maximum_subproblem_size}; growing the subproblem.")

            num_to_sample = min(num_to_sample, len(violated_indices))

            violated_list = sorted(list(violated_indices))
            sampled_violated = np.random.choice(
                violated_list, size=num_to_sample, replace=False,
                p=(weights[violated_list] / np.sum(weights[violated_list]))
            )

            # Alternatively, sample from all the indices not in support indices:
            # non_support_indices = sorted(list(set(range(N)) - support_indices))
            # sampled_non_support = np.random.choice(
            #     non_support_indices, size=num_to_sample,
            #     replace=True,
            #     p=(weights[non_support_indices] / np.sum(weights[non_support_indices]))
            # )

            active_indices = support_indices.union(sampled_violated.tolist())
        else:
            active_indices = active_indices.union(violated_indices)
        iteration += 1

        if solve_all:
            break

    if not converged and termination_reason is None:
        termination_reason = 'max_iterations'
        print(f"  [clarkson] Maximum iterations ({max_iterations}) reached with "
              f"{len(violated_indices)} violated {'scenarios' if is_scenario_based else 'constraints'}; "
              f"not converged.")

    return {
        'optimal_solution': x_optimal,
        'optimal_capacities': x_optimal,  # Alias for compatibility
        'objective_value': np.sum(x_optimal) if x_optimal is not None else 0.0,
        'total_capacity': np.sum(x_optimal) if x_optimal is not None else 0.0,
        'num_iterations': iteration,
        'final_count': len(active_indices),
        'final_scenario_count': len(active_indices),  # Alias
        'support_indices': sorted(list(support_indices)) if support_indices is not None else None,
        'converged': converged,
        'termination_reason': termination_reason,
        'violated_indices': sorted(list(violated_indices)) if not converged else [],
        'violated_scenarios': sorted(list(violated_indices)) if not converged else [],  # Alias
        'dual_var': dual_var,
        'scenario_indices': scenario_indices
    }
