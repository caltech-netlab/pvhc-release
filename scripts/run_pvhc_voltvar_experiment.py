"""scripts/run_pvhc_voltvar_experiment.py

Experiment: PVHC with fixed-schedule Volt-VAR control on utility-owned PV
inverters, compared against the standard robust PVHC baseline
(`PVHCProblemWithoutBattery`, unmodified from `src/analyzer.py`).

This mirrors the structure of `scripts/run_battery_enhancement.py` (day
sampling so that a fixed daily schedule can be reused via `t % period`) and
`scripts/run_pvhc_estimation.py` (utility/customer PV split, result storage),
but drives the new `PVHCProblemWithVoltVar` class defined in
`scripts/analyzer_voltvar.py`.

Nothing under `src/` is modified; the new problem class is a develop-on-top-
of-a-copy of the relevant `src/analyzer.py` logic, and the generic
`ClarksonProblem` interface / `clarkson_solver` are reused unmodified.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import pathlib
import pickle
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import cvxpy as cp

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parent))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, LOCATION
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery, clarkson_solver
from src.utils import PVHCTestConfig, store_optimization_results, load_optimization_results

from analyzer_voltvar import PVHCProblemWithVoltVar

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"


@dataclass
class PVHCVoltVarTestConfig(PVHCTestConfig):
    """Config for PVHC runs with fixed-schedule utility-PV Volt-VAR, mirroring
    `PVHCBatteryTestConfig` (src/utils.py) so results round-trip through the
    repo's established `store_optimization_results`/`load_optimization_results`
    convention instead of a bespoke pickle format."""

    days_sampled: int = 0
    sampling_mode: str = "whole_year"
    equal_samples_per_month: bool = False
    voltvar_period: int = 24
    inverter_rating_factor: float = 1.0

    def resolved_sub_folder(self) -> str:
        if self.sub_folder is not None:
            return self.sub_folder
        base = super().resolved_sub_folder()
        return f"voltvar_experiment/with_pv_voltvar/factor_{self.inverter_rating_factor}/{base}"


# ==============================================================================
# DataLoader init cache
#
# `DataLoader._dssParser()` drives the native OpenDSS engine (via the
# `altdss`/`dss` CFFI bindings) to parse the feeder's .dss files. In this
# environment that step was observed to be occasionally unstable (segfaults /
# native-library crashes triggered inside `_dssParser`, unrelated to anything
# in this repo's Python logic and not reproducible deterministically). Since
# `_dssParser()`'s output (`dss_data`/`node_data`/`circuit_data`) is pure
# Python/numpy state with no live references back into the native engine, we
# cache it to disk after the first successful parse and simply restore it on
# later runs -- entirely bypassing the fragile step once it has succeeded
# once for a given feeder.
# ==============================================================================

def _sanitize_for_pickle(obj: Any, _depth: int = 0):
    """Recursively strip out values that can't be pickled (e.g. live CFFI
    handles nested inside an otherwise-picklable dict), instead of dropping
    the whole containing structure. Dicts/lists/tuples are rebuilt with
    offending leaves replaced by None; only the offending leaves are lost."""
    if _depth > 8:
        try:
            pickle.dumps(obj)
            return obj
        except Exception:
            return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_pickle(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_pickle(v, _depth + 1) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_pickle(v, _depth + 1) for v in obj)
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        return None


def get_data_loader(feeder_name: str, feeder_root: str, use_cache: bool = True) -> DataLoader:
    cache_dir = pathlib.Path(__file__).resolve().parent / ".dss_state_cache"
    cache_path = cache_dir / f"{feeder_name}.pkl"

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            print(f"[DataLoader Cache] Cache at {cache_path} is unreadable ({e}); deleting and reparsing.")
            cache_path.unlink(missing_ok=True)
            return get_data_loader(feeder_name, feeder_root, use_cache=use_cache)
        data_loader = DataLoader.__new__(DataLoader)
        data_loader.__dict__.update(state)
        print(f"[DataLoader Cache] Restored parsed DSS state for {feeder_name} from {cache_path}")
        return data_loader

    data_loader = DataLoader(feeder_name)
    _ = data_loader._dssParser()
    _ = data_loader.get_circuit_model(feeder_root=feeder_root)

    if use_cache:
        # Only the plain Python/numpy state is needed downstream; some
        # attributes set during DSS parsing hold live native (CFFI) handles
        # nested inside otherwise-picklable dicts (e.g. dss_data['swt_control'])
        # -- sanitize recursively so only those specific leaves are dropped,
        # or drop the whole attribute if it's not a dict/list/tuple (e.g. the
        # networkx graph, which isn't needed again once circuit_data is set).
        picklable_state = {}
        for key, value in data_loader.__dict__.items():
            try:
                pickle.dumps(value)
                picklable_state[key] = value
                continue
            except Exception:
                pass
            sanitized = _sanitize_for_pickle(value)
            try:
                pickle.dumps(sanitized)
                picklable_state[key] = sanitized
                print(f"[DataLoader Cache] Sanitized non-picklable nested field(s) in '{key}' for caching.")
            except Exception as e:
                print(f"[DataLoader Cache] Dropping unpicklable attribute '{key}' entirely ({type(e).__name__}: {e}).")

        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(picklable_state, f)
        os.replace(tmp_path, cache_path)  # atomic on POSIX: readers never see a partial file
        print(f"[DataLoader Cache] Saved parsed DSS state for {feeder_name} to {cache_path}")

    return data_loader


# ==============================================================================
# Day sampling helpers (shared with scripts/run_battery_enhancement.py) so a
# fixed daily Volt-VAR schedule can be reused across sampled days via
# `t % voltvar_period`
# ==============================================================================

@dataclass(frozen=True)
class DaySamplingConfig:
    mode: str = "whole_year"
    days_sampled: int = 12
    equal_samples_per_month: bool = True
    seed: int = 42


def _local_to_utc(local_dt: datetime) -> datetime:
    import pytz
    tz = pytz.timezone(LOCATION)
    localized = tz.localize(local_dt, is_dst=None)
    return localized.astimezone(pytz.utc).replace(tzinfo=None)


def _date_range(start_date: datetime, end_date: datetime) -> np.ndarray:
    start_date_np = np.datetime64(start_date).astype("datetime64[D]")
    end_date_np = np.datetime64(end_date).astype("datetime64[D]")
    total_days = int((end_date_np - start_date_np).astype(int)) + 1
    return start_date_np + np.arange(total_days, dtype="timedelta64[D]")


def _sample_days(cfg: DaySamplingConfig) -> List[datetime]:
    np.random.seed(cfg.seed)

    if cfg.mode != "whole_year":
        raise ValueError(f"Only 'whole_year' sampling mode is supported here, got {cfg.mode}")

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    valid_days = _date_range(start_date, end_date)
    if cfg.days_sampled <= 0:
        raise ValueError("days_sampled must be > 0")

    if cfg.equal_samples_per_month:
        if cfg.days_sampled % 12 != 0:
            raise ValueError("days_sampled must be a multiple of 12 when equal_samples_per_month=True")
        samples_per_month = cfg.days_sampled // 12
        days_by_month = {m: [] for m in range(1, 13)}
        for d in valid_days:
            py = d.astype(object)
            days_by_month[py.month].append(d)
        picks = []
        for month in range(1, 13):
            month_days = np.array(days_by_month[month], dtype="datetime64[D]")
            if len(month_days) == 0:
                continue
            k = min(samples_per_month, len(month_days))
            picks.extend(np.random.choice(month_days, size=k, replace=False).tolist())
        picks = np.array(picks, dtype="datetime64[D]")
    else:
        k = min(cfg.days_sampled, len(valid_days))
        picks = np.random.choice(valid_days, size=k, replace=False)

    picks = np.sort(picks)
    return [datetime(d.astype(object).year, d.astype(object).month, d.astype(object).day) for d in picks]


def _fetch_full_range(
    data_loader: DataLoader,
    v_root_nom: float,
    v_root_multiplier: float,
    dt: int = 1800,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], List[dict], np.ndarray]:
    """
    Fetch the entire year's timeseries in a *single* call at 30-minute
    resolution (dt=1800s => 48 steps/day directly, matching the paper's
    scenario definition), then slice out the sampled days locally.
    """
    t0_str = "2023-01-01 00:00:00"
    t1_str = "2023-12-31 23:59:59"
    return data_loader.get_scenario_data_by_range(
        SOLAR_PROFILE_PATH, t0_str, t1_str, v_root_nom, v_root_multiplier, dt=dt,
    )


def _slice_sampled_days(
    time_array: np.ndarray,
    pv_profiles: np.ndarray,
    load_profiles: Tuple[np.ndarray, np.ndarray],
    V0_profiles: np.ndarray,
    day_starts_local: List[datetime],
    period: int = 48,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray, List[datetime]]:
    """Slice the full-range arrays down to just the sampled local days, in
    order, dropping any day that doesn't yield exactly `period` samples
    (e.g. due to the zero-voltage filtering applied in
    `get_scenario_data_by_range`)."""
    p, q = load_profiles
    time_array = time_array.astype("datetime64[s]")

    all_idx = []
    kept_days = []
    for day_start_local in day_starts_local:
        day_end_local = day_start_local + timedelta(days=1)
        utc_start = np.datetime64(_local_to_utc(day_start_local)).astype("datetime64[s]")
        utc_end = np.datetime64(_local_to_utc(day_end_local)).astype("datetime64[s]")
        idx = np.where((time_array >= utc_start) & (time_array < utc_end))[0]
        if len(idx) != period:
            print(f"[Warning] Skipping day {day_start_local.date()}: found {len(idx)} samples, expected {period}")
            continue
        all_idx.append(idx)
        kept_days.append(day_start_local)

    if not all_idx:
        raise RuntimeError("No valid sampled days found in the fetched range")

    idx_concat = np.concatenate(all_idx)
    return (
        time_array[idx_concat],
        pv_profiles[idx_concat],
        (p[idx_concat], q[idx_concat]),
        V0_profiles[idx_concat],
        kept_days,
    )


# ==============================================================================
# PV site construction
# ==============================================================================

def _split_utility_customer(
    load_info: List[dict],
    utility_pv_ratio: float,
    uniform_capacity_kw: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    PV_info = [{**load, "capacity": uniform_capacity_kw} for load in load_info]
    num_total = len(PV_info)
    num_utility = int(np.round(num_total * utility_pv_ratio))

    np.random.seed(seed)
    indices = np.random.permutation(num_total)
    utility_idx = indices[:num_utility]
    customer_idx = indices[num_utility:]

    utility_pv_info = [PV_info[i] for i in sorted(utility_idx.tolist())]
    customer_pv_info = [PV_info[i] for i in sorted(customer_idx.tolist())]
    return utility_pv_info, customer_pv_info


# ==============================================================================
# Baseline and Volt-VAR runs
# ==============================================================================

def run_baseline(
    simulator: Simulator,
    utility_pv_info: List[dict],
    customer_pv_info: List[dict],
    load_info: List[dict],
    pv_profiles: np.ndarray,
    load_profiles: Tuple[np.ndarray, np.ndarray],
    V0_profiles: np.ndarray,
    initial_sample_size: int,
    maximum_subproblem_size: int,
    max_iterations: int,
    seed: int,
) -> dict:
    problem = PVHCProblemWithoutBattery(
        simulator=simulator,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        load_info=load_info,
        is_robust=True,
        with_fairness=False,
        load_cached_limits=True,
    )
    problem.add_scenarios(
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        V0_profiles=V0_profiles,
        filter_zero_pv=False,
    )
    problem.compile()

    results = clarkson_solver(
        problem=problem,
        is_scenario_based=True,
        initial_sample_size=initial_sample_size,
        max_iterations=max_iterations,
        maximum_subproblem_size=maximum_subproblem_size,
        solver=cp.HIGHS,
        verbose=False,
        seed=seed,
        solve_all=False,
    )
    _require_converged(results, "baseline")

    x = results.get("optimal_solution")
    n_u = len(utility_pv_info)
    util_cap = np.array(x[:n_u]) if x is not None else np.zeros(n_u)
    cust_cap = np.array(x[n_u:]) if x is not None else np.zeros(len(customer_pv_info))

    return {
        "converged": results.get("converged"),
        "num_iterations": results.get("num_iterations"),
        "utility_kw": float(np.sum(util_cap)),
        "customer_kw": float(np.sum(cust_cap)),
        "total_kw": float(np.sum(util_cap) + np.sum(cust_cap)),
        "utility_capacities": util_cap,
        "customer_capacities": cust_cap,
    }


def _require_converged(results: dict, what: str) -> None:
    # An unconverged Clarkson solution violates some of the scenarios, so it is not a hosting
    # capacity, and comparing it with the other run would be meaningless.
    if not results.get("converged"):
        raise RuntimeError(
            f"Clarkson did not converge for the {what} problem ({results.get('termination_reason')}); "
            f"increase max_iterations or maximum_subproblem_size and rerun."
        )


def run_voltvar(
    simulator: Simulator,
    utility_pv_info: List[dict],
    customer_pv_info: List[dict],
    load_info: List[dict],
    pv_profiles: np.ndarray,
    load_profiles: Tuple[np.ndarray, np.ndarray],
    V0_profiles: np.ndarray,
    inverter_rating_factor: float,
    voltvar_period: int,
    initial_sample_size: int,
    maximum_subproblem_size: int,
    max_iterations: int,
    seed: int,
) -> dict:
    problem = PVHCProblemWithVoltVar(
        simulator=simulator,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        load_info=load_info,
        is_robust=True,
        with_fairness=False,
        inverter_rating_factor=inverter_rating_factor,
        voltvar_period=voltvar_period,
        load_cached_limits=True,
    )
    problem.add_scenarios(
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        V0_profiles=V0_profiles,
        filter_zero_pv=False,
    )
    problem.compile()

    results = clarkson_solver(
        problem=problem,
        is_scenario_based=True,
        initial_sample_size=initial_sample_size,
        max_iterations=max_iterations,
        maximum_subproblem_size=maximum_subproblem_size,
        solver=cp.HIGHS,
        verbose=False,
        seed=seed,
        solve_all=False,
    )
    _require_converged(results, "Volt-VAR")

    x = results.get("optimal_solution")
    if x is None:
        n_u, n_c = len(utility_pv_info), len(customer_pv_info)
        return {
            "converged": False,
            "num_iterations": results.get("num_iterations"),
            "utility_kw": 0.0, "customer_kw": 0.0, "total_kw": 0.0,
            "utility_capacities": np.zeros(n_u), "customer_capacities": np.zeros(n_c),
            "q_vv": np.zeros((voltvar_period, n_u)),
            "inverter_rating": np.zeros(n_u),
        }

    parts = problem._unpack_solution(x)
    util_cap = parts["zeta_u"]
    cust_cap = parts["zeta_cbar"]
    q_vv = parts["q_vv"]
    # Inverter rating scales with the *solved* zeta_u (decision variable),
    # not a fixed capacity cap -- computed post-hoc from the final solution.
    inverter_rating = inverter_rating_factor * util_cap

    return {
        "converged": results.get("converged"),
        "num_iterations": results.get("num_iterations"),
        "utility_kw": float(np.sum(util_cap)),
        "customer_kw": float(np.sum(cust_cap)),
        "total_kw": float(np.sum(util_cap) + np.sum(cust_cap)),
        "utility_capacities": util_cap,
        "customer_capacities": cust_cap,
        "q_vv": q_vv,
        "inverter_rating": inverter_rating,
    }


def main(
    num_days: int = 12,
    seed: int = 42,
    utility_pv_ratio: float = 0.3,
    uniform_capacity_kw: float = 10.0,
    voltvar_period: int = 24,
    initial_sample_size: int = 20,
    maximum_subproblem_size: int = 40,
    max_iterations: int = 15,
    inverter_rating_factors: Optional[List[float]] = None,
):
    feeder_name = DEFAULT_FEEDER_NAME
    cfg = FEEDERS[feeder_name]
    feeder_root = cfg["feeder_root"]
    v_root_nom = cfg["v_root_nom"]
    v_root_multiplier = cfg["v_root_multiplier"]

    NUM_DAYS = num_days
    SEED = seed
    UTILITY_PV_RATIO = utility_pv_ratio     # mirrors the paper's "mixed" ownership scenario
    UNIFORM_CAPACITY_KW = uniform_capacity_kw  # mirrors the paper's uniform per-site cap
    # NOTE: voltvar_period=24 (hourly) rather than 48 (half-hourly) is a
    # deliberate coarser fixed-schedule choice for this experiment: it halves
    # the number of Volt-VAR schedule decision variables (num_utility_pv *
    # period) relative to matching the data's native 30-min resolution,
    # which matters a lot at full-feeder scale (~175 utility PV sites here).
    # It is still a fully valid "fixed, open-loop schedule" in the sense used
    # throughout this codebase (cf. the battery's `_battery_period`).
    VOLTVAR_PERIOD = voltvar_period
    VOLTVAR_PERIOD_DT_SEC = 24 * 3600 // VOLTVAR_PERIOD

    INITIAL_SAMPLE_SIZE = initial_sample_size
    MAXIMUM_SUBPROBLEM_SIZE = maximum_subproblem_size
    MAX_ITERATIONS = max_iterations
    if inverter_rating_factors is None:
        inverter_rating_factors = [1.0, 1.2]

    print("=" * 80)
    print("PVHC WITH FIXED-SCHEDULE VOLT-VAR CONTROL (utility-owned PV only)")
    print("=" * 80)
    print(f"Feeder: {feeder_name}")

    print("\n1) Initializing DataLoader / compiling feeder...")
    data_loader = get_data_loader(feeder_name, feeder_root)

    print(f"\n2) Sampling {NUM_DAYS} days (whole_year, equal per month, seed={SEED})...")
    equal_samples_per_month = (NUM_DAYS % 12 == 0)
    day_cfg = DaySamplingConfig(
        mode="whole_year", days_sampled=NUM_DAYS,
        equal_samples_per_month=equal_samples_per_month, seed=SEED,
    )
    day_starts_local = _sample_days(day_cfg)
    print(f"   First day: {day_starts_local[0].date()}  Last day: {day_starts_local[-1].date()}")

    print("   Fetching full year's timeseries (single read) at 30-min resolution...")
    full_time, full_pv, full_load, load_info, full_V0 = _fetch_full_range(
        data_loader, v_root_nom, v_root_multiplier, dt=VOLTVAR_PERIOD_DT_SEC,
    )
    time_array, pv_profiles, load_profiles, V0_profiles, kept_days = _slice_sampled_days(
        full_time, full_pv, full_load, full_V0, day_starts_local, period=VOLTVAR_PERIOD,
    )
    T_total = len(time_array)
    assert T_total % VOLTVAR_PERIOD == 0
    print(f"   Loaded {T_total // VOLTVAR_PERIOD} / {NUM_DAYS} requested day(s) => {T_total} scenarios (timesteps)")

    print(f"\n3) Splitting PV sites: {UTILITY_PV_RATIO*100:.0f}% utility-owned, {UNIFORM_CAPACITY_KW} kW per site cap...")
    utility_pv_info, customer_pv_info = _split_utility_customer(
        load_info, UTILITY_PV_RATIO, UNIFORM_CAPACITY_KW, SEED,
    )
    print(f"   Utility PV sites: {len(utility_pv_info)}  |  Customer PV sites: {len(customer_pv_info)}")

    simulator = Simulator(data_loader)

    print("\n4) Running baseline PVHC (no Volt-VAR, PVHCProblemWithoutBattery)...")
    t0 = time.time()
    baseline = run_baseline(
        simulator, utility_pv_info, customer_pv_info, load_info,
        pv_profiles, load_profiles, V0_profiles,
        INITIAL_SAMPLE_SIZE, MAXIMUM_SUBPROBLEM_SIZE, MAX_ITERATIONS, SEED,
    )
    baseline["wall_s"] = time.time() - t0
    print(f"   Converged={baseline['converged']}  Iterations={baseline['num_iterations']}  "
          f"Utility={baseline['utility_kw']:.1f} kW  Customer={baseline['customer_kw']:.1f} kW  "
          f"Total={baseline['total_kw']:.1f} kW  ({baseline['wall_s']:.1f}s)")

    # Store results via the repo's established save/load convention
    # (src/utils.py: store_optimization_results / load_optimization_results)
    # IMMEDIATELY after each stage computes -- not batched at the very end --
    # so that a crash/timeout during a LATER (often much slower) ablation
    # factor can never discard an EARLIER stage's already-computed result.
    node_list = simulator.data_loader.network_data_dict["node_list"]
    edge_list = simulator.data_loader.network_data_dict["edge_list"]

    baseline_config = PVHCTestConfig(
        feeder_name=feeder_name,
        feeder_root=feeder_root,
        v_root_nom=v_root_nom,
        v_root_multiplier=v_root_multiplier,
        seed=SEED,
        num_scenarios=T_total,
        solver=cp.HIGHS,
        initial_sample_size=INITIAL_SAMPLE_SIZE,
        maximum_subproblem_size=MAXIMUM_SUBPROBLEM_SIZE,
        with_fairness=False,
        sub_folder="voltvar_experiment/baseline",
    )
    store_optimization_results(
        feeder_name,
        utility_pv_info,
        customer_pv_info,
        capacities_isa_utility=np.asarray(baseline["utility_capacities"], dtype=float),
        capacities_isa_customer=np.asarray(baseline["customer_capacities"], dtype=float),
        config=baseline_config,
        node_list=node_list,
        edge_list=edge_list,
        load_info=load_info,
        results_summary={
            "converged": bool(baseline["converged"]),
            "num_iterations": int(baseline["num_iterations"]) if baseline["num_iterations"] is not None else -1,
            "total_pv_kw": float(baseline["total_kw"]),
            "wall_s": float(baseline["wall_s"]),
        },
    )
    loaded_baseline = load_optimization_results(feeder_name, sub_folder=baseline_config.resolved_sub_folder())
    assert loaded_baseline["capacities_isa_utility"].shape == np.asarray(baseline["utility_capacities"]).shape
    assert loaded_baseline["capacities_isa_customer"].shape == np.asarray(baseline["customer_capacities"]).shape
    print(f"   Baseline stored + round-trip validated under "
          f"temp/optimization_results/{feeder_name}/{baseline_config.resolved_sub_folder()}")

    voltvar_results = {}
    for inverter_rating_factor in inverter_rating_factors:
        print(f"\n5) Running Volt-VAR PVHC (inverter_rating_factor={inverter_rating_factor})...")
        t0 = time.time()
        res = run_voltvar(
            simulator, utility_pv_info, customer_pv_info, load_info,
            pv_profiles, load_profiles, V0_profiles,
            inverter_rating_factor, VOLTVAR_PERIOD,
            INITIAL_SAMPLE_SIZE, MAXIMUM_SUBPROBLEM_SIZE, MAX_ITERATIONS, SEED,
        )
        res["wall_s"] = time.time() - t0
        voltvar_results[inverter_rating_factor] = res
        print(f"   Converged={res['converged']}  Iterations={res['num_iterations']}  "
              f"Utility={res['utility_kw']:.1f} kW  Customer={res['customer_kw']:.1f} kW  "
              f"Total={res['total_kw']:.1f} kW  ({res['wall_s']:.1f}s)")

        vv_config = PVHCVoltVarTestConfig(
            feeder_name=feeder_name,
            feeder_root=feeder_root,
            v_root_nom=v_root_nom,
            v_root_multiplier=v_root_multiplier,
            seed=SEED,
            num_scenarios=T_total,
            solver=cp.HIGHS,
            initial_sample_size=INITIAL_SAMPLE_SIZE,
            maximum_subproblem_size=MAXIMUM_SUBPROBLEM_SIZE,
            with_fairness=False,
            days_sampled=NUM_DAYS,
            sampling_mode="whole_year",
            equal_samples_per_month=equal_samples_per_month,
            voltvar_period=VOLTVAR_PERIOD,
            inverter_rating_factor=float(inverter_rating_factor),
        )
        store_optimization_results(
            feeder_name,
            utility_pv_info,
            customer_pv_info,
            capacities_isa_utility=np.asarray(res["utility_capacities"], dtype=float),
            capacities_isa_customer=np.asarray(res["customer_capacities"], dtype=float),
            config=vv_config,
            node_list=node_list,
            edge_list=edge_list,
            load_info=load_info,
            results_summary={
                "converged": bool(res["converged"]),
                "num_iterations": int(res["num_iterations"]) if res["num_iterations"] is not None else -1,
                "total_pv_kw": float(res["total_kw"]),
                "wall_s": float(res["wall_s"]),
                "q_vv_schedule": np.asarray(res["q_vv"], dtype=float),
                "voltvar_period": VOLTVAR_PERIOD,
                "inverter_rating_factor": float(inverter_rating_factor),
                "inverter_rating": np.asarray(res["inverter_rating"], dtype=float),
            },
        )
        loaded_vv = load_optimization_results(feeder_name, sub_folder=vv_config.resolved_sub_folder())
        assert loaded_vv["capacities_isa_utility"].shape == np.asarray(res["utility_capacities"]).shape
        assert loaded_vv["capacities_isa_customer"].shape == np.asarray(res["customer_capacities"]).shape
        assert loaded_vv["results_summary"]["q_vv_schedule"].shape == (VOLTVAR_PERIOD, len(utility_pv_info))
        print(f"   Volt-VAR (factor={inverter_rating_factor}) stored + round-trip validated under "
              f"temp/optimization_results/{feeder_name}/{vv_config.resolved_sub_folder()}")

    print("\n" + "=" * 80)
    print("SUMMARY: Total PVHC (kW)")
    print("=" * 80)
    print(f"{'Configuration':<45}{'Utility':>10}{'Customer':>10}{'Total':>10}{'Delta vs base':>14}")
    print(f"{'Baseline (no Volt-VAR)':<45}{baseline['utility_kw']:>10.1f}{baseline['customer_kw']:>10.1f}"
          f"{baseline['total_kw']:>10.1f}{'--':>12}")
    for factor, res in voltvar_results.items():
        delta = res["total_kw"] - baseline["total_kw"]
        pct = 100.0 * delta / baseline["total_kw"] if baseline["total_kw"] > 0 else float("nan")
        label = f"Volt-VAR (inverter factor={factor})"
        print(f"{label:<45}{res['utility_kw']:>10.1f}{res['customer_kw']:>10.1f}"
              f"{res['total_kw']:>10.1f}{f'{delta:+.1f} ({pct:+.1f}%)':>12}")

    return {"baseline": baseline, "voltvar_results": voltvar_results}


if __name__ == "__main__":
    main()
