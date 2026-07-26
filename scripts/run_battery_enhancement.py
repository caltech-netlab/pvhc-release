"""
scripts/run_battery_enhancement.py — Paper Appendix C, Table II (battery rows)

Runs PVHC optimization with open-loop utility battery storage, implementing the
battery extension described in Appendix C (Eq. 18a-18j).

Battery model (Appendix C):
  - Open-loop schedule: battery power injection p_b(t) is an optimization
    variable decided before observing the scenario (robust open-loop control).
  - SoC dynamics:  SoC(t+1) = SoC(t) - p_b(t)*dt / E_batt          (Eq. 18c)
  - SoC bounds:    SoC_min <= SoC(t) <= SoC_max                    (Eq. 18d)
  - Power bounds:  -P_batt <= p_b(t) <= P_batt                     (Eq. 18e)
  - Budget:        sum_i E_batt_i <= B_total (total energy budget) (Eq. 18f)
  - Continuity:    SoC(0) = SoC(T) (daily periodicity)             (Eq. 18g)

Scenario structure: samples full days (48 half-hour timesteps/day) rather than
independent hourly snapshots, to allow meaningful SoC trajectory optimization.

Key components:
  - `DaySamplingConfig`: configures day-level sampling (mode, days_sampled)
  - `utility_battery_info`: battery candidate sites drawn from load buses
  - `PVHCProblemWithBattery` from src/analyzer.py replaces `PVHCProblemWithoutBattery`
  - `zeta_b_binary=False`: continuous battery sizing (fractional installation)
  - Budget sweep: runs a range of total energy budgets to generate the PVHC
    vs. budget curve (Table II battery rows)

Outputs:
  temp/optimization_results/<feeder>/with_battery/without_fairness/
  temp/optimization_results/<feeder>/with_battery/with_fairness/
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import pathlib
import sys
from typing import List, Optional, Tuple

import numpy as np
import cvxpy as cp
import pytz

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, LOCATION, DEFAULT_CANDIDATE_PV_CAPACITY_KW
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithBattery, clarkson_solver
from src.utils import store_optimization_results, load_optimization_results, PVHCBatteryTestConfig, load_capacity_limits

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"


@dataclass(frozen=True)
class DaySamplingConfig:
    mode: str = "whole_year"  # whole_year|spring|summer|fall|winter
    days_sampled: int = 12
    equal_samples_per_month: bool = True
    seed: int = 42


def _local_to_utc(local_dt: datetime) -> datetime:
    """Convert the feeder's local time (constants.LOCATION) to UTC, accounting for DST."""
    tz = pytz.timezone(LOCATION)
    localized = tz.localize(local_dt, is_dst=None)
    return localized.astimezone(pytz.utc).replace(tzinfo=None)


def _date_range(start_date: datetime, end_date: datetime) -> np.ndarray:
    """Return a datetime64[D] array of all days in [start_date, end_date]."""
    start_date_np = np.datetime64(start_date).astype("datetime64[D]")
    end_date_np = np.datetime64(end_date).astype("datetime64[D]")
    total_days = int((end_date_np - start_date_np).astype(int)) + 1
    return start_date_np + np.arange(total_days, dtype="timedelta64[D]")


def _sample_days(cfg: DaySamplingConfig) -> List[datetime]:
    """Sample day starts (feeder-local time) based on cfg."""
    season_ranges = {
        "spring": [("2023-03-20", "2023-06-20")],
        "summer": [("2023-06-21", "2023-09-22")],
        "fall": [("2023-09-23", "2023-12-20")],
        "winter": [("2023-01-01", "2023-03-19"), ("2023-12-21", "2023-12-31")],
    }

    np.random.seed(cfg.seed)

    def _pick_from_range(start: datetime, end: datetime, k: int) -> List[datetime]:
        days = _date_range(start, end)
        picks = days if k >= len(days) else np.random.choice(days, size=k, replace=False)
        return [datetime(d.astype(object).year, d.astype(object).month, d.astype(object).day) for d in picks]

    if cfg.mode == "whole_year":
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

    if cfg.mode not in season_ranges:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    all_days: List[datetime] = []
    per_range = max(1, cfg.days_sampled // len(season_ranges[cfg.mode]))
    for start_str, end_str in season_ranges[cfg.mode]:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        all_days.extend(_pick_from_range(start, end, per_range))
    return sorted(all_days)[: cfg.days_sampled]


def _downsample_to_48(
    time_array: np.ndarray,
    pv_profiles: np.ndarray,
    load_profiles: Tuple[np.ndarray, np.ndarray],
    V0_profiles: np.ndarray,
    target_T: int = 48,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Downsample a day's timeseries to exactly 48 steps if needed."""
    T = len(time_array)
    if T == target_T:
        return time_array, pv_profiles, load_profiles, V0_profiles
    if T < target_T:
        raise ValueError(f"Day has only {T} samples; expected at least {target_T}.")

    if T % target_T == 0:
        stride = T // target_T
        idx = np.arange(0, T, stride)[:target_T]
    else:
        idx = np.linspace(0, T - 1, num=target_T).round().astype(int)

    p, q = load_profiles
    return time_array[idx], pv_profiles[idx], (p[idx], q[idx]), V0_profiles[idx]


def _stack_days(
    data_loader: DataLoader,
    v_root_nom: float,
    v_root_multiplier: float,
    day_starts_local: List[datetime],
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], List[dict], np.ndarray]:
    """Load each sampled day using get_scenario_data_by_range and stack into one long horizon."""
    all_time: List[np.ndarray] = []
    all_pv: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_q: List[np.ndarray] = []
    all_v0: List[np.ndarray] = []
    load_info_ref: Optional[List[dict]] = None

    for day_start_local in day_starts_local:
        day_end_local = day_start_local + timedelta(days=1)
        utc_start = _local_to_utc(day_start_local)
        utc_end = _local_to_utc(day_end_local)
        t0 = utc_start.strftime("%Y-%m-%d %H:%M:%S")
        t1 = utc_end.strftime("%Y-%m-%d %H:%M:%S")

        time_array, pv_profiles, load_profiles, load_info, V0_profiles = data_loader.get_scenario_data_by_range(
            SOLAR_PROFILE_PATH, t0, t1, v_root_nom, v_root_multiplier,
        )

        try:
            time_array, pv_profiles, load_profiles, V0_profiles = _downsample_to_48(
                time_array=time_array,
                pv_profiles=pv_profiles,
                load_profiles=load_profiles,
                V0_profiles=V0_profiles,
                target_T=48,
            )
        except Exception as e:
            print(f"[Warning] Skipping day {day_start_local.date()} due to sampling mismatch: {e}")
            continue

        if load_info_ref is None:
            load_info_ref = load_info
        elif len(load_info) != len(load_info_ref):
            raise RuntimeError("load_info changed across days; cannot stack safely")

        p, q = load_profiles
        all_time.append(time_array)
        all_pv.append(pv_profiles)
        all_p.append(p)
        all_q.append(q)
        all_v0.append(V0_profiles)

    if load_info_ref is None or len(all_time) == 0:
        raise RuntimeError("No valid days loaded; cannot run optimization")

    time_array = np.concatenate(all_time)
    pv_profiles = np.concatenate(all_pv)
    load_p = np.vstack(all_p)
    load_q = np.vstack(all_q)
    V0_profiles = np.vstack(all_v0)
    return time_array, pv_profiles, (load_p, load_q), load_info_ref, V0_profiles


def _build_utility_battery_info(
    data_loader: DataLoader,
    load_info: List[dict],
    num_candidates: int,
    seed: int,
    power_kw: float,
    energy_kwh: float,
) -> List[dict]:
    """Create battery candidates based on load buses/phases, with realistic bounds."""
    if num_candidates <= 0:
        return []

    rng = np.random.default_rng(seed)
    idx = np.arange(len(load_info))
    if num_candidates < len(load_info):
        idx = rng.choice(idx, size=num_candidates, replace=False)

    batt_info: List[dict] = []
    for i in sorted(idx.tolist()):
        li = dict(load_info[i])
        bus_lc = li["bus"].lower()
        li["bus"] = bus_lc
        try:
            li["connected_transformer"] = data_loader.node_data[bus_lc]["connected_transformer"]
        except Exception:
            li["connected_transformer"] = {}

        batt_info.append({
            "bus": li["bus"],
            "phases": li["phases"],
            "connected_transformer": li.get("connected_transformer", {}),
            # Bounds: p in kW, soc in kWh, q in kvar-ish (same magnitude)
            "S": float(energy_kwh),
            "p_b_min": float(-power_kw),
            "p_b_max": float(power_kw),
            "q_b_min": float(0),
            "q_b_max": float(0),
            "soc_min": 0.2 * float(energy_kwh),
            "soc_max": float(energy_kwh),
        })

    return batt_info


def run_pvhc_with_battery(
    feeder_name: str = DEFAULT_FEEDER_NAME,
    num_days: int = 12,
    sampling_mode: str = "whole_year",
    equal_samples_per_month: bool = True,
    seed: int = 42,
    initial_sample_size: int = 6,
    maximum_subproblem_size: int = 6,
    max_iterations: int = 10,
    num_battery_candidates: int = 10,
    battery_power_kw: float = 100.0,
    battery_energy_kwh: float = 200.0,
    battery_cost: float = 0.0,
    with_fairness: bool = False,
    total_capacity_limit: Optional[float] = None,
    sub_folder=None
) -> dict:
    cfg = FEEDERS[feeder_name]

    config = PVHCBatteryTestConfig(
        feeder_name=feeder_name,
        feeder_root=cfg["feeder_root"],
        v_root_nom=cfg["v_root_nom"],
        v_root_multiplier=cfg["v_root_multiplier"],
        seed=seed,
        sub_folder=sub_folder,
        num_scenarios=0,  # updated after compile()
        solver=cp.HIGHS,
        initial_sample_size=initial_sample_size,
        maximum_subproblem_size=maximum_subproblem_size,
        with_fairness=with_fairness,
        days_sampled=num_days,
        sampling_mode=sampling_mode,
        equal_samples_per_month=equal_samples_per_month,
        num_battery_candidates=num_battery_candidates,
        battery_power_kw=battery_power_kw,
        battery_energy_kwh=battery_energy_kwh,
        battery_cost=battery_cost,
        zeta_b_binary=False,
    )

    print("=" * 80)
    print("PVHC WITH BATTERY")
    print("=" * 80)
    print(f"Feeder: {feeder_name}")
    print(f"Battery timestep: 0.5h (48 steps/day)")

    print("\n1) Initializing DataLoader / compiling feeder...")
    data_loader = DataLoader(feeder_name)
    _ = data_loader._dssParser()
    _ = data_loader.get_circuit_model(feeder_root=cfg["feeder_root"])

    print("\n2) Sampling full days and loading day profiles...")
    day_cfg = DaySamplingConfig(
        mode=sampling_mode,
        days_sampled=num_days,
        equal_samples_per_month=equal_samples_per_month,
        seed=seed,
    )
    day_starts_local = _sample_days(day_cfg)
    print(f"Requested days: {len(day_starts_local)}")
    if len(day_starts_local) > 0:
        print(f"First sampled day (local): {day_starts_local[0].date()}")
        print(f"Last sampled day (local):  {day_starts_local[-1].date()}")

    time_array, pv_profiles, load_profiles, load_info, V0_profiles = _stack_days(
        data_loader=data_loader,
        v_root_nom=cfg["v_root_nom"],
        v_root_multiplier=cfg["v_root_multiplier"],
        day_starts_local=day_starts_local,
    )
    T_total = len(time_array)
    assert T_total % 48 == 0, "Expected concatenated horizon to be multiple of 48"
    num_days_loaded = T_total // 48
    print(f"Loaded {num_days_loaded} day(s) => {T_total} timesteps")
    print(f"PV profile shape: {pv_profiles.shape}")
    print(f"Load P shape: {load_profiles[0].shape}, Load Q shape: {load_profiles[1].shape}")
    print(f"V0 shape: {V0_profiles.shape}")

    print("\n3) Building PV and battery candidate sets...")
    load_info_with_limits = load_capacity_limits(feeder_name)
    if load_info_with_limits is not None:
        print("   Using extracted per-customer capacity limits (Eq. 12).")
        PV_info = load_info_with_limits
    else:
        print("   No extracted capacity limits cached; falling back to uniform "
              f"{DEFAULT_CANDIDATE_PV_CAPACITY_KW} kW default (run "
              "calculate_capacity_limit_and_store.py for realistic limits).")
        PV_info = [{**load, "capacity": DEFAULT_CANDIDATE_PV_CAPACITY_KW} for load in load_info]
    utility_pv_info = [{**load, "capacity": 0.0} for load in PV_info][:1]
    customer_pv_info = PV_info
    utility_battery_info = _build_utility_battery_info(
        data_loader=data_loader,
        load_info=PV_info,
        num_candidates=num_battery_candidates,
        seed=seed,
        power_kw=battery_power_kw,
        energy_kwh=battery_energy_kwh,
    )

    print(f"Utility PV candidates: {len(utility_pv_info)}")
    print(f"Customer PV candidates: {len(customer_pv_info)}")
    print(f"Battery candidates: {len(utility_battery_info)}")
    if len(utility_battery_info) > 0:
        print(
            f"Battery bounds example: P in [{utility_battery_info[0]['p_b_min']}, {utility_battery_info[0]['p_b_max']}] kW, "
            f"SoC in [0, {utility_battery_info[0]['S']}] kWh"
        )

    print("\n4) Creating PVHCProblemWithBattery (zeta_b_binary=False) and compiling...")
    simulator = Simulator(data_loader)
    problem = PVHCProblemWithBattery(
        simulator=simulator,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        utility_battery_info=utility_battery_info,
        load_info=load_info,
        is_robust=True,
        with_fairness=with_fairness,
        fairness_factor=100.0,
        battery_cost=battery_cost,
        zeta_b_binary=False,
        delta_t_hours=0.5,
        total_capacity_limit=len(utility_battery_info) if total_capacity_limit is None else total_capacity_limit,
        load_cached_limits=True,
    )
    problem.add_scenarios(
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        V0_profiles=V0_profiles,
        filter_zero_pv=False,
    )
    problem.compile()
    config.num_scenarios = int(problem.num_scenarios)
    print(f"Compiled. Scenarios (timesteps) after filtering: {problem.num_scenarios}")
    print(f"Decision variables (packed): {problem.num_variables}")

    print("\n5) Solving with Clarkson iterative addition...")
    solve_start = datetime.now()
    results = clarkson_solver(
        problem=problem,
        is_scenario_based=False,
        # `initial_sample_size`/`maximum_subproblem_size` here are multipliers of
        # problem.num_variables, not raw counts (see this function's own
        # parameter defaults, both 6). With the deterministic (is_scenario_based=
        # False) day-sampling formulation, N = num_scenarios * constraints_per_
        # scenario is huge (~20M for a 12-day/1195-bus run), so a too-small
        # subproblem cap forces every iteration to evict previously-satisfied,
        # non-support constraints to make room for newly-sampled violated ones --
        # they then get silently re-violated and re-discovered later, so
        # `find_violated_indices`'s count oscillates indefinitely instead of
        # shrinking to 0 (verified: a 2x cap never converged in 30 iterations at
        # this scale; honoring the caller's multiplier converges cleanly).
        initial_sample_size=int(initial_sample_size * problem.num_variables),
        max_iterations=max_iterations,
        maximum_subproblem_size=int(maximum_subproblem_size * problem.num_variables),
        solver=cp.HIGHS,
        verbose=True,
        seed=seed,
        solve_all=False,
    )
    solve_end = datetime.now()
    wall_s = (solve_end - solve_start).total_seconds()

    x = results.get("optimal_solution")
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS (WITH BATTERY)")
    print("=" * 80)
    print(f"Converged: {results.get('converged')}")
    print(f"Iterations: {results.get('num_iterations')}")
    print(f"Final active count: {results.get('final_count')}")
    print(f"Support count: {len(results.get('support_indices', []))}")
    print(f"Wall time: {wall_s:.2f} s")

    if x is None:
        print("No solution returned (infeasible or solver failure).")
        return results

    unpack = problem._unpack_solution(x)
    z_u, z_c, z_b = unpack["zeta_u"], unpack["zeta_cbar"], unpack["zeta_b"]
    p_b, q_b, soc = unpack["p_b"], unpack["q_b"], unpack["soc"]

    pv_total_kw = float(np.sum(z_u) + np.sum(z_c))
    batt_install_total = float(np.sum(z_b))
    print(f"Total PV hosting capacity (utility + customer): {pv_total_kw:.2f} kW")
    print(f"Battery install sum(zeta_b) (continuous 0..1): {batt_install_total:.3f}")

    if len(z_b) > 0:
        print(f"Battery dispatch |p_b| max: {np.max(np.abs(p_b)):.2f} kW")
        print(f"Battery dispatch |q_b| max: {np.max(np.abs(q_b)):.2f}")
        print(f"Battery SoC min/max: {np.min(soc):.2f} / {np.max(soc):.2f} kWh")
        top = np.argsort(-z_b)[: min(5, len(z_b))]
        print("Top battery sites by zeta_b:")
        for rank, j in enumerate(top.tolist(), start=1):
            bi = utility_battery_info[j]
            print(f"  {rank}. bus={bi['bus']}, phases={bi['phases']}, zeta_b={z_b[j]:.3f}")

    try:
        node_list = simulator.data_loader.network_data_dict["node_list"]
        edge_list = simulator.data_loader.network_data_dict["edge_list"]
        store_optimization_results(
            feeder_name,
            utility_pv_info,
            customer_pv_info,
            capacities_isa_utility=np.array(z_u, dtype=float),
            capacities_isa_customer=np.array(z_c, dtype=float),
            config=config,
            node_list=node_list,
            edge_list=edge_list,
            utility_battery_info=utility_battery_info,
            zeta_b=np.array(z_b, dtype=float),
            p_b=np.array(p_b, dtype=float),
            q_b=np.array(q_b, dtype=float),
            soc=np.array(soc, dtype=float),
            results_summary={
                "converged": bool(results.get("converged")),
                "num_iterations": int(results.get("num_iterations", -1)),
                "final_count": int(results.get("final_count", -1)),
                "objective_value": float(results.get("objective_value", np.nan)),
                "total_pv_kw": float(pv_total_kw),
                "sum_zeta_b": float(batt_install_total),
            },
        )
        print(f"Stored optimization results under temp/optimization_results/{feeder_name}/{config.resolved_sub_folder()}")

        loaded = load_optimization_results(feeder_name, sub_folder=config.resolved_sub_folder())
        assert "capacities_isa_utility" in loaded and "capacities_isa_customer" in loaded
        assert loaded["capacities_isa_utility"].shape == np.array(z_u, dtype=float).shape
        assert loaded["capacities_isa_customer"].shape == np.array(z_c, dtype=float).shape
        for k, ref in [("zeta_b", z_b), ("p_b", p_b), ("q_b", q_b), ("soc", soc)]:
            if len(utility_battery_info) > 0:
                assert k in loaded, f"Missing {k} in loaded results"
                assert loaded[k].shape == np.array(ref, dtype=float).shape
        print("Load-back validation passed (PV + battery arrays).")
    except Exception as e:
        print(f"[Warning] Failed to store optimization results: {e}")

    return results


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME

    # Battery parameters
    BATTERY_POWER_KW = 5       # Battery power rating in kW
    BATTERY_ENERGY_KWH = 30    # Battery energy capacity in kWh
    BATTERY_COST = 0.00001     # Battery cost coefficient
    NUM_BATTERY_CANDIDATES = 60
    WITH_FAIRNESS = False

    # Sampling parameters
    NUM_DAYS = 12
    SAMPLING_MODE = "whole_year"  # Options: whole_year, spring, summer, fall, winter
    EQUAL_PER_MONTH = True

    # Ablation study: total_capacity_limit controls the maximum sum of zeta_b
    # (battery installation fractions). None means no limit.
    ablation_configs = [
        # (total_capacity_limit, sub_folder)
        (10, f"battery_{NUM_DAYS}days_limit10"),
        (20, f"battery_{NUM_DAYS}days_limit20"),
    ]

    common_params = {
        "feeder_name": feeder_name,
        "num_days": NUM_DAYS,
        "sampling_mode": SAMPLING_MODE,
        "equal_samples_per_month": EQUAL_PER_MONTH,
        "seed": 42,  # Fixed seed ensures scenario consistency
        "initial_sample_size": 10,
        "maximum_subproblem_size": 10,
        "max_iterations": 30,
        "num_battery_candidates": NUM_BATTERY_CANDIDATES,
        "battery_power_kw": BATTERY_POWER_KW,
        "battery_energy_kwh": BATTERY_ENERGY_KWH,
        "battery_cost": BATTERY_COST,
        "with_fairness": WITH_FAIRNESS,
    }

    print("\n" + "="*80)
    print(f"ABLATION STUDY: Running {len(ablation_configs)} configurations")
    print(f"Feeder: {feeder_name}")
    print(f"Seed: {common_params['seed']} (ensures same scenarios across all runs)")
    print(f"Sampling: {NUM_DAYS} days, mode={SAMPLING_MODE}")
    print(f"Battery: {BATTERY_POWER_KW}kW / {BATTERY_ENERGY_KWH}kWh")
    print(f"Battery Candidates: {NUM_BATTERY_CANDIDATES} (fixed)")
    print(f"Fairness: {WITH_FAIRNESS} (fixed)")
    print("Varying: total_capacity_limit")
    print("="*80)

    all_results = []
    for idx, (total_cap_limit, subfolder) in enumerate(ablation_configs):
        print("\n" + "#"*80)
        print(f"# CONFIGURATION {idx+1}/{len(ablation_configs)}")
        print(f"# Total Capacity Limit: {total_cap_limit if total_cap_limit is not None else 'None (unlimited)'}")
        print(f"# Sub-folder: {subfolder}")
        print("#"*80 + "\n")

        results = run_pvhc_with_battery(
            **common_params,
            total_capacity_limit=total_cap_limit,
            sub_folder=subfolder
        )

        all_results.append({
            "config_idx": idx + 1,
            "total_capacity_limit": total_cap_limit,
            "subfolder": subfolder,
            "converged": results.get("converged", False),
            "num_iterations": results.get("num_iterations", -1),
            "objective_value": results.get("objective_value", np.nan)
        })

        loaded_results = load_optimization_results(feeder_name, sub_folder=subfolder)
        if loaded_results is None:
            print(f"\n[Config {idx+1}] Validation failed: could not load results")
        else:
            util_total = float(loaded_results["capacities_isa_utility"].sum())
            cust_total = float(loaded_results["capacities_isa_customer"].sum())
            if np.isnan(util_total) or np.isnan(cust_total):
                print(f"\n[Config {idx+1}] Validation failed: NaN detected in PV capacities")
            else:
                total_pv = util_total + cust_total
                print(f"\n[Config {idx+1}] Validation passed")
                print(f"   Total PV capacity: {total_pv:.2f} kW")
                if "zeta_b" in loaded_results:
                    sum_zeta_b = float(loaded_results["zeta_b"].sum())
                    print(f"   Sum of zeta_b: {sum_zeta_b:.3f}")

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE - SUMMARY")
    print("="*80)
    print(f"Results stored in: temp/optimization_results/{feeder_name}/")
    print(f"\nFixed parameters: Candidates={NUM_BATTERY_CANDIDATES}, Fairness={WITH_FAIRNESS}")
    print("\nConfigurations:")
    for res in all_results:
        status = "OK" if res["converged"] else "FAILED"
        limit_str = f"{res['total_capacity_limit']}" if res["total_capacity_limit"] is not None else "None"
        print(f"  [{status}] Config {res['config_idx']}: {res['subfolder']}")
        print(f"      Capacity Limit={limit_str}, "
              f"Iterations={res['num_iterations']}, Obj={res['objective_value']:.2f}")
    print("="*80)
