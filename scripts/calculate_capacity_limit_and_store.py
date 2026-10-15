# ===============================================
# Sets the per-site physical PV ceilings zeta_max -- the box constraints
# (12c)-(12d) that bound how much PV a site could host at all, before the
# network is considered.
#
# This is an *extension*, not the paper's own convention: the paper's case
# study (Section IV.A) sets zeta_max uniformly at 10 kW for every site, a
# deliberate simplification. The formulation admits any per-site vector, so
# here we derive a site-varying ceiling from each customer's own estimated
# consumption instead. See "Per-customer capacity ceilings" in README.md.
#
# Procedure:
# 1. Sample some scenarios.
# 2. Calculate the PV generation in kWh (per kW of installed capacity).
# 3. Calculate the consumption for each load bus in kWh.
# 4. Then the base capacity limit is calculated as:
#    capacity_limit = consumption / PV_generation
#    i.e. the nameplate that generates about as much energy as the site
#    consumes ("net-zero" sizing), then scaled by capacity_multiplier.
# 5. Finally, store the calculated capacity limits for future use.
#
# To use the paper's uniform 10 kW ceiling instead, skip this script and set
# USE_EXTRACTED_CAPACITY_LIMITS = False in scripts/run_pvhc_estimation.py.
# ===============================================

import numpy as np
import sys
import pathlib

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.data_loader import DataLoader
from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, DEFAULT_CANDIDATE_PV_CAPACITY_KW
from src.utils import save_capacity_limits, load_capacity_limits

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"


def calculate_and_store_capacity_limits(
    feeder_name: str = DEFAULT_FEEDER_NAME,
    num_scenarios: int = 500,
    delta_t_hours: float = 0.5,
    capacity_multiplier: float = 3.0
):
    """
    Calculate capacity limits for a feeder and store them.

    Args:
        feeder_name: Name of the feeder to analyze (see src.constants.FEEDERS)
        num_scenarios: Number of scenarios to sample
        delta_t_hours: Time step duration in hours (for kWh calculation)
        capacity_multiplier: Scales the raw consumption/generation quotient,
            which by construction sizes each site's candidate PV to roughly net-zero
            annual energy (nameplate producing ~as much as the site consumes). A
            multiplier > 1 models a policy that permits net PV export beyond
            net-metering parity (real NEM/interconnection programs vary in how much
            oversizing they allow). On this example feeder/subtree at 3x, the
            network (its upper voltage limit) holds all but 5 of 553 customers
            below their ceiling.
    """
    cfg = FEEDERS[feeder_name]

    print("="*80)
    print(f"CALCULATING CAPACITY LIMITS FOR {feeder_name}")
    print("="*80)

    print(f"\n1. Initializing DataLoader for {feeder_name}...")
    data_loader = DataLoader(feeder_name)

    print("2. Compiling OpenDSS files...")
    network_data = data_loader._dssParser()
    assert network_data, "Network data should not be empty"

    print("3. Computing LinDistFlow model parameters...")
    data_loader.get_circuit_model(feeder_root=cfg["feeder_root"])

    print(f"\n4. Sampling {num_scenarios} scenarios...")
    time_array, pv_profiles, load_profiles, load_info, V0_samples = \
        data_loader.get_scenario_data_by_sampling(
            SOLAR_PROFILE_PATH,
            N=num_scenarios,
            v_root_nom=cfg["v_root_nom"],
            v_root_multiplier=cfg["v_root_multiplier"],
            seed=42
        )

    print(f"   - Sampled scenarios: {len(time_array)}")
    print(f"   - PV profiles shape: {pv_profiles.shape}")
    print(f"   - Load P profiles shape: {load_profiles[0].shape}")
    print(f"   - Load Q profiles shape: {load_profiles[1].shape}")
    print(f"   - Number of loads: {len(load_info)}")

    # 3. Calculate total PV generation across all scenarios
    print("\n5. Calculating total PV generation...")
    if pv_profiles.ndim == 1:
        pv_gen_per_scenario = pv_profiles  # Already in kW
    else:
        pv_gen_per_scenario = np.sum(pv_profiles, axis=1)  # Sum across all PV sites

    total_pv_gen_kwh = np.sum(pv_gen_per_scenario) * delta_t_hours

    print(f"   - Total PV generation: {total_pv_gen_kwh:.2f} kWh")
    print(f"   - Average PV per scenario: {np.mean(pv_gen_per_scenario):.2f} kW")
    print(f"   - Min/Max PV per scenario: {np.min(pv_gen_per_scenario):.2f} / {np.max(pv_gen_per_scenario):.2f} kW")

    # 4. Calculate total consumption per load bus directly from load_profiles
    print("\n6. Calculating consumption per load bus from load profiles...")
    load_P = load_profiles[0]  # Active power in kW, shape (num_scenarios, num_loads)
    consumption_per_load = np.sum(np.abs(load_P), axis=0) * delta_t_hours  # kWh per load

    print(f"   - Load profiles contain {load_P.shape[1]} loads across {load_P.shape[0]} scenarios")
    print(f"   - Total consumption across all loads: {np.sum(consumption_per_load):.2f} kWh")
    print(f"   - Average consumption per load: {np.mean(consumption_per_load):.2f} kWh")
    print(f"   - Min/Max consumption per load: {np.min(consumption_per_load):.2f} / {np.max(consumption_per_load):.2f} kWh")

    # 5. Calculate capacity limits: consumption / PV_generation
    print("\n7. Calculating capacity limits...")
    if total_pv_gen_kwh < 1e-6:
        print("[Warning] Total PV generation is too small, using default limits")
        capacity_limits_per_load = np.ones(len(load_info)) * DEFAULT_CANDIDATE_PV_CAPACITY_KW
    else:
        # capacity_limit[i] = consumption[i] / total_pv_generation
        # This represents each bus's share when total capacity = total consumption
        capacity_limits_per_load = consumption_per_load / total_pv_gen_kwh

    capacity_limits_per_load = capacity_limits_per_load * capacity_multiplier
    print(f"   - Applied capacity_multiplier={capacity_multiplier} (net-zero x{capacity_multiplier})")
    print(f"   - Capacity limits calculated for {len(capacity_limits_per_load)} loads")
    print(f"   - Average capacity limit: {np.mean(capacity_limits_per_load):.3f} kW")
    print(f"   - Min/Max capacity limit: {np.min(capacity_limits_per_load):.3f} / {np.max(capacity_limits_per_load):.3f} kW")

    # 6. Create enhanced load_info with capacity field
    print("\n8. Creating load_info with capacity limits...")
    load_info_with_limits = []
    zero_consumption_loads = []

    for load_idx, load in enumerate(load_info):
        limit_value = capacity_limits_per_load[load_idx]
        load_with_limit = {**load, "capacity": limit_value}
        load_info_with_limits.append(load_with_limit)
        if limit_value < 1e-6:
            zero_consumption_loads.append(f"{load['bus']} (phases: {load['phases']})")

    if zero_consumption_loads:
        print(f"\n   [Info] Found {len(zero_consumption_loads)} loads with zero consumption "
              f"({len(zero_consumption_loads)/len(load_info)*100:.1f}%)")
        print(f"   - These loads had no consumption during the sampled scenarios")
        if len(zero_consumption_loads) <= 10:
            print(f"   - Zero-consumption loads: {zero_consumption_loads}")
        else:
            print(f"   - First 10 zero-consumption loads: {zero_consumption_loads[:10]}")

    # 7. Store capacity limits
    print("\n9. Storing capacity limits...")
    save_capacity_limits(feeder_name, load_info_with_limits)

    print(f"\n{'='*80}")
    print("CAPACITY LIMITS CALCULATION COMPLETE")
    print(f"{'='*80}")

    return load_info_with_limits


if __name__ == "__main__":
    capacity_limits = calculate_and_store_capacity_limits(
        feeder_name=DEFAULT_FEEDER_NAME,
        num_scenarios=10000,
        delta_t_hours=0.5
    )

    print("\n" + "="*80)
    print("VERIFICATION: Loading stored capacity limits...")
    print("="*80)
    loaded_limits = load_capacity_limits(DEFAULT_FEEDER_NAME)

    if loaded_limits is not None:
        print(f"Successfully loaded {len(loaded_limits)} load capacity limits")
        print(f"Sample limits (first 5):")
        for i, load_limit in enumerate(loaded_limits[:5]):
            print(f"  {i+1}. Bus: {load_limit['bus']}, Phases: {load_limit['phases']}, Capacity: {load_limit['capacity']:.3f} kW")
    else:
        print("[Error] Failed to load capacity limits")
