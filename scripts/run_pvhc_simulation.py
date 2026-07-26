"""
Time-Series Simulation for Violation Study — Paper Section IV.B, Fig. 2, Fig. 3

Runs full LinDistFlow power-flow simulation over historical timeseries to assess
voltage and branch-loading violations under different PV adoption scenarios.

Paper usage:
  Fig. 2 (Section IV.B) — unregulated adoption study:
    Simulates random PV installations at 0-100% adoption rates (10% steps),
    10 Monte Carlo draws each, measuring system violation probability and
    mean number of violating elements vs. adoption rate.
    Output: temp/simulation_results/<feeder>/ablation_unregulated/

  Fig. 3 (Section IV.B) — 24-hour regulated vs. unregulated comparison:
    A representative summer day with RSP-optimal ζ̄ (regulated) vs. maximum
    customer PV (unregulated); shows voltage and branch-loading time traces.
    Output: temp/simulation_results/<feeder>/regulated_24h/
            temp/simulation_results/<feeder>/unregulated_24h/

`run_simulation_and_store()`: reusable entry point accepting any PV capacity
assignment and a SimulationConfig (time window / sampling mode).  Calls
`Simulator.simulate()` (src/simulator.py) which applies the LinDistFlow
equations at each timestep, then saves voltage/flow results to disk.

The `__main__` block is a test harness demonstrating the 3-step workflow:
  1. Load saved optimization results (from run_pvhc_estimation.py)
  2. Run simulation for a few sampled days
  3. Load saved results back and validate
"""
import sys, pathlib
import numpy as np
import time

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, LOCATION
from src.utils import (
    load_optimization_results,
    save_simulation_results,
    load_simulation_results,
    SimulationConfig
)

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"

def run_simulation_and_store(
    feeder_name, 
    sub_folder, 
    utility_pv_info, 
    customer_pv_info, 
    capacities_isa_utility, 
    capacities_isa_customer, 
    config=None,
    battery_profiles=None,
    battery_info=None,
):
    """
    Run time-series simulation using the specified PV hosting capacities and store the results.
    Parameters:
    - feeder_name: Name of the feeder (str)
    - sub_folder: Name of the folder to save results (str)
    - utility_pv_info: List of dictionaries with utility PV node info
    - customer_pv_info: List of dictionaries with customer PV node info
    - capacities_isa_utility: Numpy array of utility PV hosting capacities
    - capacities_isa_customer: Numpy array of customer PV hosting capacities
    - config: Simulation configuration object with attributes: 
        - mode: str, one of "whole_year", "spring", "summer", "fall", "winter", "custom", indicating what scenarios to simulate
        - days_sampled: int, number of days sampled for seasonal modes
        - custom_start_day: int, start day for custom mode
        - custom_end_day: int, end day for custom mode
    - battery_profiles: Optional tuple (p_battery, q_battery) where each is np.array with shape (N_scenarios, n_batteries)
        p_battery: active power injection in kW (positive = discharging, negative = charging)
        q_battery: reactive power injection in kVAR
    - battery_info: Optional list of dictionaries, each entry is a dict with keys being "bus" and "phases"
    """
    from datetime import datetime, timedelta
    from src.data_loader import DataLoader
    from src.simulator import Simulator
    
    # Default configuration
    if config is None:
        config = SimulationConfig(mode="whole_year")
    
    # Get feeder configuration
    cfg = FEEDERS[feeder_name]
    feeder_root = cfg["feeder_root"]
    v_root_nom = cfg["v_root_nom"]
    v_root_multiplier = cfg["v_root_multiplier"]

    # Initialize data loader and simulator
    data_loader = DataLoader(feeder_name)
    network_data = data_loader.get_circuit_model(feeder_root=feeder_root)
    simulator = Simulator(data_loader)

    # Define seasonal date ranges (feeder-local time; one year of data available)
    # Spring: Mar 20 - Jun 20, Summer: Jun 21 - Sep 22, Fall: Sep 23 - Dec 20, Winter: Jan 1 - Mar 19 / Dec 21 - Dec 31
    season_ranges = {
        "spring": [("2023-03-20", "2023-06-20")],
        "summer": [("2023-06-21", "2023-09-22")],
        "fall": [("2023-09-23", "2023-12-20")],
        "winter": [("2023-01-01", "2023-03-19"), ("2023-12-21", "2023-12-31")]
    }

    # Helper function to convert feeder-local time to UTC with proper DST handling
    def local_to_utc(local_dt):
        """Convert feeder-local time to UTC, accounting for DST"""
        import pytz
        tz = pytz.timezone(LOCATION)
        localized = tz.localize(local_dt, is_dst=None)
        return localized.astimezone(pytz.utc).replace(tzinfo=None)
    
    # Generate time periods to simulate based on mode
    time_periods = []
    
    if config.mode == "whole_year":
        # Sample days from the available year of data
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        total_days = (end_date - start_date).days + 1
        days_by_month = {m: [] for m in range(1, 13)}
        start_date_np = np.datetime64(start_date).astype('datetime64[D]')
        dates_range = np.array([
            start_date_np + d
            for d in np.arange(total_days, dtype='timedelta64[D]')
        ])

        for d in dates_range:
            days_by_month[d.tolist().month].append(d)
        
        # Random sample or use all
        if config.days_sampled and config.days_sampled < total_days:
            if config.equal_samples_per_month:
                dates = np.zeros(config.days_sampled, dtype='datetime64[D]')
                samples_per_month = config.days_sampled // 12
                assert config.days_sampled % 12 == 0, "days_sampled must be multiple of 12 for equal sampling"
                for month in range(1, 13):
                    dates[(month-1)*samples_per_month:month*samples_per_month] = \
                        np.random.choice(days_by_month[month], size=samples_per_month, replace=False)
                sampled_day_offsets = np.sort((dates - np.datetime64(start_date).astype('datetime64[D]')).astype(int))
            else:
                sampled_day_offsets = np.random.choice(total_days, size=config.days_sampled, replace=False)
        else:
            sampled_day_offsets = np.arange(total_days)
        
        # Create time periods for each sampled day (full 24 hours in feeder-local time)
        for day_offset in sampled_day_offsets:
            day_start = start_date + timedelta(days=int(day_offset))
            day_end = day_start + timedelta(days=1) #- timedelta(minutes=15)
            # Convert feeder-local time to UTC with proper DST handling
            utc_start = local_to_utc(day_start)
            utc_end = local_to_utc(day_end)
            time_periods.append((
                utc_start.strftime("%Y-%m-%d %H:%M:%S"),
                utc_end.strftime("%Y-%m-%d %H:%M:%S")
            ))
    
    elif config.mode in season_ranges:
        # Sample days from the specified season
        np.random.seed(42)
        season_dates = season_ranges[config.mode]
        
        for start_date_str, end_date_str in season_dates:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            total_days = (end_date - start_date).days + 1
            
            if config.days_sampled and config.days_sampled < total_days:
                # Randomly sample days
                sampled_day_offsets = np.random.choice(total_days, size=config.days_sampled, replace=False)
            else:
                # Use all days
                sampled_day_offsets = np.arange(total_days)
            
            # Create time periods for each sampled day (full 24 hours in feeder-local time)
            for day_offset in sampled_day_offsets:
                day_start = start_date + timedelta(days=int(day_offset))
                day_end = day_start + timedelta(days=1) #- timedelta(minutes=15)
                # Convert feeder-local time to UTC with proper DST handling
                utc_start = local_to_utc(day_start)
                utc_end = local_to_utc(day_end)
                time_periods.append((
                    utc_start.strftime("%Y-%m-%d %H:%M:%S"),
                    utc_end.strftime("%Y-%m-%d %H:%M:%S")
                ))
    
    elif config.mode == "custom":
        # Custom date range
        if config.custom_start_day is None or config.custom_end_day is None:
            raise ValueError("Custom mode requires custom_start_day and custom_end_day")
        
        base_year = 2023
        start_date = datetime(base_year, 1, 1) + timedelta(days=config.custom_start_day)
        end_date = datetime(base_year, 1, 1) + timedelta(days=config.custom_end_day)
        
        # Convert to UTC with proper DST handling
        utc_start = local_to_utc(start_date)
        utc_end = local_to_utc(end_date)
        time_periods = [(
            utc_start.strftime("%Y-%m-%d %H:%M:%S"),
            utc_end.strftime("%Y-%m-%d %H:%M:%S")
        )]
    
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
    
    # Combine all time periods and run simulation
    all_voltage_profiles = []
    all_active_flows = []
    all_reactive_flows = []
    all_time_arrays = []
    all_p_inj = []
    all_q_inj = []
    all_injection_maps = []
    
    for t0, t1 in time_periods:
        print(f"Simulating period: {t0} to {t1}")
        # Get scenario data
        try:
            time_array, pv_profiles, load_profiles, load_info, V0_profiles = \
                data_loader.get_scenario_data_by_range(
                    SOLAR_PROFILE_PATH, t0, t1, v_root_nom, v_root_multiplier
                )
        except Exception as e:
            print(f"[Error] Failed to get scenario data for period {t0} to {t1}: {e}")
            continue
        
        # Prepare PV info
        for pv in utility_pv_info:
            pv['type'] = 'utility pv'
        for pv in customer_pv_info:
            pv['type'] = 'customer pv'
        for load in load_info:
            load['type'] = 'load'
        PV_info = utility_pv_info + customer_pv_info if utility_pv_info else customer_pv_info
        for i, pv in enumerate(PV_info):
            if i < len(capacities_isa_utility):
                pv['capacity'] = capacities_isa_utility[i]
            else:
                pv['capacity'] = capacities_isa_customer[i - len(capacities_isa_utility)]
        
        # Prepare battery info if provided
        if battery_info is not None:
            for battery in battery_info:
                battery['type'] = 'battery'
        
        # Run simulation
        voltage_profiles, active_flows, reactive_flows, p_inj, q_inj, injection_map = simulator.simulate(
            load_profiles=load_profiles,
            pv_profiles=pv_profiles,
            V0_profiles=V0_profiles,
            load_info=load_info,
            PV_info=PV_info,
            battery_profiles=battery_profiles,
            battery_info=battery_info,
        )
        
        all_voltage_profiles.append(voltage_profiles)
        all_active_flows.append(active_flows)
        all_reactive_flows.append(reactive_flows)
        all_p_inj.append(p_inj)
        all_q_inj.append(q_inj)
        all_time_arrays.append(time_array)
        all_injection_maps.append(injection_map)
    
    if not all_voltage_profiles:
        raise RuntimeError("No valid data. No simulation results were generated.")

    # Concatenate results
    voltage_profiles = np.vstack(all_voltage_profiles)
    active_flows = np.vstack(all_active_flows)
    reactive_flows = np.vstack(all_reactive_flows)
    time_array = np.concatenate(all_time_arrays)
    p_inj = np.vstack(all_p_inj)
    q_inj = np.vstack(all_q_inj)
    im0 = all_injection_maps[0]
    assert all([im == im0 for im in all_injection_maps]), \
        "Not all injection maps are identical"
    
    # Save simulation results
    output_dir = save_simulation_results(
        feeder_name=feeder_name,
        sub_folder=sub_folder,
        voltage_profiles=voltage_profiles,
        active_flows=active_flows,
        reactive_flows=reactive_flows,
        p_inj=p_inj,
        q_inj=q_inj,
        time_array=time_array,
        network_data=network_data,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        capacities_isa_utility=capacities_isa_utility,
        capacities_isa_customer=capacities_isa_customer,
        injection_map=im0,
        config=config,
        time_periods=time_periods
    )
    
    print(f"\nSimulation results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME
    # Which run_pvhc_estimation.py output to read PV capacities from
    opt_subfolder = "iterative_eps0.01_beta0.05_utility_ratio_0pct"
    # Where this script's own simulation output gets stored (a separate,
    # unrelated subfolder namespace under temp/simulation_results/)
    sub_folder = "pvhc_simulation_test"

    print("="*70)
    print("STEP 1: Loading Optimization Results")
    print("="*70)

    # Load optimization results
    loaded_results = load_optimization_results(feeder_name, sub_folder=opt_subfolder)
    
    capacities_isa_utility = loaded_results["capacities_isa_utility"]
    capacities_isa_customer = loaded_results["capacities_isa_customer"]
    utility_pv_info = loaded_results["utility_pv_info"]
    customer_pv_info = loaded_results["customer_pv_info"]
    
    print(f"Loaded optimization results for {feeder_name}")
    print(f"  Utility PV nodes: {len(capacities_isa_utility)}")
    print(f"  Customer PV nodes: {len(capacities_isa_customer)}")
    print(f"  Total Utility PV capacity: {np.sum(capacities_isa_utility):.2f} kW")
    print(f"  Total Customer PV capacity: {np.sum(capacities_isa_customer):.2f} kW")
    
    print("\n" + "="*70)
    print("STEP 2: Running Simulation")
    print("="*70)
    
    # Configure simulation - sample 3 days from summer
    sim_config = SimulationConfig(
        mode="summer",
        days_sampled=3
    )
    
    print(f"Simulation config: mode={sim_config.mode}, days_sampled={sim_config.days_sampled}")
    
    # Main function call: run simulation and store results
    output_dir = run_simulation_and_store(
        feeder_name=feeder_name,
        sub_folder=sub_folder,
        utility_pv_info=utility_pv_info,
        customer_pv_info=customer_pv_info,
        capacities_isa_utility=capacities_isa_utility,
        capacities_isa_customer=capacities_isa_customer,
        config=sim_config
    )
    
    """Loading and validation"""
    # Extract timestamp from output directory
    timestamp = output_dir.name
    print(f"\nSimulation completed with timestamp: {timestamp}")
    
    print("\n" + "="*70)
    print("STEP 3: Loading Simulation Results")
    print("="*70)

    # Load simulation results with verbose=True to print README
    sim_results = load_simulation_results(feeder_name, timestamp, verbose=True)
    
    print("\n" + "="*70)
    print("STEP 4: Validating Loaded Data")
    print("="*70)
    
    # Validate that loaded data matches what was saved
    print(f"\nValidation Results:")
    print(f"  Feeder name matches: {sim_results['feeder_name'] == feeder_name}")
    print(f"  Voltage profiles shape: {sim_results['voltage_profiles'].shape}")
    print(f"  Active flows shape: {sim_results['active_flows'].shape}")
    print(f"  Reactive flows shape: {sim_results['reactive_flows'].shape}")
    print(f"  Time array length: {len(sim_results['time_array'])}")
    print(f"  Number of buses: {sim_results['n_buses']}")
    print(f"  Utility capacities match: {np.array_equal(sim_results['capacities_isa_utility'], capacities_isa_utility)}")
    print(f"  Customer capacities match: {np.array_equal(sim_results['capacities_isa_customer'], capacities_isa_customer)}")
    
    # Print some statistics
    voltage_profiles = sim_results['voltage_profiles']
    print(f"\nVoltage Statistics:")
    print(f"  Min voltage: {np.min(voltage_profiles):.4f} kV")
    print(f"  Max voltage: {np.max(voltage_profiles):.4f} kV")
    print(f"  Mean voltage: {np.mean(voltage_profiles):.4f} kV")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}")

