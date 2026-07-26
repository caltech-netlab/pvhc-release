"""
Plotting Functions for Paper Figures — Figs. 1-5

Provides visualization routines for optimization and simulation results.

Paper figure mapping:
  Fig. 1 (unregulated adoption ablation) — `plot_sim_ablation`:
    System violation probability and number of elements in violation vs.
    customer PV adoption rate 0-100%; from ablation runs in
    temp/simulation_results/<feeder>/ablation_unregulated/.

  Fig. 2 (spatiotemporal violation heatmap) — `plot_sim_365`:
    Full-year violation heatmap at a single high adoption rate.

  Fig. 3 (24-hour simulation) — `plot_sim_24`:
    Voltage and branch-loading timeseries for a representative day;
    compares regulated (optimized ζ̄) vs. unregulated (random full capacity).

  Fig. 4 (PV capacity distribution) — `plot_opt_results`:
    Histogram of per-customer PV capacities ζ̄_c; compares "without_fairness"
    (baseline Eq. 10) vs. "with_fairness" (Eq. 16 fairness objective).

  Fig. 5 (utility ratio ablation) — `plot_opt_results` over subfolders:
    Six panels (a.1-b.3) for utility_pv_ratio ∈ {0%, 30%, 100%} × fairness.

Key functions:
  plot_opt_results(feeder_name, opt_subfolders): histogram + GIS capacity map
  plot_sim_365(feeder_name, sim_subfolders):    heatmaps for full-year simulation
  plot_sim_24(feeder_name, sim_subfolders):     timeseries for 24-h window
  plot_sim_ablation(feeder_name):               adoption rate vs. violation curve
  plot_opt_w_batt(feeder_name, opt_subfolders): battery capacity maps (Table II)
"""
import sys, pathlib
import numpy as np

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import *
from src import utils
from src.visualization import *
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.analyzer import PVHCProblemWithoutBattery
from matplotlib.colors import LinearSegmentedColormap
import re


def plot_opt_results(
    feeder_name: str,
    opt_subfolders: list = ["with_fairness", "without_fairness"], 
    base_output_dir: str = 'temp/plots',
    ext='png',
    ylim_max=300,
    ytick_interval=50,
    normalized=False
):
    feeder_data_path = FEEDERS[feeder_name]["data_path"]
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    """
    Load and plot optimization results
    """
    for sub_folder in opt_subfolders:
        # Load optimization results
        opt_results = utils.load_optimization_results(feeder_name, sub_folder=sub_folder)
        capacities_isa_utility = opt_results["capacities_isa_utility"]
        capacities_isa_customer = opt_results["capacities_isa_customer"]
        utility_pv_info = opt_results.get("utility_pv_info", [])
        customer_pv_info = opt_results.get("customer_pv_info", [])
        node_list = opt_results["node_list"]
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in opt_results["edge_list"]]
        
        # Plot: histogram of PV capacities with consistent y-axis for academic figures
        plot_pv_capacity_distribution(
            capacities_isa_customer, 
            capacities_isa_utility, 
            feeder_name, 
            output_dir,
            sub_folder=sub_folder,
            ext=ext,
            ylim_max=ylim_max,
            ytick_interval=ytick_interval,
            normalized=normalized,
            customer_pv_info=customer_pv_info,
            utility_pv_info=utility_pv_info
        )
        # Plot GIS visualization of PV capacities
        node_coord_file = pathlib.Path(feeder_data_path) / "opendss" / "Buscoords_ckt5.dss"
        node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
        fill_missing_node_coordinates(node_coords_array, edge_list)
        # Sum capacities across phases: bus_name -> total capacity
        pv_cap_node = np.zeros(len(node_list))
        added_bus_phases = dict()
        for (cap, payload) in zip(capacities_isa_utility.tolist() + capacities_isa_customer.tolist(), opt_results["utility_pv_info"] + opt_results["customer_pv_info"]):
            if payload["bus"].lower() not in added_bus_phases:
                added_bus_phases[payload["bus"].lower()] = set()
            if payload["phases"] not in added_bus_phases[payload["bus"].lower()]:
                added_bus_phases[payload["bus"].lower()].add(payload["phases"])
                pv_cap_node[node_list.index(payload["bus"].lower())] += cap
        gis_visualize(
            node_coords=node_coords_array,
            nodal_values=pv_cap_node,
            edge_list=edge_list,
            outfile=output_dir / f'map_pv_capacity_{sub_folder}.{ext}',
            cb_label_nodal='PV Capacity (kW)',
            cmap_nodal=LinearSegmentedColormap.from_list('blue_yellow_red', ['blue', 'yellow', 'red']),
            title=f'PV Capacity Map: {pv_cap_node.sum():.0f} kW',
            nonzero_only=True
        )
        
        
def plot_sim_365(
    feeder_name: str,
    sim_subfolders: list = [],
    base_output_dir: str = 'temp/plots',
    ext='png'
):
    feeder_root = FEEDERS[feeder_name]["feeder_root"]
    feeder_data_path = FEEDERS[feeder_name]["data_path"]
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    v_min, v_max = 1 - VOL_TOL, 1 + VOL_TOL
    
    """
    Load and plot simulation results
    """
    # Nominal voltages
    data_loader = DataLoader(feeder_name)
    data_loader.get_circuit_model(feeder_root=feeder_root)
    simulator = Simulator(data_loader)
    problem = PVHCProblemWithoutBattery(simulator, [], [], [], load_cached_limits=True)
    limits = problem.get_network_limits()
    nominal_voltages = problem.get_nominal_voltage(squared=False)
    voltage_upper_limits = np.sqrt(limits['voltage_upper'])  # Shape: (n_buses * 3,)
    voltage_lower_limits = np.sqrt(limits['voltage_lower'])  # Shape: (n_buses * 3,)
    for sub_folder in sim_subfolders:
        # Load simulation result
        sim_results = utils.load_simulation_results(feeder_name, sub_folder=sub_folder)
        t_local = np.array([utils.utc2local(t) for t in sim_results["time_array"]])
        node_list = sim_results["node_list"]
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in sim_results["edge_list"]]
        T, n = len(t_local), len(node_list) - 1
        v_pu = sim_results["voltage_profiles"] / nominal_voltages[np.newaxis, :]
        S_mag = np.sqrt(sim_results["active_flows"] ** 2 + sim_results["reactive_flows"] ** 2)
        branch_loading = np.max((S_mag / limits["S_ratings"][np.newaxis, :]).reshape(T, n, 3), axis=2)

        # Bus-level plotting functions
        # Plot over/under-voltage violation heatmaps: x-axis: 24 hours, y-axis: bus indices
        plot_v_bus_heatmap(
            t_local, 
            sim_results["voltage_profiles"],
            voltage_upper_limits,
            voltage_lower_limits,
            output_dir,
            sub_folder=sub_folder.replace("/", "_"),
            violation_type="over",
            ext=ext
        )
        plot_v_bus_heatmap(
            t_local, 
            sim_results["voltage_profiles"],
            voltage_upper_limits,
            voltage_lower_limits,
            output_dir,
            sub_folder=sub_folder.replace("/", "_"),
            violation_type="under",
            ext=ext
        )

        # Non-GIS (heatmap/timeseries) plotting functions
        # Plot voltage violation heatmap over 24 hours and 365 days
        plot_violation_24_365(
            t_local, 
            sim_results["voltage_profiles"], 
            voltage_upper_limits,
            voltage_lower_limits,
            output_dir / f"heatmap_voltage_violation_24_365_{sub_folder.replace('/', '_')}.{ext}",
            title='Average Voltage Violation Rate 24/365'
        )
        # Plot branch flow violation heatmap over 24 hours and 365 days
        plot_violation_24_365(
            t_local, 
            S_mag, 
            limits["S_ratings"][:],
            np.zeros_like(limits["S_ratings"])[:],
            output_dir / f"heatmap_branchflow_violation_24_365_{sub_folder.replace('/', '_')}.{ext}",
            violation_type="over",
            title='Average Branch Flow Violation Rate 24/365', 
            clim=(0.0, 0.01),
        )
        plot_voltage_violation_duration(
            t_local,
            v_pu,
            v_min=v_min,
            v_max=v_max,
            outfile=output_dir / f'voltage_violation_duration_{sub_folder.replace("/", "_")}.{ext}'
        )
        plot_voltage_violation_duration_amount(
            t_local,
            v_pu,
            v_min=v_min,
            v_max=v_max,
            outfile=output_dir / f'voltage_violation_duration_amount_{sub_folder.replace("/", "_")}.{ext}'
        )
        violation_duration_hist(
            t_local,
            v_pu > v_max,
            outfile=output_dir / f'voltage_violation_duration_histogram_{sub_folder.replace("/", "_")}.{ext}',
            title='Voltage Violation Duration'
        )
        violation_duration_hist(
            t_local,
            branch_loading > 1,
            outfile=output_dir / f'branch_capacity_violation_duration_histogram_{sub_folder.replace("/", "_")}.{ext}',
            title='Branch Loading Violation Duration'
        )
        
        """GIS violation probability map"""
        node_coord_file = pathlib.Path(feeder_data_path) / "opendss" / "Buscoords_ckt5.dss"
        node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
        fill_missing_node_coordinates(node_coords_array, edge_list)
        # Compute violation probabilities
        voltage_violation_prob = np.mean((v_pu > v_max).astype(int).reshape(T, n, 3), axis=(0, 2))
            # + np.mean((v_pu < v_min).astype(int).reshape(T, n, 3), axis=(0, 2))
        branch_violation_prob = np.mean((branch_loading > 1).astype(int), axis=0)
        gis_visualize(
            node_coords_array,
            output_dir / f'map_violation_probability_{sub_folder.replace("/", "_")}.{ext}',
            nodal_values=voltage_violation_prob,    
            edge_values=branch_violation_prob,
            edge_list=edge_list,
            cb_label_nodal=f'Over Voltage Violation Probability (>{VOL_TOL} p.u.)',
            cb_label_edge=f'Branch Capacity Violation Probability (>100%)',
            clim_nodal=(0.0, 0.2),
            clim_edge=(0.0, 0.2),
            zero_length_edge_threshold=0.1,
            cmap_nodal=LinearSegmentedColormap.from_list('black_red', ['black', 'red']),
            cmap_edge=LinearSegmentedColormap.from_list('black_red', ['black', 'red']),
            title='System Violation Probability',
            text="Thin lines: branches with no power ratings.\n\"X\": zero-length branches."\
                if PLOT_COLORBAR_LABEL else None
        )


def plot_sim_24(
    feeder_name: str,
    sim_subfolders: list = ["regulated_24h", "unregulated_24h"],
    base_output_dir: str = 'temp/plots',
    ext='png'
):
    feeder_root = FEEDERS[feeder_name]["feeder_root"]
    feeder_data_path = FEEDERS[feeder_name]["data_path"]
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    v_min, v_max = 1 - VOL_TOL, 1 + VOL_TOL

    """
    Load and plot simulation results
    """
    # Nominal voltages
    data_loader = DataLoader(feeder_name)
    data_loader.get_circuit_model(feeder_root=feeder_root)
    simulator = Simulator(data_loader)
    problem = PVHCProblemWithoutBattery(simulator, [], [], [], load_cached_limits=True) 
    limits = problem.get_network_limits()
    nominal_voltages = problem.get_nominal_voltage(squared=False)

    for sub_folder in sim_subfolders:
        # Load simulation result
        sim_results = utils.load_simulation_results(feeder_name, sub_folder=sub_folder)
        v_pu = sim_results["voltage_profiles"] / nominal_voltages[np.newaxis, :]
        t_local = np.array([utils.utc2local(t) for t in sim_results["time_array"]])
        node_list = sim_results["node_list"]
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in sim_results["edge_list"]]
        T = len(t_local)
        n = len(node_list) - 1
        S_mag = np.sqrt(sim_results["active_flows"] ** 2 + sim_results["reactive_flows"] ** 2)
        branch_loading = np.max((S_mag / limits["S_ratings"][np.newaxis, :]).reshape(T, n, 3), axis=2)
        
        # Plot voltage violation count over time   
        plot_voltage_violations_count(
            t_local, 
            v_pu, 
            v_min=v_min, 
            v_max=v_max, 
            outfile=output_dir / f'voltage_violation_count_{sub_folder}.{ext}'
        )
        plot_values_range_timeseries(
            t_local,
            v_pu,
            'Voltage (p.u.)',
            title=f"Bus voltages for {n} buses on {str(t_local[0])[:10]}",
            v_min=v_min,
            v_max=v_max,
            ylim=(0.9, 1.1),
            outfile=output_dir / f'range_voltage_{sub_folder}.{ext}'
        )
        plot_values_range_timeseries(
            t_local,
            branch_loading.reshape(T, -1),
            'Branch Loading Ratio',
            title=f"Branch loading for {n} branches on {str(t_local[0])[:10]}",
            v_max=1.0,
            ylim=(0, 1 if (branch_loading <= 1).all() else 2.0),
            outfile=output_dir / f'range_branch_loading_{sub_folder}.{ext}'
        )
        plot_pu_min_area_under_curve(
            t_local,
            v_pu,
            node_list,
            v_min=v_min,
            v_max=v_max,
            outfile=output_dir / f'pu_min_area_under_curve_{sub_folder}.{ext}'
        )
        results = load_dict_calculations(sim_results["p_inj"], sim_results["q_inj"], sim_results["injection_map"], t=t_local)
        plot_net_load(
            results[4],
            results[6],
            results[8],
            t=t_local,
            outfile=output_dir / f'net_load_{sub_folder}.{ext}'
        )

        """Plot GIS visualization of bus voltages and branch flow."""
        node_coord_file = pathlib.Path(feeder_data_path) / "opendss" / "Buscoords_ckt5.dss"
        node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
        fill_missing_node_coordinates(node_coords_array, edge_list)
        # Take the worst voltage deviation across phases for each bus
        v_pu_reshaped = v_pu.reshape(v_pu.shape[0], len(node_list) - 1, 3)  # Shape: (time_steps, n_buses, 3) 
        where = np.argmax(np.abs(v_pu_reshaped - 1), axis=2)  # Shape: (time_steps, n_buses)
        v_dev_arr = np.take_along_axis(v_pu_reshaped, where[:, :, None], axis=2)
        # Take the max loading across phases for each branch
        gis_visualize_timeseries(
            t_local, 
            node_coords_array,
            v_dev_arr,    
            output_dir / f'map_bus_branch_{sub_folder}',
            edge_list=edge_list,
            edge_values=branch_loading,
            cb_label_nodal='Voltage (p.u.)',
            cb_label_edge='Branch Power Loading Ratio',
            title='System Voltage and Capacity',
            clim_nodal=(0.9, 1.1),
            clim_edge=(0.0, 2.0),
            cmap_nodal=LinearSegmentedColormap.from_list('blue_green_red', ['blue', 'green', 'red']),
            ext='png'
        )


def plot_sim_ablation(
    feeder_name: str,
    base_output_dir: str = 'temp/plots',
    sim_out_basedir=pathlib.Path(__file__).resolve().parents[1] / "temp" / "simulation_results",
    folder_prefix="",
    ext='png',
    figsize=(6, 5), #(width, height) in inches
    fontsize=12
):
    """
    x axis: adoption level 0% to 100%
    y axis: system-wide violation metric
    """
    # Scan directory for ablation simulation results
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_out_basedir = pathlib.Path(sim_out_basedir / feeder_name / "ablation_unregulated")
    folders = [f.name for f in sim_out_basedir.iterdir() if f.is_dir() and f.name.startswith(folder_prefix)]
    
    # Extract penetration rates from folder names
    penetration_rates_dict = {}
    for folder in folders:
        # Extract percentage from folder name (last 2-3 digits)
        match = re.search(r'pr_(\d{1,3})', folder)
        if match:
            percentage = int(match.group(1))
            if percentage not in penetration_rates_dict:
                penetration_rates_dict[percentage] = []
            penetration_rates_dict[percentage].append(folder)
    
    # Nominal voltages
    data_loader = DataLoader(feeder_name)
    data_loader.get_circuit_model(feeder_root=FEEDERS[feeder_name]["feeder_root"])
    simulator = Simulator(data_loader)
    problem = PVHCProblemWithoutBattery(simulator, [], [], [], load_cached_limits=True) 
    limits = problem.get_network_limits()
    nominal_voltages = problem.get_nominal_voltage(squared=False)
    
    # Sort by penetration rate and load simulation results
    # Metrics per penetration rate:
    # 1) System violation probability over time (timestamps in violation / total timestamps)
    # 2) Average number of elements in violation conditioned on violation occurring
    v_means_prob, v_stds_prob = [], []
    b_means_prob, b_stds_prob = [], []
    v_means_elements, v_stds_elements = [], []
    b_means_elements, b_stds_elements = [], []
    penetration_rates = list(sorted(penetration_rates_dict.keys()))
    for percentage in tqdm(penetration_rates, desc="Simulation ablation (penetration rates)"):
        # Aggregation lists per penetration percentage (over runs/folders):
        # - v_hours_list: total hours with any bus violating voltage limits in a run
        # - b_hours_list: total hours with any branch exceeding loading limit in a run
        # - v_elements_list: avg number of buses violating, conditioned on the system being in violation
        # - b_elements_list: avg number of branches violating, conditioned on the system being in violation
        v_prob_list = []
        b_prob_list = []
        v_elements_list = []
        b_elements_list = []
        for folder in penetration_rates_dict[percentage]:
            sim_results = utils.load_simulation_results(feeder_name, sub_folder=f"ablation_unregulated/{folder}")
            t_local = np.array([utils.utc2local(t) for t in sim_results["time_array"]])
            node_list = sim_results["node_list"]
            T, n = len(t_local), len(node_list) - 1
            
            # Calculate metric the summarizes total violation in the system
            v_pu = sim_results["voltage_profiles"] / nominal_voltages[np.newaxis, :]
            # Voltage violation per element (bus): any phase in last axis violating
            v_reshaped = v_pu.reshape(T, n, 3)
            v_elem_violation = (v_reshaped > 1.05)
            v_elem_any = v_elem_violation.any(axis=2)              # (T, n)
            v_system_any = v_elem_any.any(axis=1)                  # (T,)
            # System violation probability over time
            v_prob = v_system_any.sum() / T
            v_prob_list.append(v_prob)
            # Average number of elements in violation, conditioned on violation
            if np.any(v_system_any):
                v_elements_list.append(v_elem_any[v_system_any].sum(axis=1).mean())
            else:
                v_elements_list.append(0.0)
            
            # Calculate branch flow violation metric
            S_mag = np.sqrt(sim_results["active_flows"] ** 2 + sim_results["reactive_flows"] ** 2)
            branch_loading = S_mag.reshape(T, n, 3) / limits["S_ratings"].reshape(1, n, 3)
            # Branch violation per element: any phase in last axis violating
            b_elem_violation = (branch_loading > 1.0)
            b_elem_any = b_elem_violation.any(axis=2)              # (T, n)
            b_system_any = b_elem_any.any(axis=1)                  # (T,)
            # System violation probability over time
            b_prob = b_system_any.sum() / T
            b_prob_list.append(b_prob)
            # Average number of elements in violation, conditioned on violation
            if np.any(b_system_any):
                b_elements_list.append(b_elem_any[b_system_any].sum(axis=1).mean())
            else:
                b_elements_list.append(0.0)
        
        # Aggregate per penetration percentage
        v_means_prob.append(np.mean(v_prob_list))
        v_stds_prob.append(np.std(v_prob_list))
        b_means_prob.append(np.mean(b_prob_list))
        b_stds_prob.append(np.std(b_prob_list))
        v_means_elements.append(np.mean(v_elements_list))
        v_stds_elements.append(np.std(v_elements_list))
        b_means_elements.append(np.mean(b_elements_list))
        b_stds_elements.append(np.std(b_elements_list))
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    # Subplot 1: System violation probability over time
    ax1.errorbar(penetration_rates, v_means_prob, yerr=v_stds_prob, marker='o', capsize=3, label='Bus voltage violations', color=COLOR_CUSTOMER)
    ax1.errorbar(penetration_rates, b_means_prob, yerr=b_stds_prob, marker='s', capsize=3, label='Branch capacity violations', color=COLOR_UTILITY)
    # ax1.set_xlabel('Customer PV Adoption Rate (%)', fontsize=fontsize)
    ax1.set_ylabel('System violation probability', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_xticklabels([])     # Remove x tick labels for upper subplot
    # ax1.set_xticks([])
    ax1.legend(fontsize=fontsize)

    # Subplot 2: Avg number of elements in violation when violating
    ax2.errorbar(penetration_rates, v_means_elements, yerr=v_stds_elements, marker='o', capsize=3, label='Bus voltage violations', color=COLOR_CUSTOMER)
    ax2.errorbar(penetration_rates, b_means_elements, yerr=b_stds_elements, marker='s', capsize=3, label='Branch capacity violations', color=COLOR_UTILITY)
    ax2.set_xlabel('Customer PV Adoption Rate (%)', fontsize=fontsize)
    ax2.set_ylabel('Elements in violation', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax2.legend(fontsize=fontsize)
    ax1.grid(True, which='both', linestyle='--', alpha=GRID_ALPHA)
    ax2.grid(True, which='both', linestyle='--', alpha=GRID_ALPHA)
    if PLOT_TITLE:
        fig.suptitle(f'Unregulated Customer PV Adoption Ablation: {feeder_name.replace("_", " ")}')
    fig.tight_layout()
    plt.savefig(output_dir / f'ablation_violation_metrics.{ext}')
    plt.close(fig)


def plot_opt_w_batt(
    feeder_name: str,
    base_output_dir: str = 'temp/plots',
    opt_subfolders: list = ["with_battery/without_fairness", "with_battery/with_fairness"],
    ext='png'
):
    feeder_data_path = FEEDERS[feeder_name]["data_path"]
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    """
    Load and plot battery optimization results
    """
    for sub_folder in opt_subfolders:
        opt_results = utils.load_optimization_results(feeder_name, sub_folder=sub_folder)
        capacities_isa_utility = opt_results["capacities_isa_utility"]
        capacities_isa_customer = opt_results["capacities_isa_customer"]
        node_list = opt_results["node_list"]
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in opt_results["edge_list"]]
        
        # Plot: histogram of PV capacities
        plot_pv_capacity_distribution(
            capacities_isa_customer, 
            capacities_isa_utility, 
            feeder_name, 
            output_dir,
            sub_folder=sub_folder.replace('/', '_'),
            ext=ext
        )

        # Plot GIS visualization of battery capacities
        node_coord_file = pathlib.Path(feeder_data_path) / "opendss" / "Buscoords_ckt5.dss"
        node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
        fill_missing_node_coordinates(node_coords_array, edge_list)
        # Sum capacities across phases: bus_name -> total capacity
        batt_cap_node = np.zeros(len(node_list))
        added_bus_phases = dict()
        for batt in opt_results["utility_battery_info"]:
            if batt["bus"].lower() not in added_bus_phases:
                added_bus_phases[batt["bus"].lower()] = set()
            if batt["phases"] not in added_bus_phases[batt["bus"].lower()]:
                added_bus_phases[batt["bus"].lower()].add(batt["phases"])
                batt_cap_node[node_list.index(batt["bus"].lower())] += batt["S"]
        gis_visualize(
            node_coords=node_coords_array,
            nodal_values=batt_cap_node,
            edge_list=edge_list,
            outfile=output_dir / f'map_batt_capacity_{sub_folder.replace("/", "_")}.{ext}',
            cb_label_nodal='Battery Capacity (kW)',
            cmap_nodal=LinearSegmentedColormap.from_list('blue_yellow_red', ['blue', 'yellow', 'red']),
            title=f'Battery Capacity Map: {batt_cap_node.sum():.0f} kWh',
            nonzero_only=True
        )

        # Plot battery injections
        T = opt_results['p_b'].shape[0]
        plot_batt_inj(
            opt_results['p_b'],
            opt_results['q_b'],
            t=np.array([np.datetime64('2024-01-01') + np.timedelta64(int(i * 24 * 60 / T), 'm') for i in range(len(opt_results['p_b']))]),
            outfile=output_dir / f'battery_injs_24h_{sub_folder.replace("/", "_")}.{ext}'
        )

def plot_sim_w_batt_24h(
    feeder_name: str,
    sim_subfolders: list = ["regulated_24h_batt"],
    base_output_dir: str = 'temp/plots',
    ext='png'
):
    """
    Plot simulation results with battery for 24-hour period.
    Similar to plot_sim_24 but includes battery-specific visualizations.
    """
    output_dir = pathlib.Path(base_output_dir) / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for sub_folder in sim_subfolders:
        # Load simulation result
        sim_results = utils.load_simulation_results(feeder_name, sub_folder=sub_folder)
        t_local = np.array([utils.utc2local(t) for t in sim_results["time_array"]])
        
        results = load_dict_calculations(sim_results["p_inj"], sim_results["q_inj"], sim_results["injection_map"], t=t_local)
        inj_dict, load_dict, pv_dict, batt_dict = results[:4]
        total_net_p, total_net_q = results[4], results[5]
        total_load_p, total_load_q = results[6], results[7]
        total_pv_p, total_pv_q = results[8], results[9]
        total_battery_p, total_battery_q = results[10], results[11]
        
        plot_net_load(
            total_net_p,
            total_load_p,
            total_pv_p,
            total_battery_p,
            t=t_local,
            outfile=output_dir / f'net_load_{sub_folder}.{ext}'
        )
