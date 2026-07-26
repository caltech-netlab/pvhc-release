#!/usr/bin/env python3
"""Quick script to print PV capacity summary table for battery optimization results."""

import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.utils import load_optimization_results
from src.constants import IEEE_STYLE, COLOR_UTILITY, COLOR_CUSTOMER, DEFAULT_FEEDER_NAME


def print_summary_table(feeder_name: str, config_folders: dict):
    """Print summary table of PV capacities and battery installations."""
    
    # Load all results
    results_dict = {}
    for name, folder in config_folders.items():
        try:
            results = load_optimization_results(feeder_name, folder)
            results_dict[name] = results
            print(f"[OK] Loaded {folder}")
        except Exception as e:
            print(f"[FAILED] Failed to load {folder}: {e}")
    
    if not results_dict:
        print("Error: No results loaded.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY TABLE: PV CAPACITY AND BATTERY INSTALLATION")
    print("="*80)
    print(f"Feeder: {feeder_name}\n")
    
    # Collect data
    table_data = []
    for config_name, results in sorted(results_dict.items()):
        pv_utility = results.get("capacities_isa_utility", np.array([]))
        pv_customer = results.get("capacities_isa_customer", np.array([]))
        total_pv_utility = pv_utility.sum() if len(pv_utility) > 0 else 0.0
        total_pv_customer = pv_customer.sum() if len(pv_customer) > 0 else 0.0
        total_pv = total_pv_utility + total_pv_customer
        
        zeta_b = results.get("zeta_b")
        if zeta_b is not None:
            total_battery_installed = zeta_b.sum()
            num_active_batteries = np.sum(zeta_b >= 0.5)
            avg_zeta_b = zeta_b.mean()
            max_zeta_b = zeta_b.max()
        else:
            total_battery_installed = 0.0
            num_active_batteries = 0
            avg_zeta_b = 0.0
            max_zeta_b = 0.0
        
        objective = results.get("objective_value", results.get("total_capacity", 0.0))
        
        table_data.append({
            'config': config_name,
            'total_pv': total_pv,
            'pv_utility': total_pv_utility,
            'pv_customer': total_pv_customer,
            'battery_sum': total_battery_installed,
            'battery_active': num_active_batteries,
            'battery_avg': avg_zeta_b,
            'battery_max': max_zeta_b,
            'objective': objective
        })
    
    # Print table
    print(f"{'Configuration':<20} {'Total PV':>12} {'Utility PV':>12} {'Customer PV':>12} "
          f"{'Sum zeta_b':>10} {'Active':>8} {'Avg zeta_b':>10} {'Max zeta_b':>10} {'Objective':>12}")
    print(f"{'':20} {'(kW)':>12} {'(kW)':>12} {'(kW)':>12} "
          f"{'':>10} {'Batt.':>8} {'':>10} {'':>10} {'Value':>12}")
    print("-" * 125)
    
    for row in table_data:
        print(f"{row['config']:<20} "
              f"{row['total_pv']:>12.2f} "
              f"{row['pv_utility']:>12.2f} "
              f"{row['pv_customer']:>12.2f} "
              f"{row['battery_sum']:>10.2f} "
              f"{row['battery_active']:>8d} "
              f"{row['battery_avg']:>10.4f} "
              f"{row['battery_max']:>10.4f} "
              f"{row['objective']:>12.2f}")
    
    print("-" * 125)
    
    # Print insights
    if len(table_data) > 1:
        print("\nKey Insights:")
        baseline = next((row for row in table_data if 'limit 0' in row['config'].lower() or 'baseline' in row['config'].lower()), None)
        
        if baseline:
            baseline_pv = baseline['total_pv']
            print(f"  Baseline (no batteries): {baseline_pv:.2f} kW total PV")
            
            for row in table_data:
                if row['config'] != baseline['config'] and row['battery_sum'] > 0:
                    pv_increase = row['total_pv'] - baseline_pv
                    pv_increase_pct = (pv_increase / baseline_pv * 100) if baseline_pv > 0 else 0
                    pv_per_battery = pv_increase / row['battery_sum'] if row['battery_sum'] > 0 else 0
                    print(f"  {row['config']}: +{pv_increase:.2f} kW ({pv_increase_pct:+.1f}%) "
                          f"with {row['battery_sum']:.1f} battery units ({pv_per_battery:.2f} kW/battery)")
        
        max_pv_row = max(table_data, key=lambda x: x['total_pv'])
        print(f"\n  Maximum PV achieved: {max_pv_row['total_pv']:.2f} kW ({max_pv_row['config']})")
    
    print("="*80 + "\n")


def plot_pv_vs_active_batteries(feeder_name: str, config_folders: dict, save_path: str = None):
    """Plot total PV capacity vs number of active batteries.
    
    Args:
        feeder_name: Name of the feeder
        config_folders: Dictionary mapping config names to folder names
        save_path: Optional path to save the figure
    """
    # Load all results
    results_dict = {}
    for name, folder in config_folders.items():
        try:
            results = load_optimization_results(feeder_name, folder)
            results_dict[name] = results
        except Exception as e:
            print(f"Warning: Failed to load {folder}: {e}")
    
    if not results_dict:
        print("Error: No results loaded for plotting.")
        return
    
    # Collect data
    active_batteries = []
    total_pv_capacities = []
    config_labels = []
    
    for config_name, results in sorted(results_dict.items()):
        pv_utility = results.get("capacities_isa_utility", np.array([]))
        pv_customer = results.get("capacities_isa_customer", np.array([]))
        total_pv_utility = pv_utility.sum() if len(pv_utility) > 0 else 0.0
        total_pv_customer = pv_customer.sum() if len(pv_customer) > 0 else 0.0
        total_pv = total_pv_utility + total_pv_customer
        
        zeta_b = results.get("zeta_b")
        if zeta_b is not None:
            num_active = np.sum(zeta_b >= 0.5)
        else:
            num_active = 0
        
        active_batteries.append(num_active)
        total_pv_capacities.append(total_pv)
        config_labels.append(config_name)
    
    # Apply IEEE style
    plt.rcParams.update(IEEE_STYLE)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot data
    sorted_indices = np.argsort(active_batteries)
    active_batteries = np.array(active_batteries)[sorted_indices]
    total_pv_capacities = np.array(total_pv_capacities)[sorted_indices]
    config_labels = np.array(config_labels)[sorted_indices]
    ax.plot(active_batteries, total_pv_capacities, 'o-', 
            color=COLOR_UTILITY, markersize=6, linewidth=1.5, 
            markerfacecolor='white', markeredgewidth=1.5)
    
    # Add labels for each point
    # for i, label in enumerate(config_labels):
    #     ax.annotate(label.replace('Limit ', 'L').replace(' (Baseline)', ''), 
    #                (active_batteries[i], total_pv_capacities[i]),
    #                textcoords="offset points", xytext=(0, 8), 
    #                ha='center', fontsize=8)
    
    # Labels and formatting
    ax.set_xlabel('Number of Active Batteries')
    ax.set_ylabel('Total PV Capacity (kW)')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set x-axis to start from 0
    ax.set_xlim(left=-1)

    y_min = min(total_pv_capacities)
    y_max = max(total_pv_capacities)
    y_pad = max((y_max - y_min) * 0.1, 1.0)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_battery_states(feeder_name: str, config_folders: dict, save_dir: str = None):
    """Plot aggregated battery SOC and charging/discharging rates for each configuration.
    
    Creates separate plots for each configuration showing:
    - Aggregated SOC over time
    - Charging rate over time
    - Discharging rate over time
    
    Args:
        feeder_name: Name of the feeder
        config_folders: Dictionary mapping config names to folder names
        save_dir: Optional directory path to save the figures
    """
    # Load all results
    results_dict = {}
    for name, folder in config_folders.items():
        try:
            results = load_optimization_results(feeder_name, folder)
            results_dict[name] = results
        except Exception as e:
            print(f"Warning: Failed to load {folder}: {e}")
    
    if not results_dict:
        print("Error: No results loaded for plotting.")
        return
    
    # Filter configs with batteries
    configs_with_batteries = {k: v for k, v in results_dict.items() 
                             if v.get("zeta_b") is not None and np.sum(v.get("zeta_b") >= 0.5) > 0}
    
    if not configs_with_batteries:
        print("No configurations with active batteries found.")
        return
    
    # Apply IEEE style
    plt.rcParams.update(IEEE_STYLE)
    
    # Create separate plots for each configuration
    for config_name, results in sorted(configs_with_batteries.items()):
        # Extract battery data
        soc = results.get("soc")  # Shape: (Tb, num_battery)
        p_b = results.get("p_b")  # Net battery power (positive = discharge, negative = charge)
        zeta_b = results.get("zeta_b")
        
        if soc is None or p_b is None:
            print(f"Warning: Missing battery data for {config_name}")
            continue
        
        # Aggregate over all batteries (sum)
        soc_total = soc.sum(axis=1) if len(soc.shape) > 1 else soc
        p_b_total = p_b.sum(axis=1) if len(p_b.shape) > 1 else p_b
        
        # Separate charging (negative) and discharging (positive)
        p_charge = np.where(p_b_total < 0, -p_b_total, 0)  # Charging power (make positive)
        p_discharge = np.where(p_b_total > 0, p_b_total, 0)  # Discharging power
        
        # Convert timesteps to hours (30-min intervals = 0.5 hours)
        timesteps = np.arange(len(soc_total))
        time_hours = timesteps * 0.5
        
        # Create figure with 3 subplots
        fig, (ax_soc, ax_charge, ax_discharge) = plt.subplots(1, 3, figsize=(10.5, 2.5))
        
        # Plot 1: Aggregated SOC
        ax_soc.plot(time_hours, soc_total, color=COLOR_UTILITY, linewidth=1.5)
        ax_soc.set_xlabel('Time (hours)')
        ax_soc.set_ylabel('Total SOC (kWh)')
        ax_soc.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_soc.set_xlim(0, 24)
        ax_soc.set_xticks(np.arange(0, 25, 6))
        num_active = np.sum(zeta_b >= 0.5)
        ax_soc.set_title(f'State of Charge ({num_active} batteries)', fontsize=9)
        
        # Plot 2: Charging rate
        ax_charge.plot(time_hours, p_charge, color=COLOR_CUSTOMER, linewidth=1.5)
        ax_charge.set_xlabel('Time (hours)')
        ax_charge.set_ylabel('Charging Power (kW)')
        ax_charge.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_charge.set_xlim(0, 24)
        ax_charge.set_xticks(np.arange(0, 25, 6))
        ax_charge.set_title('Charging Power', fontsize=9)
        
        # Plot 3: Discharging rate
        ax_discharge.plot(time_hours, p_discharge, color='#d62728', linewidth=1.5)
        ax_discharge.set_xlabel('Time (hours)')
        ax_discharge.set_ylabel('Discharging Power (kW)')
        ax_discharge.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_discharge.set_xlim(0, 24)
        ax_discharge.set_xticks(np.arange(0, 25, 6))
        ax_discharge.set_title('Discharging Power', fontsize=9)
        
        # Overall title
        fig.suptitle(f'{config_name}', fontsize=10, y=1.02)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show
        if save_dir:
            # Create safe filename from config name
            safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            save_path = pathlib.Path(save_dir) / f"battery_states_{safe_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Battery states plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME

    # Matches the sub_folder names produced by scripts/run_battery_enhancement.py
    configs = {
        "Limit 10": "battery_12days_limit10",
        "Limit 20": "battery_12days_limit20",
    }
    
    # Print summary table
    print_summary_table(feeder_name, configs)
    
    # Create PV vs batteries plot
    save_path = pathlib.Path(__file__).parents[1] / "temp" / "plots" / feeder_name / "pv_vs_active_batteries.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pv_vs_active_batteries(feeder_name, configs, save_path=str(save_path))
    
    # Create battery states plots (one per configuration)
    battery_plots_dir = pathlib.Path(__file__).parents[1] / "temp" / "plots" / feeder_name
    plot_battery_states(feeder_name, configs, save_dir=str(battery_plots_dir))
