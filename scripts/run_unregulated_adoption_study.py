"""
Unregulated PV Adoption Study — Paper Section IV.B, Fig. 1 and Fig. 2

Simulates the effect of *unregulated* (uncoordinated) customer PV adoption on
feeder voltage/branch-loading violations, without any PVHC optimization: at
each adoption rate r, a random subset of customers each install a fixed
10 kW system independently with probability r, and the resulting network is
simulated over historical load/PV timeseries.

Fig. 1: sweeps r in {0%, 10%, ..., 100%}, 10 Monte Carlo installation draws
  per rate, and reports system violation probability and mean number of
  violating elements vs. adoption rate (`plot_figures.plot_sim_ablation`).
Fig. 2: spatiotemporal violation heatmaps at a single high-adoption rate
  (80% by default) (`plot_figures.plot_sim_365`).

Output: temp/simulation_results/<feeder>/ablation_unregulated/<sub_folder>/
"""
import copy
import pathlib
import sys

import numpy as np
import tqdm

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, DEFAULT_CANDIDATE_PV_CAPACITY_KW
from src.data_loader import DataLoader
from src.utils import SimulationConfig
from scripts.run_pvhc_simulation import run_simulation_and_store
from scripts.plot_figures import plot_sim_ablation, plot_sim_365


def run_adoption_ablation(
    feeder_name=DEFAULT_FEEDER_NAME,
    penetration_rates=np.linspace(0, 1, 11),
    sim_config=None,
    sub_folder_suffix="",
    installation_profiles=10,
    seed=42,
    skip_if_exists=True,
):
    """
    For each penetration rate r, draw `installation_profiles` independent random
    installation patterns (each customer installs DEFAULT_CANDIDATE_PV_CAPACITY_KW
    with probability r) and simulate the resulting network, storing results under
    temp/simulation_results/<feeder>/ablation_unregulated/.
    """
    cfg = FEEDERS[feeder_name]
    data_loader = DataLoader(feeder_name)
    network_data = data_loader.get_circuit_model(feeder_root=cfg["feeder_root"])
    customer_pv_info = copy.deepcopy(network_data['load_info'])
    for l in customer_pv_info:
        l['connected_transformer'] = data_loader.node_data[l['bus'].lower()]['connected_transformer']

    for r in tqdm.tqdm(penetration_rates, desc="Adoption rate"):
        for profile_idx in tqdm.tqdm(range(installation_profiles), desc="Installation draw", leave=False):
            np.random.seed(seed + profile_idx)
            capacities_isa_customer = np.array([
                DEFAULT_CANDIDATE_PV_CAPACITY_KW if np.random.rand() < r else 0.0
                for _ in range(len(customer_pv_info))
            ])
            sub_folder = "ablation_unregulated/" + sub_folder_suffix + f"_pr_{int(r*100)}_seed_{profile_idx}"
            sim_out_dir = pathlib.Path(__file__).resolve().parents[1] / "temp" / "simulation_results" / feeder_name / sub_folder
            if skip_if_exists and sim_out_dir.exists():
                continue
            run_simulation_and_store(
                feeder_name=feeder_name,
                sub_folder=sub_folder,
                utility_pv_info=[],
                customer_pv_info=customer_pv_info,
                capacities_isa_utility=np.array([]),
                capacities_isa_customer=capacities_isa_customer,
                config=sim_config,
            )


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME
    seed = 42
    ext = "png"
    HIGH_ADOPTION_RATE_PCT = 80  # for the Fig. 2 spatiotemporal heatmap

    np.random.seed(seed)

    print("Running unregulated adoption ablation (Fig. 1)...")
    run_adoption_ablation(
        feeder_name,
        penetration_rates=np.linspace(0, 1, 11),
        sim_config=SimulationConfig(mode="whole_year", days_sampled=48, equal_samples_per_month=True),
        seed=seed,
    )
    plot_sim_ablation(feeder_name, ext=ext)

    print(f"\nPlotting spatiotemporal violation heatmaps at {HIGH_ADOPTION_RATE_PCT}% adoption (Fig. 2)...")
    plot_sim_365(
        feeder_name,
        sim_subfolders=[f"ablation_unregulated/_pr_{HIGH_ADOPTION_RATE_PCT}_seed_0"],
        ext=ext,
    )
