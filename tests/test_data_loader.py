import numpy as np
import sys, pathlib
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.data_loader import DataLoader
import src.three_phase_lindistflow as ldf
from src.constants import FEEDERS, DEFAULT_FEEDER_NAME
from src.simulator import Simulator
np.set_printoptions(precision=3)
np.set_printoptions(threshold=10000)
np.set_printoptions(linewidth=170)

SOLAR_PROFILE_PATH = "data/example_feeder/solar_capacity_factor.csv"


def end2end_dataloader_test():
    """
    End-to-end test for the DataLoader class: compiles the example feeder's
    OpenDSS files, computes the LinDistFlow model, cross-checks the resulting
    nominal voltages against those computed independently from transformer
    turns ratios, and runs a short simulation.
    """
    cfg = FEEDERS[DEFAULT_FEEDER_NAME]
    feeder_name = DEFAULT_FEEDER_NAME
    feeder_root, v_root_nom, v_root_multiplier = \
        cfg["feeder_root"], cfg["v_root_nom"], cfg["v_root_multiplier"]
    data_loader = DataLoader(feeder_name)

    # Compile the OpenDSS files
    network_data = data_loader._dssParser()
    assert network_data, "Network data should not be empty"

    # Compute LinDistFlow system model parameters
    network_data = data_loader.get_circuit_model(feeder_root=feeder_root)
    node_list, edge_list, R, X, C_tilde_0, C_tilde_S = \
        (network_data[k] for k in ["node_list", "edge_list", "R", "X", "C_tilde_0", "C_tilde_S"])

    # Get bus nominal voltage (kV) from the LinDistFlow base-voltage term
    N = len(node_list) - 1
    v0_ref_squared = ldf.compute_v0_ref(v_root_nom * np.ones(3)).swapaxes(0, 1).reshape(9, 1)
    nominal_squared_voltages = (C_tilde_0.reshape(N, 3, 9) @ v0_ref_squared).reshape(N * 3)
    assert (np.imag(nominal_squared_voltages) < 1e-6).all(), \
        f"Nominal voltages should be real: {nominal_squared_voltages[np.imag(nominal_squared_voltages) >= 1e-6]}"
    nominal_squared_voltages = np.real(nominal_squared_voltages)
    nominal_voltages = np.sqrt(nominal_squared_voltages)

    # Check against nominal voltages computed independently from transformer turns ratio
    nominal_from_turns_ratio = data_loader.compute_nominal_voltages_from_turns_ratio(node_list, v_root_nom)
    if not np.allclose(nominal_voltages, nominal_from_turns_ratio, rtol=1e-3):
        diff = nominal_voltages - nominal_from_turns_ratio
        where = np.abs(diff) > 1e-4
        nodes_wrong_v = sorted(set(node_list[1 + (i // 3)] for i in np.where(where)[0]))
        raise AssertionError(
            f"{len(nodes_wrong_v)} node(s) have nominal voltage mismatch vs. turns-ratio "
            f"computation, e.g. {nodes_wrong_v[:5]}"
        )

    branch_pwr_ratings = network_data["branch_pwr_ratings"]
    print(f"branch_pwr_ratings: {len(branch_pwr_ratings)}")

    # Get time series (a short one-day window)
    time_array, pv_profiles, load_profiles, load_info, V0_profiles = data_loader.get_scenario_data_by_range(
        SOLAR_PROFILE_PATH,
        '2023-01-05 14:30:00',
        '2023-01-06 14:30:00',
        v_root_nom,
        v_root_multiplier,
    )
    print(f"T: {len(time_array)}, pv_profiles: {pv_profiles.shape}, load_profiles: {load_profiles[0].shape}, V0_profiles: {V0_profiles.shape}")
    assert len(time_array) > 0, "Expected at least one sampled timestep"

    # Get annual consumption estimate
    load_info = data_loader.get_annual_consumption(load_info)
    assert all("annual_consumption" in load for load in load_info)

    # Test Simulator end-to-end
    simulator = Simulator(data_loader)
    PV_info = [{**load, "capacity": 6} for load in load_info]  # Dummy PV info
    voltage_profiles, active_line_flows, reactive_line_flows, _, _, _ = simulator.simulate(
        load_profiles=load_profiles,
        pv_profiles=pv_profiles,
        V0_profiles=V0_profiles,
        load_info=load_info,
        PV_info=PV_info
    )
    print(f"voltage_profiles: {voltage_profiles.shape}, active_line_flows: {active_line_flows.shape}, reactive_line_flows: {reactive_line_flows.shape}")

    voltage_drop = (voltage_profiles - nominal_voltages) / nominal_voltages
    print(f"voltage (normalized) difference from nominal: max {np.max(voltage_drop)}, min {np.min(voltage_drop)} p.u.")
    assert np.max(np.abs(voltage_drop)) < 0.5, "Voltage deviation from nominal is implausibly large"


if __name__ == "__main__":
    end2end_dataloader_test()
