"""
Power Flow Simulator — applies 3-phase LinDistFlow at each timestep

`Simulator` wraps `DataLoader` and provides two public methods:
  - `simulate()`: Given load profiles, PV profiles, V0 profiles, and PV/battery
    info, applies the real-valued LinDistFlow equations (from
    `three_phase_lindistflow_real()`) to compute per-timestep squared voltages
    and branch power flows.  Used by `run_pvhc_simulation.py` for Figs. 2-3.
  - `get_network_data()`: Returns precomputed LinDistFlow matrices (R̃, X̃, etc.)
    from `data_loader.get_circuit_model()`.
  - `_map_injections_to_buses()`: Maps per-device injection timeseries to a
    3N-dimensional nodal injection vector (used by `dual_analysis.py` to
    reconstruct p^(k) for Appendix B sensitivity computation).

The `simulate()` output (voltage_profiles, active_flows, reactive_flows) is
saved by `run_pvhc_simulation.py` and visualized by `plot_figures.py`.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from data_loader import DataLoader
import three_phase_lindistflow as ldf

from constants import *


class Simulator:
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the simulator with a DataLoader instance

        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
        self.feeder_name = data_loader.feeder_name

        # Auxiliary data for simulation
        self.R_tilde_load = None  # To be defined based on load models
        self.X_tilde_load = None  # To be defined based on load models

    def get_network_data(self) -> Dict:
        """Retrieve the LinDistFlow network data for this feeder."""
        return self.data_loader.get_circuit_model()

    def simulate(
        self,
        load_profiles: Tuple[np.ndarray, np.ndarray],
        pv_profiles: np.ndarray,
        V0_profiles: np.ndarray,
        load_info: List[Dict[str, Any]] = None,
        PV_info: List[Dict[str, Any]] = None,
        battery_profiles: Tuple[np.ndarray, np.ndarray] = None,
        battery_info: List[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run power flow simulation with direct profile input

        Args:
            load_profiles: (np.array([[]]),np.array([[]])) where the first array is active power and the second array is reactive power
            and the first idx is time index and the second idx is the bus index, and the unit is kW for active power and kVAR for reactive power
            pv_profiles: np.array([[]]) where the first idx is time index and the second idx is the bus index, you can also input a pv_profile that has shape (N_scenarios,) which represents that all PVs have the same per-unit generation profile, and the unit is kW per kW installed capacity,
            notice PV is assumed to have zero reactive power injection
            V0_profiles: np.ndarray shape(N_scenarios, 3) where the first idx is time index and the second idx is the bus index
            load_info: List of load information dictionaries, each entry is a dict with keys being "bus" and "phases"
            PV_info: List of dictionaries, each entry is a dict with keys being "bus" and "phases" and "capacity" in kW
            battery_profiles: Optional tuple (p_battery, q_battery) where each is np.array with shape (N_scenarios, n_batteries)
                p_battery: active power injection in kW (positive = discharging, negative = charging)
                q_battery: reactive power injection in kVAR
            battery_info: Optional list of dictionaries, each entry is a dict with keys being "bus" and "phases"

        Returns:
            Tuple of (voltage_profiles, active_line_flows, reactive_line_flows, p_inj, q_inj, injection_map)
        """
        # Get network data
        network_data = self.data_loader.get_circuit_model()

        # Determine number of scenarios from profile data
        N_scenarios = load_profiles[0].shape[0]
        assert N_scenarios == pv_profiles.shape[0], "Inconsistent number of scenarios"
        assert N_scenarios == load_profiles[1].shape[0], "Inconsistent number of scenarios"
        assert V0_profiles is None or V0_profiles.shape[0] == N_scenarios, "Inconsistent number of scenarios"

        assert V0_profiles is None or V0_profiles.shape[1] == 3, "V0_profiles must have shape (N_scenarios, 3)"
        assert pv_profiles.ndim == 1 or pv_profiles.shape[1] == network_data["n_buses"] or pv_profiles.shape[1] == 1, "pv_profiles must match the number of buses in the network or have shape (N_scenarios, 1)"

        assert load_profiles[0].shape[1] == load_profiles[1].shape[1], "Inconsistent number of loads between active and reactive power profiles"
        assert len(load_info) == load_profiles[0].shape[1], "Inconsistent number of loads between load_info and load_profiles"

        # Validate battery inputs
        if battery_profiles is not None:
            assert battery_info is not None, "battery_info must be provided when battery_profiles is provided"
            assert N_scenarios % battery_profiles[0].shape[0] == 0, "Inconsistent number of scenarios for battery profiles"
            assert battery_profiles[0].shape == battery_profiles[1].shape, "Inconsistent number of scenarios for battery profiles"
            N_days = N_scenarios // battery_profiles[0].shape[0]

        # Prepare injection map and capacities
        PV_installation_capacities = np.array([pv['capacity'] for pv in PV_info]) if PV_info else None
        injection_map = load_info.copy() if load_info else []
        injection_map += PV_info if PV_info else []
        injection_map += battery_info if battery_info else []
        
        # Pre-allocate all injection arrays
        n_loads = load_profiles[0].shape[1]
        n_pvs = len(PV_installation_capacities) if PV_installation_capacities is not None else 0
        n_batteries = len(battery_info) if battery_info else 0
        total_injections = n_loads + n_pvs + n_batteries
        p_inj = np.zeros((N_scenarios, total_injections), dtype=np.float64, order='C')
        q_inj = np.zeros((N_scenarios, total_injections), dtype=np.float64, order='C')
        
        # Fill in load data
        p_inj[:, :n_loads] = load_profiles[0]
        q_inj[:, :n_loads] = load_profiles[1]
        
        # Fill in PV data (vectorized)
        if PV_installation_capacities is not None:
            pv_injection = pv_profiles[:, np.newaxis] * PV_installation_capacities[np.newaxis, :]  # Broadcasting
            p_inj[:, n_loads:n_loads + n_pvs] = pv_injection
            # q_inj for PV remains zero (already initialized)
        
        # Fill in battery data (vectorized)
        if battery_profiles is not None and n_batteries > 0:
            # TODO: This assumes that the scenarios are sampled whole days
            p_inj[:, n_loads + n_pvs:] = np.tile(battery_profiles[0], (N_days, 1))
            q_inj[:, n_loads + n_pvs:] = np.tile(battery_profiles[1], (N_days, 1))
        
        # Vectorized power flow solve for all scenarios at once
        voltage_profiles, active_line_flows, reactive_line_flows = self._solve_lin_dist_flow(
            network_data, V0_profiles, p_inj, q_inj, injection_map
        )

        return voltage_profiles, active_line_flows, reactive_line_flows, p_inj, q_inj, injection_map

    def _map_injections_to_buses(
        self, 
        p_injection: np.ndarray, 
        q_injection: np.ndarray, 
        injection_map: List[Dict[str, Any]], 
        node2idx: Dict[str, int], 
        T: int, 
        N: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map the injections to the correct bus indices based on the injection map and node2idx"""
        # Pre-allocate arrays for wye injections (for vectorized matrix multiplication)
        p_wye_mapped = np.zeros((T, 3*N), dtype=np.float64)
        q_wye_mapped = np.zeros((T, 3*N), dtype=np.float64)
        complex_injection_mapped = np.zeros((T, 3*N), dtype=np.complex128)
        
        # Process each injection and accumulate into mapped arrays
        for i, injection_data in enumerate(injection_map):
            bus_name, phases = injection_data['bus'], injection_data['phases'] 
            try: 
                bus_idx = node2idx[bus_name]
            except KeyError:
                raise KeyError(f"Bus '{bus_name}' not found in node2idx")
            assert phases[0] != 0, f"Invalid phase ordering: {phases}, injection_data: {injection_data}"
            
            # Wye-connected (phase-to-neutral) load
            if phases[1] == 0: 
                b_idx = (bus_idx-1) * 3 + phases[0] - 1
                # Accumulate injections into the mapped arrays
                p_wye_mapped[:, b_idx] += p_injection[:, i]
                q_wye_mapped[:, b_idx] += q_injection[:, i]
                complex_injection_mapped[:, b_idx] += p_injection[:, i] + 1j * q_injection[:, i]
            
            # Delta-connected (phase-to-phase) load
            else: 
                phase_idx = ldf.STD_DELTA_ORDER.index(tuple(sorted(phases)))
                conn_trans = injection_data['connected_transformer']
                
                # Determine transformation vector based on transformer type
                if conn_trans['element_type'] in {'line', 'YY', 'DD', 'YD', 'DY', 'YD_alt', 'DY_alt', 'openD_openD', 'openY_openY'}:
                    transformation_vec = ldf.D2Y[:, phase_idx]
                elif conn_trans['element_type'] in {'openY_openD_split', 'openD_openD_split'}:
                    assert conn_trans['split_phase'] in conn_trans['winding_pairing'], conn_trans
                    _, permutation = ldf.build_permutation_matrix_open_split(conn_trans)
                    transformation_vec = (permutation.T @ ldf.openD_split2Y @ permutation)[:, phase_idx]
                else:
                    raise ValueError(f"Unknown transformer type for delta load: {conn_trans['element_type']}")
                
                # Transform delta to wye
                Y_injection = (p_injection[:, i:i+1] + 1j * q_injection[:, i:i+1]) * transformation_vec  # (T, 1) * (3,) -> (T, 3)
                p_wye_injection = np.real(Y_injection)
                q_wye_injection = np.imag(Y_injection)
                
                # Accumulate into mapped arrays
                b_idx = (bus_idx-1) * 3
                p_wye_mapped[:, b_idx:b_idx+3] += p_wye_injection
                q_wye_mapped[:, b_idx:b_idx+3] += q_wye_injection
                complex_injection_mapped[:, b_idx:b_idx+3] += Y_injection
        
        return p_wye_mapped, q_wye_mapped, complex_injection_mapped

    def _solve_lin_dist_flow(
        self,
        network_data: Dict,
        V0: np.ndarray,
        p_injection: np.ndarray,
        q_injection: np.ndarray,
        injection_map: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve linearized DistFlow equations (paper Eq. 4-5).
        Args:
            network_data: see get_circuit_model
            V0: np.ndarray shape=(3,) or (T, 3) dtype=complex, unit: kV
            p_injection: np.ndarray, shape=(#loads) or (T,#loads), unit: kW
            q_injection: np.ndarray, shape=(#loads) or (T,#loads), unit: kVAR
            injection_map: List[Dict[str, Any]]
        Returns:
            Tuple of (voltages, active line power flows, reactive line power flows) 
            whose units are respectively (kV, kW, kVAR)
        """
        # Extract network data
        N = network_data["n_buses"]
        C_tilde_0 = network_data["C_tilde_0"]
        C_tilde_S = network_data["C_tilde_S"]
        R = network_data["R"]       # shape = (N*3， N*3)
        X = network_data["X"]       # shape = (N*3， N*3)
        node2idx = network_data["node2idx"]

        # Ensure 2D arrays
        if p_injection.ndim == 1:
            p_injection = p_injection[np.newaxis, :]    # (T, #loads)
            q_injection = q_injection[np.newaxis, :]    # (T, #loads)
        if V0.ndim == 1:
            V0 = V0[np.newaxis, :]
        T = p_injection.shape[0]
        num_loads = p_injection.shape[1]
        assert q_injection.shape[0] == T and q_injection.shape[1] == num_loads, "Inconsistent shape for p_injection and q_injection"
        assert V0.shape == (T, 3), f"Expected V0 shape {(T, 3)}, but got {V0.shape}"
        assert R.shape == X.shape == (N*3, N*3), (R.shape, X.shape)
        assert len(injection_map) == num_loads

        # Compute the nominal voltages (v0 reference) for each bus
        v0_ref_squared = ldf.compute_v0_ref(V0).swapaxes(1, 2).reshape(T, 1, 9, 1)
        v_hat_squared = (C_tilde_0.reshape(1, N, 3, 9) @ v0_ref_squared).reshape(T, N*3) # Base voltage term
        v_hat_squared = np.real(v_hat_squared).astype(float)
        
        # Map injections to bus indices (vectorized)
        p_wye_mapped, q_wye_mapped, complex_injection_mapped = self._map_injections_to_buses(
            p_injection, q_injection, injection_map, node2idx, T, N
        )
        
        # Vectorized voltage drop computation (single matrix multiplication for all timesteps)
        # First convert p, q from kW to W (*1e3), then convert V^2 to kV^2 (*1e-6) so net *1e-3
        v_hat_squared += 1e-3 * (p_wye_mapped @ R.T + q_wye_mapped @ X.T)
        
        # Vectorized branch power flow computation
        branch_complex_pwr = complex_injection_mapped @ C_tilde_S.T
        
        # Ensure voltage is real and non-negative
        assert v_hat_squared.dtype == float, f"v_hat_squared should be float, got {v_hat_squared.dtype}"
        assert (v_hat_squared >= 0).all(), f"Negative voltage found: {v_hat_squared[v_hat_squared < 0]}"
        v_hat = np.sqrt(v_hat_squared)
    
        # Calculate branch power flow (computation in complex numbers, return real numbers)
        P, Q = np.real(branch_complex_pwr), np.imag(branch_complex_pwr)

        return np.squeeze(v_hat), np.squeeze(P), np.squeeze(Q)

