"""Load-profile construction and imbalance metrics.

These helpers sit above the models: they turn a parsed DSS feeder into nodal
injections, and summarize how unbalanced a set of three-phase phasors is. They
live in the package (rather than alongside the experiment scripts) because the
test suite and any downstream user needs them, not just the paper experiments.
"""

import logging
from pathlib import Path

import numpy as np

from .constants import pos_seq
from .dss_parser import DataLoader
from .topology import zero_sequence_islands

logger = logging.getLogger(__name__)

__all__ = [
    "average_imbalance",
    "average_zero_sequence_component",
    "build_imbalanced_load_profile",
    "remove_zero_sequence_load_current",
    "load_dss_feeder",
]


def average_imbalance(phase_vectors, normalize=False):
    """
    Return the average magnitude of the component orthogonal to the balanced sequence reference.
    The imbalance percent is calculated as the following:
        e.g. for standard orthonormal vectors p, n, and input
            phase_vector = np.sqrt(0.2) * p + np.sqrt(0.8) * n
        yields imbalance = sqrt(0.8) 
    """
    phase_vectors = np.asarray(phase_vectors, dtype=complex)
    positive_sequence = pos_seq / np.sqrt(3.0)
    positive_projection = (phase_vectors @ np.conj(positive_sequence))[:, None] * positive_sequence
    residual = phase_vectors - positive_projection
    residual_norm = np.linalg.norm(residual, axis=1)
    phase_vector_norm = np.linalg.norm(phase_vectors, axis=1)
    if normalize:
        weights = phase_vector_norm
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            avg_residual = 0.0
            avg_residual_pct = 0.0
        else:
            avg_residual = float(np.average(residual_norm, weights=weights))
            avg_residual_pct = float(np.sum(residual_norm) / weight_sum * 100.0)
    else:
        avg_residual = float(np.mean(residual_norm))
        avg_residual_pct = float(np.mean(residual_norm / np.maximum(1e-12, phase_vector_norm)) * 100.0)
    return avg_residual, avg_residual_pct


def average_zero_sequence_component(phase_vectors, normalize=False):
    """Return the average magnitude of the zero-sequence component."""
    phase_vectors = np.asarray(phase_vectors, dtype=complex)
    zero_sequence = np.array([1.0, 1.0, 1.0], dtype=complex) / np.sqrt(3.0)
    zero_sequence_projection = (phase_vectors @ np.conj(zero_sequence))[:, None] * zero_sequence
    zero_sequence_norm = np.linalg.norm(zero_sequence_projection, axis=1)
    phase_vector_norm = np.linalg.norm(phase_vectors, axis=1)
    if normalize:
        weights = phase_vector_norm
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            avg_zero_sequence = 0.0
            avg_zero_sequence_pct = 0.0
        else:
            avg_zero_sequence = float(np.average(zero_sequence_norm, weights=weights))
            avg_zero_sequence_pct = float(np.sum(zero_sequence_norm) / weight_sum * 100.0)
    else:
        avg_zero_sequence = float(np.mean(zero_sequence_norm))
        avg_zero_sequence_pct = float(np.mean(zero_sequence_norm / np.maximum(1e-12, phase_vector_norm)) * 100.0)
    return avg_zero_sequence, avg_zero_sequence_pct


def build_imbalanced_load_profile(dss_data, node_list, phase_weights):
    """
    Build nodal phase injections from parsed DSS loads and intentionally skew them by phase.
    The DSS load table provides the starting active/reactive power at each bus, and the
    returned profile redistributes that power so phase A is loaded more heavily than B/C.
    
    phase_weights: Skew the load power to different phases. With [1, 1, 1] every load
        keeps its DSS kW/kvar.
    returns p_phase_w (N-1,3), q_phase_var (N-1,3) where N = len(node_list) and the root node is excluded.
    """
    phase_weights = np.asarray(phase_weights, dtype=float)
    node_index = {str(node).lower(): idx for idx, node in enumerate(node_list[1:], start=0)}
    p_phase_w = np.zeros((len(node_list) - 1, 3), dtype=float)
    q_phase_var = np.zeros((len(node_list) - 1, 3), dtype=float)

    for load in dss_data.get("loads", {}).values():
        bus_key = str(load["bus"]).lower()
        node_idx = node_index.get(bus_key)
        if node_idx is None:
            continue

        phases = [p for p in load.get("phases", ()) if p in (1, 2, 3)]
        if not phases:
            # A bus given without node suffixes (e.g. "Bus1=860 Phases=3") connects to
            # nodes 1..Phases, as in OpenDSS.
            phases = list(range(1, int(load.get("num_phases", 0)) + 1))

        kw = float(load.get("kw", 0.0))
        kvar = float(load.get("kvar", 0.0))

        # Per-phase share of this DSS load, skewed by phase_weights:
        # - Three-phase loads: kW/kvar is the total over the three phases (OpenDSS
        #   convention), so each phase takes a third of it.
        # - Single-phase loads: all of it on their phase.
        # - Phase-to-phase loads: approximated as +/- injections on the two phases.
        if len(phases) == 3:
            weights = phase_weights / 3.0
        elif len(phases) == 1:
            weights = np.zeros(3, dtype=float)
            weights[phases[0] - 1] = phase_weights[phases[0] - 1]
        elif len(phases) == 2:
            weights = np.zeros(3, dtype=float)
            weights[phases[0] - 1] = phase_weights[phases[0] - 1]
            weights[phases[1] - 1] = -phase_weights[phases[0] - 1]
            logger.warning("Phase-to-phase load %s is approximated as two single-phase injections",
                           load['name'])
        else:
            logger.warning("Load %s must be specified as single-phase here; setting its injection to zero",
                           load['name'])
            weights = np.zeros(3, dtype=float)
            
        kw_phase = kw * weights * 1e3
        kvar_phase = kvar * weights * 1e3
        p_phase_w[node_idx, :] -= kw_phase
        q_phase_var[node_idx, :] -= kvar_phase

        # Mutate dss_data so downstream OpenDSS solve sees the changed totals.
        # Store per-phase components and update the load's aggregate kw/kvar fields.
        load['kw_phase'] = kw_phase.tolist()
        load['kvar_phase'] = kvar_phase.tolist()
        load['kw'] = float(np.sum(kw_phase))
        load['kvar'] = float(np.sum(kvar_phase))

    return p_phase_w, q_phase_var


def remove_zero_sequence_load_current(dss_data, circuit_data, node_list, p_phase_w, q_phase_var):
    """Make the loads that no zero-sequence current can reach draw none, at balanced voltages.

    Below a delta winding or a floating wye neutral (see zero_sequence_islands), nothing can
    supply the zero-sequence part of the currents the loads draw. Skewed per phase by
    build_imbalanced_load_profile, a three-phase load there would draw some: the power flow
    would drop it, and the ground truth fitted to the network admittance would then have to
    distort the loads to make up for it. So each such load gets the per-phase powers nearest
    to its own whose currents at balanced voltages have no zero-sequence part::

        s' = s - a (a^H s) / 3,   a = [1, alpha, alpha^2]

    This keeps its total complex power: for a delta-connected load, these are the per-phase
    powers of a delta load of that total. Loads on fewer phases there raise, since without
    the other phases a load cannot draw its zero-sequence part from them; loads elsewhere
    are left as they are.

    Call it after build_imbalanced_load_profile, with what that returned for ``node_list``
    and the ``dss_data`` it annotated, whose loads' ``kw_phase``/``kvar_phase`` are updated
    too. Returns new ``p_phase_w``, ``q_phase_var`` arrays.
    """
    p_phase_w = np.array(p_phase_w, dtype=float)
    q_phase_var = np.array(q_phase_var, dtype=float)
    island_buses = {str(bus).lower() for island in zero_sequence_islands(circuit_data) for bus in island}
    node_index = {str(node).lower(): idx for idx, node in enumerate(node_list[1:], start=0)}
    for load in dss_data.get("loads", {}).values():
        bus_key = str(load["bus"]).lower()
        if bus_key not in island_buses or bus_key not in node_index:
            continue
        # As in build_imbalanced_load_profile: a bare bus connects nodes 1..num_phases.
        phases = [p for p in load.get("phases", ()) if p in (1, 2, 3)] \
            or list(range(1, int(load.get("num_phases", 0)) + 1))
        if len(phases) != 3:
            raise NotImplementedError(
                f"Load {load['name']} on phases {phases} is below a delta winding or a floating wye "
                "neutral, where only three-phase loads are supported")
        if "kw_phase" not in load:
            raise ValueError(f"Load {load['name']} has no per-phase powers: call build_imbalanced_load_profile first")
        # (In W and var, as build_imbalanced_load_profile stores them.)
        s = np.asarray(load["kw_phase"]) + 1j * np.asarray(load["kvar_phase"])
        s_new = s - pos_seq * (pos_seq.conj() @ s) / 3
        node_idx = node_index[bus_key]
        p_phase_w[node_idx] -= s_new.real - s.real
        q_phase_var[node_idx] -= s_new.imag - s.imag
        load["kw_phase"] = s_new.real.tolist()
        load["kvar_phase"] = s_new.imag.tolist()
    return p_phase_w, q_phase_var


def load_dss_feeder(feeder, feeder_root="source"):
    """Load DSS files and return the parsed feeder metadata plus circuit data."""
    feeder = Path(feeder)
    logger.info("Loading feeder from %s", feeder)
    dl = DataLoader(str(feeder))
    dss_data = dl.parse()
    circuit_data = dl.to_circuit_data(feeder_root=feeder_root, plot_circuit=False)

    return dss_data, circuit_data
