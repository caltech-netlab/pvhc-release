"""Nominal voltage propagation and voltage reference helpers."""

import logging
import numpy as np

from .constants import Gamma, gamma

logger = logging.getLogger(__name__)

__all__ = [
    "compute_nominal_voltages_from_turns_ratio",
    "compute_nominal_complex_voltages_from_turns_ratio",
    "compute_v0_ref",
]


def _turns_ratio(branch, source, target):
    """
    Turns ratio (a_jk) of a branch, for propagating nominal voltages downstream.

    A line has no windings, so a missing key means unity. A transformer without a
    turns ratio has no defined voltage base on its secondary side, and defaulting it
    to 1.0 would silently produce wrong nominal voltages (and hence a wrong per-unit
    base) for the whole subtree below it, so that case is an error.
    """
    if 'turns_ratio' in branch:
        return branch['turns_ratio']
    element_type = branch.get('element_type')
    if element_type == 'line':
        return 1.0
    raise ValueError(
        f"Branch {source!r} -> {target!r} of element_type {element_type!r} has no "
        "'turns_ratio'. A transformer's turns ratio sets the nominal voltage (and the "
        "per-unit base) of everything downstream of it and cannot be assumed to be 1.0; "
        "only element_type 'line' defaults to unity. DataLoader.to_circuit_data emits "
        "'turns_ratio' on every branch, so check for a hand-built circuit_data entry or "
        "a misspelled key."
    )


def compute_nominal_voltages_from_turns_ratio(graph, node_list, root_node_voltage, circuit_dict, node2idx, single_phase=False):
    """
    Traverse the tree down and compute the nominal voltages at each node from the turns ratio.
    For DY and YD transformer we divide and multiply the turns ratio by sqrt(3) respectively.
    """
    assert root_node_voltage, "Root node voltage must be provided."
    to_visit = list(graph.successors(node_list[0]))
    nominal_voltages = np.zeros((len(node_list), 3))
    nominal_voltages[0] = np.array([root_node_voltage] * 3)
    while to_visit:
        k = to_visit.pop(0)
        j = next(graph.predecessors(k))
        a_jk = _turns_ratio(circuit_dict[(j, k)], j, k)
        if circuit_dict[(j, k)]['element_type'] in ('DY', 'DY_alt'):
            a_jk /= np.sqrt(3)
        elif circuit_dict[(j, k)]['element_type'] in ('YD', 'YD_alt'):
            a_jk *= np.sqrt(3)
        elif circuit_dict[(j, k)]['element_type'] in ('openD_openD_split', 'openD_openD', 'openY_openD_split',):
            if single_phase:
                logger.warning("single_phase=True but transformer %s->%s is split phase.", j, k)
            if 'split_phase' in circuit_dict[(j, k)] and circuit_dict[(j, k)]['split_phase']:
                a_jk = np.array([circuit_dict[(j, k)]['turns_ratio']] * 3)
                wp = list(circuit_dict[(j, k)]['winding_pairing'])
                if 'split_phase' in circuit_dict[(j, k)] and circuit_dict[(j, k)]['split_phase']:
                    (p_pri_sp, p_sec_sp) = circuit_dict[(j, k)]['split_phase']
                    for p in p_sec_sp:
                        a_jk[p - 1] *= 2
                    wp.remove((p_pri_sp, p_sec_sp))
                # This one is subtle. The phase C voltage needs to consider that the ground 
                # reference is now in the middle of AB leg. Draw a diagram and use trig to see why. 
                for _, p_sec in wp:
                    for p in p_sec:
                        if p not in p_sec_sp:
                            a_jk[p - 1] *= 2 / np.sqrt(3)
        nominal_voltages[node2idx[k]] = nominal_voltages[node2idx[j]] / a_jk
        to_visit += list(graph.successors(k))
    if single_phase:
        nominal_voltages = np.max(nominal_voltages, axis=1)
    return nominal_voltages


def compute_nominal_complex_voltages_from_turns_ratio(graph, node_list, root_node_voltage, circuit_dict, node2idx, single_phase=False):
    """
    Traverse the tree down and compute the nominal voltages at each node from the turns ratio.
    For DY and YD transformer we divide and multiply the turns ratio by sqrt(3) respectively.
    """
    assert root_node_voltage is not None, "Root node voltage must be provided."
    assert len(root_node_voltage) == 3, "Root node voltage must be a 3-element array for three phases."
    to_visit = list(graph.successors(node_list[0]))
    nominal_voltages = np.zeros((len(node_list), 3), dtype=np.complex128)
    nominal_voltages[0] = root_node_voltage
    while to_visit:
        k = to_visit.pop(0)
        j = next(graph.predecessors(k))
        a_jk = _turns_ratio(circuit_dict[(j, k)], j, k)
        if circuit_dict[(j, k)]['element_type'] in ('DY',):
            V_transform_matrix = Gamma
        elif circuit_dict[(j, k)]['element_type'] in ('DY_alt',):
            V_transform_matrix = Gamma.T
        elif circuit_dict[(j, k)]['element_type'] in ('YD',):
            V_transform_matrix = Gamma.T / 3
        elif circuit_dict[(j, k)]['element_type'] in ('YD_alt'):
            V_transform_matrix = Gamma / 3
        elif circuit_dict[(j, k)]['element_type'] in ('YY', 'DD', 'line', 'openY_openY', 'OpenD_OpenD'):
            V_transform_matrix = np.eye(3)
        else:
            raise ValueError(f"Un-implemented transformer type {circuit_dict[(j, k)]['element_type']} for complex voltage calculation.")
        nominal_voltages[node2idx[k]] = V_transform_matrix @ (nominal_voltages[node2idx[j]] / a_jk)
        to_visit += list(graph.successors(k))
    return nominal_voltages


def compute_v0_ref(V0):
    """
    Compute the squared complex voltage v0 in mathbb{S}^3 given the 
    voltage magnitudes |V0| in mathbb{R}^3. The voltage phase angles are
    assumed to be 120 degree apart.    
    :param V0: np.array, shape=(3,) or (T, 3). The voltage complex voltages.
    :return: 
        v_ref: np.array, shape=(3, 3) or (T, 3, 3), dtype=np.float
            The voltage reference vector, where we have the squared line-to-ground voltages where ref_selector is False
            and the squared line-to-line voltages where ref_selector is True.
    """
    # Shape and type check
    no_time_axis = False
    if len(V0.shape) == 1:
        assert V0.shape[0] == 3, V0.shape
        V0 = V0[np.newaxis, :]
        no_time_axis = True
    T = V0.shape[0]

    V0_abs = np.abs(V0)
    v0 = (V0_abs.reshape(T, 3, 1) * V0_abs.reshape(T, 1, 3)) * gamma
    if no_time_axis:
        v0 = v0[0]
    return v0
