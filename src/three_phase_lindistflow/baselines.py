"""Reference implementations of prior LinDistFlow models, for comparison."""

import logging
import networkx as nx
import numpy as np
import scipy as sp

from .constants import gamma
from .topology import check_network_validity
from .branches import convert_series_impedance_3x3, to_single_phase_series_impedance, validate_line_phase_structure
from .voltages import compute_nominal_voltages_from_turns_ratio

logger = logging.getLogger(__name__)

__all__ = [
    "hxu_3pldf_YY_only",
    "single_phase_lindistflow",
]

def hxu_3pldf_YY_only(circuit_data, node_list=None, return_additional=False):
    """
    For usage, p, q are defined as nodal injection (positive as into the node)
    returns:
        C_tilde_0: np.ndarray, shape=(N*3, 3)
        R: np.ndarray, shape=(N*3, N*3)
        X: np.ndarray, shape=(N*3, N*3)
        C_inv_kron: np.ndarray, shape=(N*3, N*3)
        node_list: list, the list of node names
        edge_list: list, the list of edge tuples
    """
    # Build networkx graph from circuit_data
    _, node_list_g, edge_list, _ = check_network_validity(circuit_data)
    node_list = node_list or node_list_g
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    edge2idx = {edge: e for e, edge in enumerate(edge_list)}
    G = nx.DiGraph(edge_list)
    # Incidence matrix (note that networkx has the opposite of canonical sign convention)
    node_ind = [node2idx[n] for n in node_list]
    C_matrix = -nx.incidence_matrix(G, oriented=True, nodelist=node_ind, edgelist=edge_list).toarray()
    c_0 = C_matrix[0]
    C_hat = C_matrix[1:, :]
    C_inv = np.linalg.inv(C_hat)

    # Diagonal impedance, turns ratio matrix
    D_z, T = [], []
    for e in edge_list:
        branch_data = circuit_data[edge2idx[e]]
        validate_line_phase_structure(branch_data)
        if 'series_impedance' in branch_data:
            # Convert to 3x3 matrix
            z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
        else:
            z_ij = np.zeros((3, 3), dtype=np.complex128)
            logger.warning("No series impedance found for branch: %s", e)
        if 'turns_ratio' in branch_data:
            # Map all transformers to YY equivalent.
            # Adjust the turns ratio and z_ij by 3 where appropriate.
            if branch_data['element_type'] in ['DY', 'DY_alt']:
                T.append((branch_data['turns_ratio'] / np.sqrt(3)) ** 2)
                z_ij /= 3
            elif branch_data['element_type'] in ['DD', 'DD_alt']:
                T.append(branch_data['turns_ratio'] ** 2)
                z_ij /= 3
            elif branch_data['element_type'] in ['YD', 'YD_alt']:
                T.append((branch_data['turns_ratio'] * np.sqrt(3)) ** 2)
            else:
                T.append(branch_data['turns_ratio'] ** 2)
        elif branch_data['element_type'] == 'line':
            T.append(1.0)
        else:
            raise ValueError("Unknown element {} has no turns ratio specified".format(branch_data['element_type']))
        D_z.append(z_ij * gamma.T)
    D_z = sp.linalg.block_diag(*D_z)
    D_r, D_x = np.real(D_z), np.imag(D_z)
    T = np.diag(T)
    # Matrix that map branch voltage drops to nodal voltages
    C_n, C_p = np.maximum(-C_hat.T, 0), np.maximum(C_hat.T, 0)
    C_T_T = C_p - T @ C_n
    C_T_inv_T_kron = np.kron(np.linalg.inv(C_T_T), np.eye(3))
    C_inv_kron = np.kron(C_inv, np.eye(3))
    # Assemble the final equation. These two lines are slow.
    R = 2 * C_T_inv_T_kron @ D_r @ C_inv_kron
    X = 2 * C_T_inv_T_kron @ D_x @ C_inv_kron
    # Matrix for mapping the v_0 voltage reference of root node
    c_0 = np.kron(c_0, np.eye(3)).T
    C_tilde_0 = - C_T_inv_T_kron @ c_0
    
    edge_list_out = [(node_list[e[0]], node_list[e[1]]) for e in edge_list]
    if return_additional:
        return C_tilde_0, R, X, C_inv_kron, node_list, edge_list_out, C_T_inv_T_kron, D_r, D_x, c_0
    else:
        return C_tilde_0, R, X, C_inv_kron, node_list, edge_list_out


def single_phase_lindistflow(circuit_data, per_unit=True, s_base=1, root_node_voltage=None, node_list=None):
    """
    An implementation of the single-phase LinDistFlow is provided for reference.
    Note that the R, X values are AVERAGED (not SUMMED) across the three phases.
    This means you should average the p, q injections as well, not sum.
    :param circuit_data: see three_phase_lindistflow function
    :param per_unit: bool, default True. If True, R, X and C_tilde_0 are returned in
        per-unit of the computed voltage base, and a voltage base is required: pass
        root_node_voltage (line-to-neutral volts at the root), or supply a
        'nominal_voltage' key on the first downstream branch. If False, everything is in
        SI units (ohms, volts^2) and no voltage base is needed -- this is the path the
        test suite and the experiment scripts use.
    :param s_base: float, base power for per-unit conversion
    return: 
    C_tilde_0: np.ndarray, shape=(N)
    R: np.ndarray, shape=(N, N)
    X: np.ndarray, shape=(N, N)
    C_matrix: np.ndarray, shape=(N+1, M), the incidence matrix of the graph
    """
    graph, node_list_g, edge_list, root = check_network_validity(circuit_data)
    node_list = node_list or node_list_g
    N = len(node_list) - 1
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    G = nx.DiGraph(edge_list)
    # Incidence matrix (note that networkx has the opposite of canonical sign convention)
    node_ind = [node2idx[n] for n in node_list]
    C_matrix = -nx.incidence_matrix(G, oriented=True, nodelist=node_ind, edgelist=edge_list).toarray()
    c_0 = C_matrix[0]
    C_hat = C_matrix[1:, :]
    C_inv = np.linalg.inv(C_hat)

    resistances, reactances, V_base, Z_base = \
        np.zeros(N), np.zeros(N), np.zeros(N+1), np.zeros(N)
    if per_unit:
        circuit_dict = {(e['source'], e['target']): e for e in circuit_data}
        if not root_node_voltage:
            # Fall back to a nominal voltage carried on the first downstream branch, if the
            # caller supplied one. circuit_data produced by DataLoader.to_circuit_data does
            # NOT carry 'nominal_voltage', so for parsed feeders this is a hard error rather
            # than a usable default.
            first_branch = circuit_dict[(root, list(graph.successors(root))[0])]
            if 'nominal_voltage' not in first_branch:
                raise ValueError(
                    "single_phase_lindistflow(per_unit=True) needs a voltage base. Pass "
                    "root_node_voltage=<line-to-neutral volts at the root>, or pass "
                    "per_unit=False to work in SI units. (Branch data from "
                    "DataLoader.to_circuit_data does not carry a 'nominal_voltage' key.)"
                )
            root_node_voltage = first_branch['nominal_voltage'][0]
        nominal_voltages = compute_nominal_voltages_from_turns_ratio(
            graph, node_list, root_node_voltage, circuit_dict, node2idx, single_phase=True
        )
    T = np.ones(N)
    for e, (i, j) in enumerate(edge_list):
        edge = circuit_data[e]
        validate_line_phase_structure(edge)
        # Take the equivalent positive sequence impedance, which is 
        # different from just the average of the diagonal entries.
        # Discard the interaction due to negative and zero sequence components.
        z_ij = to_single_phase_series_impedance(edge)
        resistances[e], reactances[e] = np.real(z_ij), np.imag(z_ij)
        if 'turns_ratio' in edge:
            if edge['element_type'] in ['DY', 'DY_alt']:
                T[e] = (edge['turns_ratio'] / np.sqrt(3)) ** 2
                resistances[e] /= 3
                reactances[e] /= 3
            elif edge['element_type'] in ['YD', 'YD_alt']:
                T[e] = (edge['turns_ratio'] * np.sqrt(3)) ** 2
            elif edge['element_type'] == 'DD':
                # As for DY, the impedance is per delta winding; its wye equivalent is a third.
                T[e] = edge['turns_ratio'] ** 2
                resistances[e] /= 3
                reactances[e] /= 3
            else:
                T[e] = edge['turns_ratio'] ** 2
        if per_unit:
            V_base[i] = nominal_voltages[i]
            V_base[j] = nominal_voltages[j]
            Z_base[e] = nominal_voltages[i] ** 2 / s_base
    T = np.diag(T)
    if per_unit:
        C_T_inv_T = C_inv.T @ np.diag(Z_base ** -1)
    else:
        C_n, C_p = np.maximum(-C_hat.T, 0), np.maximum(C_hat.T, 0)
        C_T_T = C_p - T @ C_n
        C_T_inv_T = np.linalg.inv(C_T_T)
        # Alternatively
        # C_T_inv_T = np.diag(V_base[1:] ** 2) @ C_inv.T @ np.diag(Z_base ** -1)

    R_matrix = C_T_inv_T @ np.diag(resistances) @ C_inv
    X_matrix = C_T_inv_T @ np.diag(reactances) @ C_inv
    if not (R_matrix >= 0).all():
        logger.warning("Some entries in R_matrix are negative.")
    if not (X_matrix >= 0).all():
        logger.warning("Some entries in X_matrix are negative.")
        
    C_tilde_0 = - C_T_inv_T @ c_0
    
    edge_list = [(node_list[e[0]], node_list[e[1]]) for e in edge_list]

    return C_tilde_0, R_matrix, X_matrix, C_inv, node_list, V_base, edge_list
