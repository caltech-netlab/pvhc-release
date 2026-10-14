"""The three-phase LinDistFlow model: solution and linear-constraint forms."""

import logging
import networkx as nx
import numpy as np
import scipy as sp

from .constants import BETA, BETA_H, I_9, branch_internal_pwr_relation, branch_pwr_relation, gamma_hat, gamma_hat_H
from .topology import build_node_data, check_network_validity, is_downward_oriented, subtree, to_edge_list
from .branches import build_permutation_matrix_open_split, component_model, validate_line_phase_structure

logger = logging.getLogger(__name__)

__all__ = [
    "branch_nodal_power",
    "three_phase_lindistflow",
    "three_phase_lindistflow_real",
    "three_phase_lindistflow_real_wrapper",
    "three_phase_lindistflow_linear_constraint",
]

def branch_nodal_power(graph, circuit_data, edge2idx, N, return_internal=False):
    """
    Construct the mapping matrix from nodal power injection to branch power flow.
    This implements the S-s solution equation in the paper.
    Args:
        graph (networkx.DiGraph): Directed graph representing the network topology.
        circuit_data (list or dict): List or dictionary containing branch data for each edge, including element types.
        edge2idx (dict): Mapping from edge tuples (i, j) to their corresponding indices in circuit_data.
        N (int): Number of nodes in the network.
    Returns:
        np.ndarray: A (N*9, N*9) complex matrix mapping nodal power injections to branch power flows.
    """
    def sort_edges_by_depth(graph):
        depth = {n: 0 for n in graph.nodes}
        root = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
        for node in nx.bfs_tree(graph, root).nodes:
            for child in graph.successors(node):
                depth[child] = depth[node] + 1
        return sorted(graph.edges, key=lambda e: depth[e[1]], reverse=True)

    C_inv = np.zeros((N, 9, N, 9), dtype=np.complex128)    # (edge by node)
    if return_internal: 
        C_inv_internal = np.zeros((N, 9, N, 9), dtype=np.complex128)  # similar to the above, but for internal power
    
    for (i, j) in sort_edges_by_depth(graph):
        e = edge2idx[(i, j)]
        G_9x9 = branch_pwr_relation[circuit_data[e]['element_type']]
        G_9x9_internal = branch_internal_pwr_relation[circuit_data[e]['element_type']]
        for k in graph.successors(j):
            e_out = edge2idx[(j, k)]
            # Add each child branch's power solution and multiply by G_9x9
            C_inv[e] += np.einsum('ab,bnj->anj', G_9x9, C_inv[e_out])
            if return_internal:
                C_inv_internal[e] += np.einsum('ab,bnj->anj', G_9x9_internal, C_inv[e_out])
        # Add the power injection at node j
        C_inv[e, :, j - 1, :] -= G_9x9
        if return_internal:
            C_inv_internal[e, :, j - 1, :] -= G_9x9_internal
    return (C_inv, C_inv_internal) if return_internal else C_inv


def three_phase_lindistflow(circuit_data, node_list=None, verbose=False, debug=False):
    """
    :param circuit_data: list of dicts, each dict is branch data, e.g.
        {
            'element_type': 'DY',
            'source': 'bus_1',
            'target': 'bus_2',
            'series_impedance': np.array with shape = (3, 3),
            'phases': tuple of int, e.g. (1, 2) for AB, (1, 0) for A only,
            'winding_connections': ['delta', 'wye'],  # only for transformers,
            'turns_ratio': float, only for transformers, default to 1.0 if not provided
                # Note that this is for the winding pairs (internal variables), N_pri / N_sec
        }
    :param node_list: list of node names, in the order they will be indexed in the output matrices. If None, will use the order from BFS traversal of the graph.
        Otherwise it must list every node once, starting at the root (ValueError if not).
    :return:
        C_0: np.ndarray, shape=(N*9, 9), maps root node voltage to nodal voltage references
        Z: np.ndarray, shape=(N*9, N*9), impedance like matrix maps nodal injection to nodal voltage drop
        Z_H: np.ndarray, shape=(N*9, N*9), similar to Z, but maps hermitian nodal injection to nodal voltage drop
        C_inv: np.ndarray, shape=(N*9, N*9), maps nodal injection to branch power flow
        node_list: list of node names, in the order they are indexed in the output matrices
        edge_list: list of tuples, each tuple is (source_node, target_node) in the order of the output matrices
    """
    # Build networkx graph from circuit_data, check network validity
    _, node_list_g, edge_list, root = check_network_validity(circuit_data)
    N = len(node_list_g) - 1    # Excluding the root node
    # Build node_list and assign node id
    node_list = node_list or node_list_g
    # The matrices index the root as node 0 and leave it out of their rows.
    if len(node_list) != len(node_list_g) or set(node_list) != set(node_list_g) or node_list[0] != root:
        raise ValueError(f"node_list must list each of the {N + 1} nodes of the network once, starting at the root")
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    edge2idx = {edge: e for e, edge in enumerate(edge_list)}
    # Caution: graph.edges is generally in a differnt order as edge_list
    graph = nx.DiGraph(edge_list)
    # Pre-process circuit_data. Lines default to a unit turns ratio, but we must not
    # write that default into the caller's dicts, so rebind to a local list holding
    # shallow copies of only the branches that need the default. Branches that already
    # declare a turns ratio are reused by reference (no copying of series_impedance).
    circuit_data = [
        {**branch_data, 'turns_ratio': 1.0}
        if (branch_data['element_type'] == 'line') and ('turns_ratio' not in branch_data)
        else branch_data
        for branch_data in circuit_data
    ]
    for branch_data in circuit_data:
        validate_line_phase_structure(branch_data)
    
    # Step 1: Get component models
    if verbose:
        logger.debug("Building component models...")
    A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 = component_model(circuit_data)
    M_jk = np.stack([
        circuit_data[e]['turns_ratio']**(-2) * np.linalg.pinv(A_k_9x9[e]) 
        for e in range(N)
    ])      # M_jk = a_jk^{-2} pinv(A_k) for each branch, as in the paper.
    
    # Step 2: Form C_inv_tr (paper: \hat{C}_inv^tr), C_0
    C_inv_tr = np.zeros((N, 9, N, 9), dtype=np.complex128)
    for e in range(N):
        j, k = edge_list[e]
        C_inv_tr[k-1, :, e, :] = I_9
        # Down the subtree at k, each branch (p, i) after the one into p, whatever the node order
        for p, i in nx.dfs_edges(graph, k):
            l = edge2idx[(p, i)]
            C_inv_tr[i-1, :, e, :] = M_jk[l] @ A_j_9x9[l] @ C_inv_tr[p-1, :, e, :]
    # Use C_inv_tr to speed up computation for C_0 (dynamic programming)
    #  C_0 at node i is C_inv_tr at 0j to node i, multiplied by the branch relation at 0j
    C_0 = np.zeros((N, 9, 9), dtype=np.complex128)
    for j in graph.successors(0):
        e = edge2idx[(0, j)]
        for i in [j] + subtree(graph, j):
            C_0[i-1] = C_inv_tr[i-1, :, e, :] @ M_jk[e] @ A_j_9x9[e]
    C_inv_tr = - C_inv_tr.reshape(N*9, N*9)
    C_0 = C_0.reshape(N*9, 9)

    # Step 3: Form D_z, D_z_H, with shape=(N, 9, 9)
    D_z = M_jk @ B_9x9
    D_z_H = M_jk @ B_H_9x9
    
    # Step 4: Form C_inv
    C_inv, C_inv_internal = branch_nodal_power(graph, circuit_data, edge2idx, N, return_internal=True)
    
    # Assemble Z, Z_H. Use einsum to accelerate multiplication by block diagonal.
    Z = C_inv_tr @ np.einsum('iab,ibjc->iajc', D_z, C_inv_internal).reshape(N*9, N*9)
    Z_H = C_inv_tr @ np.einsum('iab,ibjc->iajc', D_z_H, C_inv_internal).reshape(N*9, N*9)
    C_inv = C_inv.reshape(N*9, N*9)

    # Note that edge_list is possibly in a different order as graph.edges
    edge_list_out = [(node_list[e[0]], node_list[e[1]]) for e in edge_list]

    if debug:
        return C_0, Z, Z_H, C_inv, node_list, edge_list_out, C_inv_tr, C_inv_internal.reshape(N*9, N*9), D_z, D_z_H
    else:
        return C_0, Z, Z_H, C_inv, node_list, edge_list_out


def three_phase_lindistflow_real(C_0, Z, Z_H, C_inv, N, node_list, node_data, verbose=False):
    """
    returns: 
        C_diag: np.ndarray, shape=(N*3, 9)
        R: np.ndarray, shape=(N*3, N*3)
        X: np.ndarray, shape=(N*3, N*3)
        C_inv_real: np.ndarray, shape=(N*3, N*3)
    """
    if verbose:
        logger.debug("Converting to real-valued LinDistFlow model...")
    # This acts as the O matrix
    diag_indices = np.array([0, 4, 8])
    diag_indices = np.concatenate([diag_indices + 9*n for n in range(N)])
    Z, Z_H = Z[diag_indices], Z_H[diag_indices]
    C_diag = C_0[diag_indices]
    assert (np.abs(C_diag.imag) < 1e-6).all(), "C_diag has imaginary parts."
    C_diag = C_diag.real
    # Apply BVA (or equivalent assumption for open delta) to create s-diag(s) relation
    gamma_list = []
    gamma_H_list = []
    for node in node_list[1:]:
        conn_trans = node_data[node]['connected_transformer']
        if (conn_trans is None) or (conn_trans['element_type'] in {'line', 'YY', 'YD', 'DY', 'DD', 'DY_alt', 'YD_alt', 'openD_openD', 'openY_openY'}):
            gamma_list.append(gamma_hat)
            gamma_H_list.append(gamma_hat_H)
        elif conn_trans['element_type'] in {'openY_openD_split', 'openD_openD_split'}:
            _, permutation_sec = build_permutation_matrix_open_split(conn_trans)
            perm_kron = np.kron(permutation_sec, permutation_sec)
            gamma_list.append(perm_kron.T @ BETA @ permutation_sec)
            gamma_H_list.append(perm_kron.T @ BETA_H @ permutation_sec)
        else:
            raise ValueError(f"Unknown transformer type: {conn_trans}")
    gamma_tilde = sp.linalg.block_diag(*tuple(gamma_list))
    gamma_tilde_H = sp.linalg.block_diag(*tuple(gamma_H_list))
    R = np.real(Z @ gamma_tilde + Z_H @ gamma_tilde_H)
    X = np.imag(-Z @ gamma_tilde + Z_H @ gamma_tilde_H)
    C_inv_real = C_inv[diag_indices] @ gamma_tilde
    
    return C_diag, R, X, C_inv_real


def three_phase_lindistflow_real_wrapper(circuit_data, node_data=None, node_list=None, verbose=False):
    C_0, Z, Z_H, C_inv, node_list, edge_list = \
        three_phase_lindistflow(circuit_data, node_list=node_list, verbose=verbose)
    if node_data is None:
        _, node_data = build_node_data(circuit_data=circuit_data, node_list=node_list)
    C_diag, R, X, C_inv_real = \
        three_phase_lindistflow_real(C_0, Z, Z_H, C_inv, len(node_list)-1, node_list, node_data, verbose=verbose)
    return C_diag, R, X, C_inv_real, node_list, edge_list


def three_phase_lindistflow_linear_constraint(circuit_data, node_list=None):
    """
    C_T @ v = D_z @ S + D_z_H @ S^H
    Produce the parameter matrices C_T, D_z, D_z_H.
    Note the comparing with three_phase_lindistflow, 
    the C_T matrix is on the LHS of the equation here.
    This avoids the noninvertibility of A_k matrices.
    Note that here the D_z and D_z_H are different from the D_z, D_z_H in three_phase_lindistflow, where here we don't multiply by pseudoinverse.
    
    returns:
        C_T: np.ndarray, shape=(N*9, N*9)
        D_z: np.ndarray, shape=(N*9, N*9)
        D_z_H: np.ndarray, shape=(N*9, N*9)
        node_list: list, the list of node names
        edge_list: list, the list of edge tuples
    """
    # Build networkx graph from circuit_data, check network validity
    edge_list = to_edge_list(circuit_data)
    graph = nx.DiGraph(edge_list)
    root_node = [n for n in graph.nodes if graph.in_degree(n) == 0]
    assert len(root_node) == 1, f"There should be only one root node: {root_node}"
    assert nx.is_directed_acyclic_graph(graph), "Graph contains a cycle. Only radial networks are supported."
    assert is_downward_oriented(graph, root_node[0]), "Graph is not downward oriented from the root node."
    N = len(graph.nodes) - 1    # Excluding the root node
    assert len(graph.edges) == N, len(graph.edges)
    # Build node_list and assign node id
    if node_list:
        assert set(node_list) == set(graph.nodes), f"Node list {node_list} does not match graph nodes {graph.nodes}"
    else:
        node_list = node_list or list(nx.bfs_tree(graph, root_node[0]).nodes())
    assert node_list[0] == root_node[0], f"Root node {node_list[0]} does not match {root_node[0]}"
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    # Caution: graph.edges is generally in a differnt order as edge_list
    graph = nx.DiGraph(edge_list)
    # Pre-process circuit_data. Lines default to a unit turns ratio, but we must not
    # write that default into the caller's dicts, so rebind to a local list holding
    # shallow copies of only the branches that need the default. Branches that already
    # declare a turns ratio are reused by reference (no copying of series_impedance).
    circuit_data = [
        {**branch_data, 'turns_ratio': 1.0}
        if (branch_data['element_type'] == 'line') and ('turns_ratio' not in branch_data)
        else branch_data
        for branch_data in circuit_data
    ]
    for branch_data in circuit_data:
        validate_line_phase_structure(branch_data)
    
    # Iterate over edges: get component models and assemble
    A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 = component_model(circuit_data)
    C_T = np.zeros((N*9, (N+1)*9), dtype=np.complex128)
    D_z = np.zeros((N*9, N*9), dtype=np.complex128)
    D_z_H = np.zeros((N*9, N*9), dtype=np.complex128)
    for e, (j, k) in enumerate(edge_list):
        a_jk = circuit_data[e]['turns_ratio']
        C_T[e*9:(e+1)*9, j*9:(j+1)*9] = A_j_9x9[e]
        C_T[e*9:(e+1)*9, k*9:(k+1)*9] = - (a_jk ** 2) * A_k_9x9[e]
        D_z[e*9:(e+1)*9, e*9:(e+1)*9] = B_9x9[e]
        D_z_H[e*9:(e+1)*9, e*9:(e+1)*9] = B_H_9x9[e]

    edge2idx = {edge: e for e, edge in enumerate(edge_list)}
    C_inv, C_inv_internal = branch_nodal_power(graph, circuit_data, edge2idx, N, return_internal=True)
    C_inv, C_inv_internal = C_inv.reshape(N*9, N*9), C_inv_internal.reshape(N*9, N*9)

    return C_T, D_z, D_z_H, C_inv, C_inv_internal, node_list, edge_list
