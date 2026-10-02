"""
3-Phase Linear DistFlow (LinDistFlow) Model — Paper Section II.A, Eq. (1)-(6)

Implements the 3-phase linearized DistFlow equations for radial distribution
networks.  Given per-unit nodal power injections p_n and q_n, the model
computes squared voltage magnitudes v̂ and branch power flows P, Q via:

    v̂ = C̃_0 v_0 + R̃ p + X̃ q               (Eq. 4)
    [P; Q] = C̃_S [p; q]                       (Eq. 5)

where R̃, X̃ are built from the network incidence matrix and branch impedances,
and C̃_S maps nodal injections to branch power flows.

`three_phase_lindistflow()` (and its real-valued wrapper
`three_phase_lindistflow_real_wrapper()`) is the general derivation, supporting
DY, YD, DD, open-delta, and split-phase transformers. `compute_branch_sensitivity_matrices()`
separately exposes the per-branch impedance matrices needed by the Appendix B
dual-variable sensitivity analysis (see its docstring for the scope of
networks this applies to).

Key constants defined at module level:
  alpha  = exp(-j·2π/3)                  — 120° phase rotation (Eq. 3)
  gamma  = [[1, α², α], [α, 1, α²], …]  — 3-phase coupling matrix (Eq. 3)
  D2Y    — delta-to-wye transformation matrix (Appendix A)

Paper: "Regulated PV Capacity for 3-Phase Distribution System via
Robust-Scenario Program".
"""
import os
import numpy as np
import networkx as nx
import scipy as sp
from tqdm import tqdm

"""
Constants
"""
I_3 = np.eye(3, dtype=np.csingle)
I_9 = np.eye(9, dtype=np.csingle)
I_23 = np.eye(2,3, dtype=np.csingle)
Gamma = np.array([
    [1, -1, 0],
    [0, 1, -1],
    [-1, 0, 1]
], dtype=np.csingle)
# Phase rotation factor: α = e^{-j2π/3} (120° lag), Eq. (3) in paper
alpha = np.exp(-1j * 2/3 * np.pi)
# 3-phase coupling matrix γ used in unbalanced voltage equations, Eq. (3)
gamma = np.array([
    [1, alpha**2, alpha],
    [alpha, 1, alpha**2],
    [alpha**2, alpha, 1]
], dtype=np.csingle)
alpha_p = gamma[:, 0]
gamma_hat = np.array([
    [1,        0,       0],
    [alpha,    0,       0],
    [alpha**2, 0,       0],
    [0,        alpha**2,0],
    [0,        1,       0],
    [0,        alpha,   0],
    [0,        0,       alpha],
    [0,        0,       alpha**2],
    [0,        0,       1]
], dtype=np.csingle)
gamma_hat_H = np.array([
    [1,        0,       0],
    [0,        alpha,   0],
    [0,        0,       alpha**2],
    [alpha**2, 0,       0],
    [0,        1,       0],
    [0,        0,       alpha],
    [alpha,    0,       0],
    [0,        alpha**2,0],
    [0,        0,       1]
], dtype=np.csingle)
M = np.array([
    [        1, 0, 0,      0],
    [        0, 0, 1,      0],
    [alpha**2, 0, 0,      0],
    [        0, 1, 0,      0],
    [        0, 0, 0,      1],
    [        0, 0, 0,  alpha],
    [        0, 0, 0,      0],
    [        0, 0, 0,      0],
    [        0, 0, 0,      0],
], dtype=np.csingle)
Q_delta = np.array([
    [ 1, 0,  0],
    [ 0, 0, -1],
], dtype=np.csingle)
Q_deltasplit = np.array([
    [ 1, 0, 0],
    [ 0, 1, 1],
    [ 0, 0, -1],
], dtype=np.csingle)
Q_windingsplit = np.array([
    [ 1/2, -1/2,  0],
    [ 0, 0, 1],
], dtype=np.csingle)
U_delta = np.array([
    [ 1/2,  0],
    [-1/2,  0],
    [-1/2, -1],
], dtype=np.csingle)
U_delta_3x3 = np.array([
    [ 1/2,  0,    0],
    [-1/2,  0,    0],
    [-1/2, -1,    0],
], dtype=np.csingle)
T_delta = np.array([
    [ 1,  0],
    [ 0,  1],
    [-1, -1],
], dtype=np.csingle)
beta = np.array([
    [1],
    [-1],
    [np.sqrt(3)*1j]
], dtype=np.csingle)
beta_inv = np.array([
    [1],
    [-1],
    [-(1/np.sqrt(3))*1j]
], dtype=np.csingle)
BETA = np.kron(I_3, beta @ beta_inv.T)[:, [0, 4, 8]]   # (9, 3), maps diag(s_k) to vec(s_k)
BETA_H = np.kron((beta @ beta_inv.T).conj(), I_3)[:, [0, 4, 8]]  # (9, 3), maps diag(s_k^H) to vec(s_k^H)
# Transformation matrix that maps delta load [s(1,2), s(2,3), s(3,1)] to wye load [s(1), s(2), s(3)]
# This is equal to $\frac{1}{3} (\Gamma^\top \gamma) \circ \Gamma^\top$
D2Y = 1/(1-alpha)*np.array([[1, 0, -alpha], [-alpha, 1, 0], [0, -alpha, 1]])
# Matrix mapping delta load connected upstream to an open delta transformer to wye load 
openD_split2Y = np.kron(Gamma.T, U_delta_3x3 @ gamma)[[0, 4, 8]][:, [0, 4, 8]]
branch_pwr_relation = {
    'line': I_9,
    'YY': I_9,
    'DY': np.kron(Gamma.T, Gamma.T) / 3,
    'DY_alt': np.kron(Gamma, Gamma) / 3,
    'YD':  np.kron(Gamma, Gamma) / 3,
    'DD': I_9,
    'openD_openD_split': M @ np.kron(Q_windingsplit @ Q_deltasplit, Gamma[:2]),
    'openD_openD': I_9,
    'openD_openD_split': (1/3) * np.kron((Gamma[:2].T @ Q_windingsplit @ Q_deltasplit), (Gamma.T @ T_delta @ Gamma[:2]))
}
STD_DELTA_ORDER = [(1,2), (2,3), (1,3)]

"""
Helper functions
"""
def to_edge_list(edges, include_name=False):
    """
    :param edges: list of dict
    :param include_name: bool, whether to include edge attributes
    :return: list of tuple of str (src, tgt) where src/tgt are source/target node names
    """
    if include_name:
        return {(e["source"], e["target"]): e['element_name'] for e in edges}
    else:
        return [(e["source"], e["target"]) for e in edges]


def to_edge_dict(edges, flip=False):
    """
    :param edges: list of dict
    :param flip: bool, whether to flip value and keys
    :return: dict, key: edge name, value: tuple of str (src, tgt)
        where src/tgt are source/target node names. If flip is True,
        values become keys and keys become values.
    """
    return (
        {(e["source"], e["target"]): name for name, e in edges}
        if flip
        else {name: (e["source"], e["target"]) for name, e in edges}
    )


def edges_between_nodes(graph, source, target):
    """
    :param G: networkx.DiGraph
    :param source: str or int, source node name
    :param target: str or int, target node name
    :return: list of tuple of (src, tgt), which are edges in path
    """
    nodes_in_path = list(nx.all_simple_paths(graph, source=source, target=target))
    if not nodes_in_path:
        return []
    else:
        nodes_in_path = nodes_in_path[0]
    edges = list(zip(nodes_in_path[:-1], nodes_in_path[1:]))
    return edges


"""
Component models: voltage drop in terms of branch power
The generalized branch model (for internal variables) is:
A1 v_j A2  -  a_jk^2 A3 v_k A4  =  B1 S_jk B2  +  B2^H S_jk^H B1^H
"""
def line(z_jk: np.ndarray):
    # v_j - v_k = S_jk z_jk^H + z_jk S_jk^H
    A1, A2, A3, A4 = I_3, I_3, I_3, I_3
    B1, B2 = I_3, z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def YY_transformer(z_jk: np.ndarray):
    # v_j - a_jk^2 v_k = S_jk z_jk^H + z_jk S_jk^H
    A1, A2, A3, A4 = I_3, I_3, I_3, I_3
    B1, B2 = I_3, z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def DY_transformer(z_jk: np.ndarray):
    # Γ v_j Γ^T - a_jk^2 v_k = (1/3 Γ S_jk Γ^T) z_jk^H + z_jk (1/3 Γ S_jk^H Γ^T)
    A1, A2, A3, A4 = Gamma, Gamma.T, I_3, I_3
    B1, B2 = (1/3) * Gamma, Gamma.T @ z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def YD_transformer(z_jk: np.ndarray):
    # v_j - a_jk^2 Γ v_k Γ^T = S_jk z_jk^H + z_jk S_jk^H
    A1, A2, A3, A4 = I_3, I_3, Gamma, Gamma.T
    B1, B2 = I_3, z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def DD_transformer(z_jk: np.ndarray):
    # Γ v_j Γ^T - a_jk^2 Γ v_k Γ^T = (1/3 Γ S_jk Γ^T) z_jk^H + z_jk (1/3 Γ S_jk^H Γ^T)
    A1, A2, A3, A4 = Gamma, Gamma.T, Gamma, Gamma.T
    B1, B2 = (1/3) * Gamma, Gamma.T @ z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def DY_alt_transformer(z_jk: np.ndarray):
    # Alternative Delta-Wye winding pairing
    # Γ^T v_j Γ - a_jk^2 v_k ≈ z_jk (1/3 Γ^T S_jk Γ)^H + (1/3 Γ^T S_jk Γ) z_jk^H
    A1, A2, A3, A4 = Gamma.T, Gamma, I_3, I_3
    B1 = (1/3) * Gamma.T
    B2 = Gamma @ z_jk.conj().T
    return A1, A2, A3, A4, B1, B2


def component_model(circuit_data):
    """
    The component model is 
        G_jk @ diag(v_j or v_j_Delta) - a_jk^2 * diag(v_k) = R_jk @ P_jk + X_jk @ Q_jk
        G_jk_Delta @ diag(v_j_Delta) - a_jk^2 * diag(v_k_Delta) = R_jk_Delta @ P_jk + X_jk_Delta @ Q_jk
    :param branch_data: dict, branch data, e.g.
        {
            'element_type': 'DY',
            'source': 'bus_1',
            'target': 'bus_2',
            'series_impedance': np.array with shape = (3, 3),
        }
    :param z_ij: np.array, shape=(3, 3), series impedance matrix
    :return: 
        np.ndarray A_j, A_k, B, B_H, shape=(N,9,9)
        Note that the arrays are indexed by branch.
    """
    N = len(circuit_data)
    A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 = [None]*N, [None]*N, [None]*N, [None]*N
    for e, branch_data in enumerate(circuit_data):
        # Standardize series impedance to 3x3 matrix
        if 'series_impedance' in branch_data:
            if type(branch_data['series_impedance']) in (float, complex):
                z_ij = branch_data['series_impedance'] * np.eye(3)
            else:
                z_ij = np.array(branch_data['series_impedance'])
                if z_ij.shape == (3,):
                    z_ij = np.diag(z_ij)
                elif z_ij.shape == (3,3):
                    pass
                elif z_ij.shape == (9):
                    z_ij = z_ij.reshape(3,3)
                else:
                    raise ValueError(f"Invalid series impedance shape: {z_ij.shape}\n{z_ij}")
        else:
            z_ij = np.zeros((3, 3), dtype=np.csingle)
            name = branch_data['element_name'] if 'element_name' in branch_data \
                else f"{branch_data['source']} -> {branch_data['target']}"
            print(f"[Warning] branch {name} has no series impedance. Using zero matrix.")
        # Generalized branch model
        if branch_data['element_type'] in ['line', 'YY', 'DY', 'YD', 'DD', 'DY_alt', 'openY_openY']:
            if branch_data['element_type'] == 'line':
                A_1, A_2, A_3, A_4, B_1, B_2 = line(z_ij)
            elif branch_data['element_type'] in {'YY', 'openY_openY'}:
                A_1, A_2, A_3, A_4, B_1, B_2 = YY_transformer(z_ij)
            elif branch_data['element_type'] == 'DD':
                A_1, A_2, A_3, A_4, B_1, B_2 = DD_transformer(z_ij)
            elif branch_data['element_type'] == 'DY':
                A_1, A_2, A_3, A_4, B_1, B_2 = DY_transformer(z_ij)
            elif branch_data['element_type'] == 'DY_alt':
                A_1, A_2, A_3, A_4, B_1, B_2 = DY_alt_transformer(z_ij)
            elif branch_data['element_type'] == 'YD':
                A_1, A_2, A_3, A_4, B_1, B_2 = YD_transformer(z_ij)
            A_j_9x9[e] = np.kron(A_2.T, A_1)
            A_k_9x9[e] = np.kron(A_4.T, A_3)
            B_9x9[e] = np.kron(B_2.T, B_1)
            B_H_9x9[e] = np.kron(B_1.conj(), B_2.conj().T)
        # TODO: The following two have not been verified. Write test cases.
        elif branch_data['element_type'] == 'openY_openD_split':
            wp = list(branch_data['winding_pairing'])
            assert len(wp) == 2, wp
            # Split phase must be present for the derivation of openY_openD transformer (U_delta matrix)
            assert branch_data['split_phase'] in wp, (branch_data['split_phase'], wp)
            permutation_pri, permutation_sec = build_permutation_matrix_open_split(branch_data)
            p_pri_kron = np.kron(permutation_pri, permutation_pri)
            p_sec_kron = np.kron(permutation_sec, permutation_sec)
            # Instantiate the matrices for canonical pairing
            z_ij = z_ij[:2, :2]
            A_j_9x9[e] = np.kron(U_delta, U_delta) @ np.kron(I_23, I_23)
            A_k_9x9[e] = I_9
            B_9x9[e] = np.kron((U_delta @ z_ij.conj()), U_delta) @ np.kron(I_23, I_23)
            B_H_9x9[e] = np.kron(U_delta, (U_delta @ z_ij)) @ np.kron(I_23, I_23)
            # Apply permutation to get the actual pairing
            A_j_9x9[e] = p_sec_kron @ A_j_9x9[e] @ p_pri_kron
            A_k_9x9[e] = p_sec_kron @ A_k_9x9[e] @ p_pri_kron
            B_9x9[e] = p_sec_kron @ B_9x9[e] @ p_pri_kron
            B_H_9x9[e] = p_sec_kron @ B_H_9x9[e] @ p_pri_kron
        elif branch_data['element_type'] in {'openD_openD', 'openD_openD_split'}:
            wp = list(branch_data['winding_pairing'])
            assert len(wp) == 2, wp
            # TODO: remove
            if ('split_phase' in branch_data) and branch_data['split_phase']:
                assert branch_data['split_phase'] in wp, (branch_data['split_phase'], wp)
            permutation_pri, permutation_sec = build_permutation_matrix_open_split(branch_data)
            p_pri_kron = np.kron(permutation_pri, permutation_pri)
            p_sec_kron = np.kron(permutation_sec, permutation_sec)
            # Instantiate the matrices for canonical pairing
            z_ij = z_ij[:2, :2]
            A_j_9x9[e] = I_9
            A_k_9x9[e] = I_9
            B_9x9[e] = 1/9 * np.kron(Gamma.T @ T_delta, Gamma.T @ T_delta) \
                @ np.kron(z_ij.conj() @ Q_delta, Gamma[:2])
            B_H_9x9[e] = 1/9 * np.kron(Gamma.T @ T_delta, Gamma.T @ T_delta) \
                @ np.kron(Gamma[:2], z_ij @ Q_delta)
            # Apply permutation to get the actual pairing
            A_j_9x9[e] = p_sec_kron @ A_j_9x9[e] @ p_pri_kron
            A_k_9x9[e] = p_sec_kron @ A_k_9x9[e] @ p_pri_kron
            B_9x9[e] = p_sec_kron @ B_9x9[e] @ p_pri_kron
            B_H_9x9[e] = p_sec_kron @ B_H_9x9[e] @ p_pri_kron            
        else:
            raise ValueError(f"Unknown element type: {branch_data}")
    return A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 


def build_permutation_matrix_open_split(branch_data):
    """
    The permutation matrix (i, j)-th index means (to, from).
    """
    # Canonical pairing: A - AB (split), B - BC, C - CA (open)
    wp = list(branch_data['winding_pairing'])
    permutation_pri, permutation_sec = np.zeros((3, 3), dtype=np.csingle), np.zeros((3, 3), dtype=np.csingle)
    # Split phase first
    p = 0
    if ('split_phase' in branch_data) and branch_data['split_phase']:
        wp.remove(branch_data['split_phase'])
        pri_idx = branch_data['split_phase'][0][0] - 1 if len(branch_data['split_phase'][0]) == 1 else STD_DELTA_ORDER.index(branch_data['split_phase'][0])
        sec_idx = STD_DELTA_ORDER.index(branch_data['split_phase'][1])
        permutation_pri[0, pri_idx] = 1        # Move C to A
        permutation_sec[0, sec_idx] = 1  # Move split phase to A
        p = 1
    # Regular connection
    for pri, sec in wp:
        pri_idx = pri[0] - 1 if len(pri) == 1 else STD_DELTA_ORDER.index(pri)
        sec_idx = sec[0] - 1 if len(sec) == 1 else STD_DELTA_ORDER.index(sec)
        permutation_pri[p, pri_idx] = 1
        permutation_sec[p, sec_idx] = 1
        p += 1
    # Open connection
    pri, sec = branch_data['open_phase']
    # TODO: The sign matters (+1 vs. -1). Right now we just ignored them.
    pri_idx = pri[0] - 1 if len(pri) == 1 else STD_DELTA_ORDER.index(pri)
    sec_idx = sec[0] - 1 if len(sec) == 1 else STD_DELTA_ORDER.index(sec)
    permutation_pri[2, pri_idx] = 1          # Move primary side open phase to C
    permutation_sec[2, sec_idx] = 1         # Move secondary side open phase to CA
    permutation_sec = permutation_sec.T
    assert (permutation_pri.sum(axis=0) == 1).all() and (permutation_pri.sum(axis=1) == 1).all(), (permutation_pri, branch_data)
    assert (permutation_sec.sum(axis=0) == 1).all() and (permutation_sec.sum(axis=1) == 1).all(), (permutation_sec, branch_data)
    return permutation_pri, permutation_sec


"""
Network relations
"""
import numpy as np
import networkx as nx

import numpy as np
import networkx as nx

def branch_nodal_power(graph, branch_data, edge2idx, N):
    def sort_edges_by_depth(graph):
        depth = {n: 0 for n in graph.nodes}
        root = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
        for node in nx.bfs_tree(graph, root).nodes:
            for child in graph.successors(node):
                depth[child] = depth[node] + 1
        return sorted(graph.edges, key=lambda e: depth[e[1]], reverse=True)

    M = np.zeros((N, 9, N, 9), dtype=np.csingle)
    B = branch_pwr_relation[branch_data['element_type']]

    for (i, j) in sort_edges_by_depth(graph):
        e = edge2idx[(i, j)]
        for k in graph.successors(j):
            e_out = edge2idx[(j, k)]
            # B (9,9) applied to M[e_out] (9, N, 9) -> result (9, N, 9)
            M[e] += np.einsum('ab,bnj->anj', B, M[e_out])
        M[e, :, j - 1, :] -= B

    return M.reshape(N * 9, N * 9)


def is_downward_oriented(G, root):
    # Check in-degree conditions
    if G.in_degree(root) != 0:
        return False
    for node in G.nodes:
        if node != root and G.in_degree(node) != 1:
            return False

    # Check reachability: all nodes must be reachable from root
    reachable = nx.descendants(G, root) | {root}
    return len(reachable) == len(G.nodes)


def three_phase_lindistflow(circuit_data, node_list=None, return_internal_voltages=False, verbose=False):
    """
    The main function for 3-phase LinDistFlow.
    Returns the bus voltage magnitudes (squared) in terms of nodal power injections.
    Note that we define positive branch current and power direction to follow the edge direction. 
    We define positive nodal current and power injection as into the node.

    :param circuit_data: dict, edge list, e.g.
        [
            {
                'element_name': 'line_1',
                'element_type': 'DY',
                'source': 'bus_1',
                'target': 'bus_2',
                'series_impedance': np.array with shape = (3, 3),
                'turns_ratio': 1.0,
            },
            ...
        ]
    :param node_list: list, the list of node names. If None, use BFS order from root node.
    :param return_internal_voltages: bool, if True, then the voltage solutions are the
        internal voltages. Moreover, the transformation matrix mapping the external
        voltages to the internal voltages is provided, i.e. v_internal = A_k_9x9_node @ v_external.
    :return:
        C_0: np.ndarray, shape=(N*9, 9)
        K_1: np.ndarray, shape=(N*9, N*9)
        K_2: np.ndarray, shape=(N*9, N*9)
        C_S: np.ndarray, shape=(N*9, N*9)
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
    edge2idx = {edge: e for e, edge in enumerate(edge_list)}
    # Caution: graph.edges is generally in a differnt order as edge_list
    graph = nx.DiGraph(edge_list)
    # Pre-process circuit_data
    for branch_data in circuit_data:
        if (branch_data['element_type'] == 'line') and ('turns_ratio' not in branch_data):
            branch_data['turns_ratio'] = 1.0
    
    # Step 1: Get component models
    if verbose: print("Building component models...")
    A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 = component_model(circuit_data)
    # Change A_k_9x9 to be indexed by node
    A_k_9x9_node = np.zeros((N+1, 9, 9), dtype=np.csingle)
    A_k_9x9_node[0] = I_9
    for e, (j, k) in enumerate(edge_list):
        A_k_9x9_node[k] = A_k_9x9[e]

    # Step 2: Branch voltage drop in terms of branch power, v_0 voltage reference
    if verbose: print("Building voltage drop relations...")
    nodes_list_bfs_order = list(nx.bfs_tree(graph, source=0).nodes)
    K_1 = np.zeros((N+1, 9, N, 9), dtype=np.csingle)
    K_2 = np.zeros((N+1, 9, N, 9), dtype=np.csingle)
    C = np.zeros((N+1, 9, 9), dtype=np.csingle)   # C[k]: branch voltage at node k in terms of v_0
    C[0] = I_9
    # Traverse BFS to ensure parents is visited before children
    # Dynamic programming recursion on branch voltage
    for k in nodes_list_bfs_order[1:]:
        j = list(graph.predecessors(k))[0]
        e = edge2idx[(j, k)]
        branch_data = circuit_data[e]
        a_jk = branch_data['turns_ratio']
        # Here we keep everything in terms of internal voltages. We don't do the 
        # pseudo-inverse to obtain the external voltages until when necessary. 
        # This avoids unnecessary inaccuracy and application of span 1 assumption.
        if np.allclose(A_j_9x9[e], A_k_9x9_node[j]):
            pinv = I_9
        else:
            pinv = np.linalg.pinv(A_k_9x9_node[j])
        # Coefficient for v_0
        C[k] = (a_jk ** (-2)) * A_j_9x9[e] @ pinv @ C[j]
        # Voltage drops for all parent branches
        M_mat = (a_jk ** (-2)) * (A_j_9x9[e] @ pinv)
        # K_1[j] has shape (9, N, 9). We need to left-multiply M_mat (9x9) to the 9-dim axis:
        K_1[k] = np.einsum('ab,bcd->acd', M_mat, K_1[j])
        K_2[k] = np.einsum('ab,bcd->acd', M_mat, K_2[j])
        # Voltage drop for this branch: place the 9x9 block into axis 1 (edge index)
        K_1[k][:, e, :] = (a_jk ** (-2)) * B_9x9[e]
        K_2[k][:, e, :] = (a_jk ** (-2)) * B_H_9x9[e]
    if not return_internal_voltages:
        for n in range(N):
            if (A_k_9x9_node[n] == I_9).all(): continue
            pinv = np.linalg.pinv(A_k_9x9_node[n])
            C[n] = pinv @ C[n]
            # Left-multiply pinv into each 9x9 block: K_1[n] shape (9, N, 9)
            K_1[n] = np.einsum('ab,bcd->acd', pinv, K_1[n])
            K_2[n] = np.einsum('ab,bcd->acd', pinv, K_2[n])
    # Discard the first element (root node) from the lists and re-shape
    C_0 = C[1:].reshape(N*9, 9)
    K_1 = -K_1[1:].reshape(N*9, N*9)
    K_2 = -K_2[1:].reshape(N*9, N*9)

    # Step 3: branch power in terms of nodal power injections
    if verbose: print("Building branch power relations...")
    C_S = branch_nodal_power(graph, branch_data, edge2idx, N)
    # Note that edge_list is possibly in a different order as graph.edges
    edge_list_out = [(node_list[e[0]], node_list[e[1]]) for e in edge_list]

    return C_0, K_1, K_2, C_S, node_list, edge_list_out


def three_phase_lindistflow_real(C_0, K_1, K_2, C_S, N, node_list, node_data, verbose=False):
    if verbose: print("Converting to real-valued LinDistFlow model...")
    # This acts as the O matrix
    diag_indices = np.array([0, 4, 8])
    diag_indices = np.concatenate([diag_indices + 9*n for n in range(N)])
    K_1, K_2 = K_1[diag_indices], K_2[diag_indices]
    C_0_real = C_0[diag_indices]
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
    R = np.real(K_1 @ C_S @ gamma_tilde + K_2 @ C_S @ gamma_tilde_H)
    X = np.imag(-K_1 @ C_S @ gamma_tilde + K_2 @ C_S @ gamma_tilde_H)
    C_S_real = C_S[diag_indices] @ gamma_tilde
    assert (np.abs(C_S_real.imag) < 1e-6).all(), "C_S_real has imaginary parts."
    C_S_real = C_S_real.real
    assert (np.abs(C_0_real.imag) < 1e-6).all(), "C_0_real has imaginary parts."
    C_0_real = C_0_real.real
    
    return C_0_real, R, X, C_S_real


def build_node_data(feeder_root=None, graph=None, circuit_data=None, circuit_dict=None, node_list=None):
    """
    Populate nodal data. So far we only track the upstream connected transformer for each node.
    """
    assert circuit_data or circuit_dict, "Either circuit_data or circuit_dict must be provided."
    if not circuit_dict:
        circuit_dict = {(e['source'], e['target']): e for e in circuit_data}
    if graph is None:
        edge_list = to_edge_list(circuit_data)
        graph = nx.DiGraph(edge_list)
    if feeder_root is None:
        feeder_root = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    node_data = {feeder_root: {'connected_transformer': None}}
    nodes_to_keep = node_list or graph.nodes
    # Single BFS from feeder_root gives shortest paths to every node at once,
    # instead of re-searching the graph from scratch for each node below.
    paths_from_root = nx.shortest_path(graph, feeder_root)
    num_without_transformer = 0
    for k in nodes_to_keep:
        if k == feeder_root: continue
        connected_transformer = None
        path = paths_from_root[k]
        for idx in list(range(len(path)-1))[::-1]:
            i, j = path[idx], path[idx+1]
            if circuit_dict[(i, j)]['element_type'] != 'line':
                connected_transformer = circuit_dict[(i, j)]
                break
        node_data[k] = {
            'connected_transformer': connected_transformer,
        }
        if connected_transformer is None:
            num_without_transformer += 1
    if num_without_transformer:
        # Expected: primary-side buses (upstream of any service transformer) have
        # no "connected transformer" by definition -- not an error.
        print(f"[Info] {num_without_transformer} node(s) have no upstream transformer "
              f"(primary-side buses).")
    return circuit_dict, node_data


def three_phase_lindistflow_real_wrapper(circuit_data, node_data=None, node_list=None, verbose=False):
    C_0, K_1, K_2, C_S, node_list, edge_list = \
        three_phase_lindistflow(circuit_data, node_list=node_list, verbose=verbose)
    if node_data is None:
        _, node_data = build_node_data(circuit_data=circuit_data, node_list=node_list)
    C_0_real, R, X, C_S_real = \
        three_phase_lindistflow_real(C_0, K_1, K_2, C_S, len(node_list)-1, node_list, node_data, verbose=verbose)
    return C_0_real, R, X, C_S_real, node_list, edge_list


def compute_branch_sensitivity_matrices(circuit_data, node_list):
    """
    Compute the per-branch block-diagonal resistance/reactance matrices (D_r,
    D_x) and incidence-based coefficient matrix (C_coef_inv, c_0) used by the
    Appendix B dual-variable sensitivity analysis (Danskin's theorem), in the
    same node ordering as `three_phase_lindistflow_real_wrapper`'s output.

    This uses a scalar per-branch turns ratio, which is exact for line
    segments and for 3-phase transformers with a uniform per-phase ratio
    (YY, DY, YD), but is not a valid representation for split-phase or
    open-delta transformers with per-phase-differentiated ratios. Check that
    your feeder has no such transformers before relying on this for a new
    feeder (see README).
    """
    edge_list = to_edge_list(circuit_data)
    node2idx = {node: k for k, node in enumerate(node_list)}
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    edge2idx = {edge: e for e, edge in enumerate(edge_list)}
    G = nx.DiGraph(edge_list)
    N = len(node_list) - 1
    node_ind = list(range(N + 1))
    # Incidence matrix (note that networkx has the opposite of canonical sign convention)
    C_matrix = -nx.incidence_matrix(G, oriented=True, nodelist=node_ind, edgelist=edge_list).toarray()
    c_0 = C_matrix[0]
    C_hat = C_matrix[1:, :]

    # Diagonal impedance, turns ratio matrix
    D_z, T = [], []
    for e in edge_list:
        branch_data = circuit_data[edge2idx[e]]
        if 'series_impedance' in branch_data:
            # Convert to 3x3 matrix
            if type(branch_data['series_impedance']) in (float, complex):
                z_ij = branch_data['series_impedance'] * np.eye(3)
            else:
                z_ij = np.array(branch_data['series_impedance'])
                if z_ij.shape == (3,):
                    z_ij = np.diag(z_ij)
                elif z_ij.shape == (3,3):
                    pass
                elif z_ij.shape == (9):
                    z_ij = z_ij.reshape(3,3)
                else:
                    raise ValueError(f"Invalid series impedance shape: {z_ij.shape}\n{z_ij}")
        else:
            z_ij = np.zeros((3, 3), dtype=np.csingle)
            print('Warning: No series impedance found for branch:', e)
        D_z.append(z_ij * gamma.T)
        if 'turns_ratio' in branch_data:
            if branch_data['element_type'] in ['DY', 'DY_alt']:
                T.append((branch_data['turns_ratio'] / np.sqrt(3)) ** 2)
            elif branch_data['element_type'] in ['YD', 'YD_alt']:
                T.append((branch_data['turns_ratio'] * np.sqrt(3)) ** 2)
            else:
                T.append(branch_data['turns_ratio'] ** 2)
        elif branch_data['element_type'] == 'line':
            T.append(1.0)
        else:
            raise ValueError("Unknown element {} has no turns ratio specified".format(branch_data['element_type']))
    D_z = sp.linalg.block_diag(*D_z)
    D_r, D_x = np.real(D_z), np.imag(D_z)
    T = np.diag(T)
    # Matrix that maps branch voltage drops to nodal voltages
    C_n, C_p = np.maximum(-C_hat.T, 0), np.maximum(C_hat.T, 0)
    C_coef = C_p - T @ C_n
    C_coef_inv = np.kron(np.linalg.inv(C_coef), np.eye(3))
    c_0_kron = np.kron(c_0, np.eye(3)).T
    return C_coef_inv, D_r, D_x, c_0_kron


def single_phase_lindistflow(circuit_data, per_unit=True, s_base=1, node_list=None):
    """
    An implementation of the single-phase LinDistFlow is provided for reference.
    Note that the R, X values are AVERAGED (not SUMMED) across the three phases.
    This means you should average the p, q injections as well, not sum.
    :param circuit_data: see three_phase_lindistflow function
    :param per_unit: bool, if True, convert all quantities to per-unit
        Only p.u. is supported for now.
    :param s_base: float, base power for per-unit conversion
    return: R_matrix, X_matrix, C_matrix, node_list
        R_matrix: np.array, shape=(N, N)
        X_matrix: np.array, shape=(N, N)
        C_matrix: np.array, shape=(N+1, N)
        node_list: list of str, node names
        Usage:
            nodal_voltages = v0 + 2 * (R_matrix @ p_inj[1:] + X_matrix @ q_inj[1:])
            branch_power = np.linalg.inv(C_matrix[1:, :]) @ (p + 1j * q)
    """
    edge_list = to_edge_list(circuit_data)
    graph = nx.DiGraph(edge_list)
    root_node = [n for n in graph.nodes if graph.in_degree(n) == 0]
    assert len(root_node) == 1, f"There should be only one root node: {root_node}"
    assert nx.is_directed_acyclic_graph(graph), "Graph contains a cycle. Only radial networks are supported."
    N = len(graph.nodes) - 1    # Excluding the root node
    assert len(graph.edges) == N, len(graph.edges)
    # Build node_list and assign node id
    node_list = node_list or list(nx.bfs_tree(graph, root_node[0]).nodes())
    assert node_list[0] == root_node[0], f"Root node {node_list[0]} does not match {root_node[0]}"
    node2idx = {node: k for k, node in enumerate(node_list)}
    # Replace node names with integer indicies
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in edge_list]
    graph = nx.DiGraph(edge_list)
    # Check that the node_list is in BFS or DFS order
    for i in range(1, N+1):
        assert not (set(graph.predecessors(i)) - set(range(i))), \
            str('\n'.join([n.split('|')[0] for n in node_list]) + '\n' + str(set(graph.predecessors(i))) + ' ' + str(i))
    
    # Careful! The graph needs to be in the down orientation.
    # networkx incidence_matrix has the opposite sign convention
    C_matrix = -nx.incidence_matrix(graph, oriented=True, nodelist=list(range(N+1))).toarray()
    ge = list(graph.edges)
    permutation = [ge.index((i, j)) for i, j in edge_list]
    C_matrix = C_matrix[:, permutation]
    C_inv = np.linalg.inv(C_matrix[1:])
    # plot_nx_graph(graph, node_info={i: n.split('|')[0] for i, n in enumerate(node_list)})

    resistances, reactances, V_base, Z_base = np.zeros(N), np.zeros(N), np.zeros(N+1), np.zeros(N)
    for e, (i, j) in enumerate(edge_list):
        # Average impedance over three phases
        z_ij = np.diag(circuit_data[e]['series_impedance']).mean()
        v0, v1 = circuit_data[e]['nominal_voltage']
        V_base[i] = v0
        V_base[j] = v1
        Z_base[e] = v0 ** 2 / s_base
        resistances[e] = np.real(z_ij)
        reactances[e] = np.imag(z_ij)
    if per_unit:
        C_inv_T_T = C_inv.T @ np.diag(Z_base ** -1)
    else:
        C_inv_T_T = np.diag(V_base[1:] ** 2) @ C_inv.T @ np.diag(Z_base ** -1)
    R_matrix = C_inv_T_T @ np.diag(resistances) @ C_inv
    X_matrix = C_inv_T_T @ np.diag(reactances) @ C_inv
    if not (R_matrix >= 0).all():
        print("Warning: Some entries in R_matrix are negative.")
    if not (X_matrix >= 0).all():
        print("Warning: Some entries in X_matrix are negative.")
    return R_matrix, X_matrix, C_matrix, node_list, V_base, edge_list

    
"""Helpful usage functions"""
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


def plot_nx_graph(
        G, 
        outpath='temp/graph.png', 
        show=True, 
        edge_info=None, 
        node_info=None, 
        figsize=(12,12), 
        node_size=100, 
        pos=None,
        highlight_cycles=True,
        plot_cycles=False,
        node_color='blue',
    ):
    import random
    import matplotlib
    matplotlib.use("Agg")  # non-interactive: avoid blocking on plt.show() with no display
    import matplotlib.pyplot as plt
    if not pos:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")
        except Exception as e:
            print("pygraphviz not installed or error. Using spring layout.")
            pos = nx.spring_layout(G)
    pos = {k: (v[0], v[1] + (random.random() - 0.5) * 40) for k, v in pos.items()}

    fig = plt.figure(figsize=figsize)

    if edge_info:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: edge_info[k] for k in G.edges}, font_color="red", rotate=False)
    if node_info:
        pos_ = {k: (v[0] + 5, v[1] + 5) for k, v in pos.items()}
        nx.draw_networkx_labels(G, pos_, labels=node_info, font_color="red", font_size=figsize[0], font_weight='bold')

    nx.draw(G, pos, node_size=node_size, node_color=node_color, with_labels=True, font_size=12)

    if highlight_cycles:
        cycles = nx.cycle_basis(G.to_undirected())
        for cycle in cycles:
            edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", width=2)
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if plot_cycles and outpath:
        cycles = nx.cycle_basis(G.to_undirected())
        for c, cycle in enumerate(cycles):
            plt.figure(figsize=figsize)
            H = G.subgraph(cycle)
            nx.draw(H, pos, node_size=node_size, with_labels=True, font_size=12)
            name, ext = os.path.splitext(outpath)
            plt.savefig(name + f'_cycle_{c}{ext}')
            plt.close()
