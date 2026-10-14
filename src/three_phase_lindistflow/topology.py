"""Radial-network graph construction, validation and traversal."""

import logging
import networkx as nx
import numpy as np

from .branches import _winding_phases, floating_neutrals

logger = logging.getLogger(__name__)

__all__ = [
    "to_edge_list",
    "to_edge_dict",
    "edges_between_nodes",
    "subtree",
    "is_downward_oriented",
    "check_network_validity",
    "build_node_data",
    "zero_sequence_groups",
    "zero_sequence_islands",
]

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


def edges_between_nodes(graph, source, target, cache=None):
    """
    :param G: networkx.DiGraph
    :param source: str or int, source node name
    :param target: str or int, target node name
    :return: list of tuple of (src, tgt), which are edges in path
    """
    if cache is not None and (source, target) in cache:
        return cache[(source, target)]
    nodes_in_path = list(nx.all_simple_paths(graph, source=source, target=target))
    if not nodes_in_path:
        if cache is not None:
            cache[(source, target)] = []
        return []
    else:
        nodes_in_path = nodes_in_path[0]
    edges = list(zip(nodes_in_path[:-1], nodes_in_path[1:]))
    if cache is not None:
        cache[(source, target)] = edges
    return edges


def subtree(graph, node, cache=None, sort=True):
    """
    Return the list of nodes in subtree at node, excluding node.
    To include node, use subtree(graph, node) | {node}
    """
    if cache is not None and node in cache:
        return cache[node]
    subtree_nodes = nx.descendants(graph, node)
    if sort:
        subtree_nodes = sorted(subtree_nodes)
    if cache is not None:
        cache[node] = subtree_nodes
    return subtree_nodes


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


def check_network_validity(circuit_data):
    edge_list = to_edge_list(circuit_data)
    graph = nx.DiGraph(edge_list)

    # Root: exactly one node with no parent
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    assert len(roots) == 1, f"Expected exactly one root (in_degree==0), got: {roots}"
    root = roots[0]

    # Downward-oriented from root: edges must follow BFS parent relation (since it's a tree)
    bfs_tree = nx.bfs_tree(graph, root)
    assert set(bfs_tree.edges) == set(graph.edges), (
        "Edge directions are not consistently outward from the root (some edges contradict BFS orientation)."
    )

    # No directed cycles (equivalent to no cycles in downward-oriented graph)
    assert nx.is_directed_acyclic_graph(graph), "Graph contains a directed cycle; only radial networks supported."

    # Connected from the root (reachability)
    reachable = set(nx.descendants(graph, root)) | {root}
    assert reachable == set(graph.nodes), (
        f"Graph is not fully reachable from root {root}. "
        f"Unreachable nodes: {sorted(set(graph.nodes) - reachable)}"
    )

    # Radial structure: |E| = |V|-1 and each non-root has exactly one parent
    assert len(graph.edges) == len(graph.nodes) - 1, (
        f"Expected |E|=|V|-1 for a tree, got |E|={len(graph.edges)}, |V|={len(graph.nodes)}"
    )
    bad_indeg = [n for n in graph.nodes if n != root and graph.in_degree(n) != 1]
    assert not bad_indeg, (
        f"Each non-root node must have in_degree==1 (exactly one parent). Bad nodes: {bad_indeg}"
    )

    # Nodes in bfs order
    node_list = list(bfs_tree.nodes())
    assert node_list and node_list[0] == root, f"BFS root mismatch: node_list[0]={node_list[0]} vs root={root}"

    return graph, node_list, edge_list, root


def build_node_data(root_node=None, graph=None, circuit_data=None, circuit_dict=None, node_list=None):
    """
    Populate nodal data. So far we only track the upstream connected transformer for each node.
    """
    assert circuit_data or circuit_dict, "Either circuit_data or circuit_dict must be provided."
    if not circuit_dict:
        circuit_dict = {(e['source'], e['target']): e for e in circuit_data}
    if graph is None:
        edge_list = to_edge_list(circuit_data)
        graph = nx.DiGraph(edge_list)
    if root_node is None:
        root_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    node_data = {root_node: {'connected_transformer': None}}
    nodes_to_keep = node_list or graph.nodes
    for k in nodes_to_keep:
        if k == root_node: continue
        connected_transformer = None
        path = nx.shortest_path(graph, root_node, k)
        for idx in list(range(len(path)-1))[::-1]:
            i, j = path[idx], path[idx+1]
            if circuit_dict[(i, j)]['element_type'] != 'line':
                connected_transformer = circuit_dict[(i, j)]
                break
        node_data[k] = {
            'connected_transformer': connected_transformer,
        }
        if connected_transformer is None:
            logger.info("No upstream transformer found for node %s.", k)
    return circuit_dict, node_data


def zero_sequence_groups(circuit_data):
    """Groups of buses joined by zero-sequence paths, with what supplies each zero sequence.

    Zero-sequence current passes lines and wye-wye banks whose neutrals are both grounded; a
    delta winding or a floating wye neutral blocks it. It is supplied by the root (an ideal
    source) and by a grounded wye winding facing a delta one, which circulates it: the
    secondary of a delta-wye bank and the primary of a three-phase wye-delta bank. The
    center-tapped secondary of a split-phase bank is grounded at its tap, so it is a source
    as well.

    Returns a list of (buses, sources) pairs, one per group, ordered by their first bus in the
    BFS order of the network. ``buses`` is a set of bus names and ``sources`` a list of
    ``("root", root)`` and ``(element_type, (source, target))`` entries.
    """
    _, node_list, _, root = check_network_validity(circuit_data)
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    sources = {root: [("root", root)]}
    for branch_data in circuit_data:
        edge = (branch_data["source"], branch_data["target"])
        element_type = branch_data["element_type"]
        floating = floating_neutrals(branch_data)
        if element_type == "line" or (element_type in ("YY", "openY_openY") and not any(floating)):
            graph.add_edge(*edge)
        elif (element_type in ("DY", "DY_alt") and not floating[1]) or element_type.endswith("_split"):
            sources.setdefault(edge[1], []).append((element_type, edge))
        # (A lone single-phase wye-delta unit has no closed delta to circulate it in.)
        elif element_type in ("YD", "YD_alt") and not floating[0] and _winding_phases(branch_data, 0) in (None, {1, 2, 3}):
            sources.setdefault(edge[0], []).append((element_type, edge))
    order = {node: k for k, node in enumerate(node_list)}
    groups = sorted(nx.connected_components(graph), key=lambda buses: min(order[bus] for bus in buses))
    return [(buses, [source for bus in sorted(buses, key=order.get) for source in sources.get(bus, [])])
            for buses in groups]


def zero_sequence_islands(circuit_data):
    """Sets of buses that no zero-sequence current can reach (see zero_sequence_groups).

    They lie below a delta winding or a floating wye neutral, with no grounded wye-delta bank
    in between. Nothing can supply or take the zero-sequence part of the currents injected in
    an island, so their sums over the phases must cancel over its buses, once referred
    through the turns ratios of the wye-wye banks inside it (in per unit: sum to zero).
    """
    return [buses for buses, sources in zero_sequence_groups(circuit_data) if not sources]
