"""Optional network plotting helpers (requires matplotlib, and graphviz for layout)."""

import logging
import networkx as nx
import os
import random

logger = logging.getLogger(__name__)

__all__ = [
    "plot_nx_graph",
]

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
        node_color='yellow',
    ):
    import random
    import matplotlib.pyplot as plt
    if not pos:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")
        except Exception as e:
            logger.warning("pygraphviz not installed or error. Using spring layout.")
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
