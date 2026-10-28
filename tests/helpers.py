import networkx as nx

from rcatwin.twin import _finish
import numpy as np


def tree_twin():
    """g(0) - a(1) - {b(2) - {c(4), d(5)}, e(3)};  g is the gateway.  b's failure disconnects c, d."""
    G = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)])
    for v in G:
        G.nodes[v].update(tier="x", region="hierarchical")
    return _finish(G, "hierarchical", [0], np.random.default_rng(0))


def ring_twin(k=8):
    """A ring with one gateway (0): no single non-gateway failure disconnects anything."""
    G = nx.cycle_graph(k)
    for v in G:
        G.nodes[v].update(tier="x", region="mesh")
    return _finish(G, "mesh", [0], np.random.default_rng(0))
