"""Synthetic network digital twin: seeded topology generators.

A twin is an undirected networkx.Graph whose nodes carry
  tier    : core / agg / edge / cell / hub / leaf / mesh / gw   (role of the device)
  region  : hierarchical / star / mesh                         (local topology type; ground truth for triage)
  poll    : polling phase in [0, 1) used by the alarm simulator
and graph attributes
  kind     : hierarchical / star / mesh / hybrid
  gateways : nodes that connect the network to the outside (service reachability is measured to them)
  layer    : dict node -> hop distance to the nearest gateway in the healthy network
Upstream dependency edges u -> v (v one hop closer to a gateway) are given by `upstream(G)`.
"""
import numpy as np
import networkx as nx

KINDS = ("hierarchical", "star", "mesh", "hybrid")


def _finish(G, kind, gateways, rng):
    G.graph.update(kind=kind, gateways=list(gateways))
    G.graph["layer"] = layers(G, gateways)
    for v in G:
        G.nodes[v]["poll"] = float(rng.random())
    assert nx.is_connected(G), "generator produced a disconnected twin"
    return G


def layers(G, gateways, removed=None):
    """Multi-source BFS hop distance to the nearest gateway, optionally with `removed` nodes deleted."""
    H = G if removed is None else nx.restricted_view(G, removed, [])
    src = [g for g in gateways if g not in (removed or ())]
    return dict(nx.multi_source_dijkstra_path_length(H, src, weight=lambda u, v, d: 1)) if src else {}


def upstream(G):
    """Directed dependency graph: edge u -> v when v is one hop closer to a gateway than u."""
    L = G.graph["layer"]
    D = nx.DiGraph()
    D.add_nodes_from(G)
    D.add_edges_from((u, v) for a, b in G.edges for u, v in ((a, b), (b, a)) if L[v] == L[u] - 1)
    return D


def _add(G, tier, region):
    v = G.number_of_nodes()
    G.add_node(v, tier=tier, region=region)
    return v


def hierarchical(n, rng, p_dual_edge=0.2, n_core=4, region_star=False):
    """core (full mesh, gateways) -> agg (dual-homed to 2 cores) -> edge (single-homed, p_dual_edge dual) -> cell."""
    G = nx.Graph()
    cores = [_add(G, "core", "hierarchical") for _ in range(n_core)]
    G.add_edges_from((a, b) for i, a in enumerate(cores) for b in cores[i + 1:])
    n_agg = max(2, int(round(0.04 * n)))
    n_edge = max(n_agg, int(round(0.2 * n)))
    aggs = []
    for _ in range(n_agg):
        a = _add(G, "agg", "hierarchical"); aggs.append(a)
        G.add_edges_from((a, c) for c in rng.choice(cores, 2, replace=False))
    edges = []
    for _ in range(n_edge):
        e = _add(G, "edge", "hierarchical"); edges.append(e)
        k = 2 if rng.random() < p_dual_edge else 1
        G.add_edges_from((e, a) for a in rng.choice(aggs, k, replace=False))
    while G.number_of_nodes() < n:
        c = _add(G, "cell", "hierarchical")
        G.add_edge(c, int(rng.choice(edges)))
    return _finish(G, "hierarchical", cores, rng)


def star(n, rng, leaves_per_hub=50, p_dual_leaf=0.0):
    """Hub-and-spoke: hubs on a ring with a few chords (2 hubs are gateways); every leaf homes on one hub."""
    G = nx.Graph()
    n_hub = max(3, int(round(n / (leaves_per_hub + 1))))
    hubs = [_add(G, "hub", "star") for _ in range(n_hub)]
    G.add_edges_from((hubs[i], hubs[(i + 1) % n_hub]) for i in range(n_hub))
    for _ in range(max(1, n_hub // 4)):
        a, b = rng.choice(hubs, 2, replace=False)
        G.add_edge(int(a), int(b))
    while G.number_of_nodes() < n:
        leaf = _add(G, "leaf", "star")
        k = 2 if rng.random() < p_dual_leaf else 1
        G.add_edges_from((leaf, int(h)) for h in rng.choice(hubs, k, replace=False))
    gw = [hubs[0], hubs[n_hub // 2]]
    return _finish(G, "star", gw, rng)


def _geometric(n, rng, mean_degree=6.0):
    """Connected random geometric graph in the unit square (components stitched to their nearest neighbour)."""
    pos = rng.random((n, 2))
    r = np.sqrt(mean_degree / (np.pi * n))
    G = nx.random_geometric_graph(n, r, pos={i: tuple(p) for i, p in enumerate(pos)})
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = set(comps[0])
    for c in comps[1:]:
        c = list(c); m = list(main)
        d = ((pos[c][:, None, :] - pos[m][None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        G.add_edge(c[i], m[j]); main |= set(c)
    return G, pos


def mesh(n, rng, mean_degree=6.0):
    """Random geometric mesh; the 4 nodes nearest the corners are gateways."""
    M, pos = _geometric(n, rng, mean_degree)
    G = nx.Graph()
    for v in M:
        G.add_node(v, tier="mesh", region="mesh")
    G.add_edges_from(M.edges)
    gw = [int(np.argmin(((pos - c) ** 2).sum(1))) for c in ([0, 0], [0, 1], [1, 0], [1, 1])]
    for g in gw:
        G.nodes[g]["tier"] = "gw"
    return _finish(G, "mesh", sorted(set(gw)), rng)


def hybrid(n, rng, region_size=30):
    """Hierarchical core/agg; under each agg an access region that is a star (hub + leaves), a mesh (small
    geometric graph attached at 2 nodes) or a hierarchical subtree (edge -> cell), chosen uniformly."""
    G = nx.Graph()
    cores = [_add(G, "core", "hierarchical") for _ in range(4)]
    G.add_edges_from((a, b) for i, a in enumerate(cores) for b in cores[i + 1:])
    n_agg = max(3, int(round(n / (region_size + 1))))
    for _ in range(n_agg):
        a = _add(G, "agg", "hierarchical")
        G.add_edges_from((a, int(c)) for c in rng.choice(cores, 2, replace=False))
        kind = rng.choice(["star", "mesh", "hierarchical"])
        if kind == "star":
            h = _add(G, "hub", "star"); G.add_edge(a, h)
            for _ in range(region_size - 1):
                G.add_edge(_add(G, "leaf", "star"), h)
        elif kind == "mesh":
            M, _ = _geometric(region_size, rng, 5.0)
            ids = {v: _add(G, "mesh", "mesh") for v in M}
            G.add_edges_from((ids[u], ids[v]) for u, v in M.edges)
            for v in rng.choice(list(ids.values()), 2, replace=False):
                G.add_edge(a, int(v))
        else:
            es = [_add(G, "edge", "hierarchical") for _ in range(max(2, region_size // 6))]
            G.add_edges_from((a, e) for e in es)
            for _ in range(region_size - len(es)):
                G.add_edge(_add(G, "cell", "hierarchical"), int(rng.choice(es)))
    return _finish(G, "hybrid", cores, rng)


def make_twin(kind, n, seed):
    rng = np.random.default_rng(seed)
    return {"hierarchical": hierarchical, "star": star, "mesh": mesh, "hybrid": hybrid}[kind](n, rng)
