"""Topology triage agent (rule-based, no LLM): classify the affected subgraph as hierarchical / star / mesh from
graph metrics, then pick or weight the alarm-conditioned trio.

Features of the candidate subgraph S (after the cascade):
  hub_share   max degree / (|S| - 1)            (star: one node adjacent to almost everything)
  cycle_rank  (|E| - |V| + #components) / |V|   (0 for trees; meshes are cycle-rich)
  depth       max - min gateway layer within S  (hierarchy depth)
  diameter    of the largest component (double-sweep lower bound)
  deg_cv      coefficient of variation of degrees
  clustering  average clustering coefficient
"""
import numpy as np
import networkx as nx

LABELS = ("hierarchical", "star", "mesh")
# which trio member each topology emphasises (Amazon Science post: PPR -> common ancestors in hierarchies,
# alarm-degree -> star hubs, alarm-closeness -> peripheral failures in meshes)
SELECT = {"hierarchical": "ppr", "star": "alarm_degree", "mesh": "alarm_closeness"}
WEIGHTS = {lab: {k: (0.6 if k == SELECT[lab] else 0.2) for k in SELECT.values()} for lab in LABELS}
EQUAL = {k: 1 / 3 for k in SELECT.values()}


def features(G, nodes):
    H = G.subgraph(nodes)
    n, m = H.number_of_nodes(), H.number_of_edges()
    deg = np.array([d for _, d in H.degree()], float) if n else np.zeros(1)
    comps = nx.number_connected_components(H) if n else 0
    L = G.graph["layer"]
    lay = [L[v] for v in nodes]
    big = max(nx.connected_components(H), key=len) if n else set()
    diam = 0
    if len(big) > 1:
        Hb = H.subgraph(big)
        far = max(nx.single_source_shortest_path_length(Hb, next(iter(big))).items(), key=lambda kv: kv[1])[0]
        diam = max(nx.single_source_shortest_path_length(Hb, far).values())
    return {"n": n, "hub_share": float(deg.max() / max(1, n - 1)), "cycle_rank": (m - n + comps) / max(1, n),
            "depth": int(max(lay) - min(lay)) if lay else 0, "diameter": int(diam),
            "deg_cv": float(deg.std() / max(1e-9, deg.mean())), "clustering": float(nx.average_clustering(H)) if n else 0.0}


def classify(f, hub_share_min=0.5, cycle_mesh_min=0.15):
    """Rules (thresholds fixed on the dev seeds, see scripts/calibrate_triage.py)."""
    if f["hub_share"] >= hub_share_min:
        return "star"
    if f["cycle_rank"] >= cycle_mesh_min:
        return "mesh"
    return "hierarchical"


def true_label(G, root):
    return G.nodes[root]["region"]
