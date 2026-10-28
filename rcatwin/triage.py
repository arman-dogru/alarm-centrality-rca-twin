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
# Which trio member each topology emphasises. This is MY operationalisation of the Amazon Science post
# (https://www.amazon.science/blog/graph-centric-agentic-intelligence): the post says the agent selects the
# algorithm combination according to topology; the hard label -> member mapping and the 0.6/0.2 weights are my choice.
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


def classify(cycle_rank, deg_cv, cycle_mesh_min, deg_cv_star_min):
    """The triage rule used by the pipeline (vectorised): cycle-rich -> mesh, else very uneven degrees -> star, else
    hierarchical. Thresholds are grid-searched on the dev twins only (scripts/aggregate.py: calibrate)."""
    cyc, cv = np.asarray(cycle_rank, float), np.asarray(deg_cv, float)
    return np.where(cyc >= cycle_mesh_min, "mesh", np.where(cv >= deg_cv_star_min, "star", "hierarchical"))


def true_label(G, root):
    return G.nodes[root]["region"]
