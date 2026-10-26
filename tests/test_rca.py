import networkx as nx
import numpy as np
import pytest

from rcatwin import rca
from rcatwin.faults import Incident
from tests.helpers import tree_twin


def ctx_for(G, alarms):
    return rca.context(G, Incident(root=-1, alarms=dict(alarms), kinds={a: "X" for a in alarms}))


def test_rank_of_pessimistic_ties_and_neg_inf():
    s = np.array([0.5, 0.9, 0.5, -np.inf])
    assert rca.rank_of(s, 1) == 1
    assert rca.rank_of(s, 0) == 3 and rca.rank_of(s, 2) == 3  # tie of two: both get the worse rank
    assert rca.rank_of(s, 3) == 4


def test_ppr_matches_networkx_and_is_a_distribution():
    G = nx.karate_club_graph()
    nodes = list(G)
    M, _ = rca._adj(G, nodes)
    pers = np.zeros(len(nodes)); pers[[0, 5, 33]] = 1
    x = rca.ppr(M, pers, alpha=0.85)
    ref = nx.pagerank(G, alpha=0.85, personalization={0: 1, 5: 1, 33: 1}, tol=1e-12, weight=None)
    assert abs(x.sum() - 1) < 1e-9 and np.allclose(x, [ref[v] for v in nodes], atol=1e-6)


def star_with_gateway():
    """hub 0 with leaves 1..5; gateway-like node 6 with 50 quiet neighbours 7..56, linked to the hub."""
    G = nx.star_graph(5)
    G.add_edges_from((6, v) for v in range(7, 57))
    G.add_edge(0, 6)
    G.graph["layer"] = nx.single_source_shortest_path_length(G, 6)
    return G


def test_alarm_degree_counts_only_alarming_neighbours():
    """Blog example: a gateway with 50 links but none to alarming nodes scores 0; a switch with 5 alarm links scores 5."""
    G = star_with_gateway()
    c = ctx_for(G, {v: 0.0 for v in range(1, 6)})
    s = rca.trio(c, list(G))["alarm_degree"]
    assert s[0] == 5 and s[6] == 0 and np.argmax(s) == 0


def test_alarm_closeness_prefers_centre_of_alarms():
    G = nx.path_graph(9)  # alarms at 2, 4, 6 -> node 4 is closest on average
    G.graph["layer"] = nx.single_source_shortest_path_length(G, 0)
    s = rca.trio(ctx_for(G, {2: 0.0, 4: 0.0, 6: 0.0}), list(G))["alarm_closeness"]
    assert np.argmax(s) == 4 and s[4] > s[3] > s[1] > s[0]
    assert s[3] == s[5] and s[2] == s[6]


def test_ppr_seeded_on_alarms_finds_hub():
    G = star_with_gateway()
    s = rca.trio(ctx_for(G, {v: 0.0 for v in range(1, 6)}), list(G))["ppr"]
    assert np.argmax(s) == 0
    s_dir = rca.trio(ctx_for(G, {v: 0.0 for v in range(1, 6)}), list(G), directed=True)["ppr"]
    assert np.isfinite(s_dir[list(G)]).all()


def test_static_centralities_ignore_alarms():
    G = star_with_gateway()
    a = rca.static_centralities(ctx_for(G, {1: 0.0, 2: 0.0}), list(G))
    b = rca.static_centralities(ctx_for(G, {40: 0.0}), list(G))
    for k in a:
        assert np.allclose(a[k], b[k])
    assert np.argmax(a["static_degree"]) == 6  # the big quiet gateway wins statically


def test_cascade_decomposition_drops_isolated_noise():
    G = nx.disjoint_union(nx.star_graph(6), nx.path_graph(30))  # star 0..6, path 7..36
    G.graph["layer"] = {v: 0 for v in G}
    c = ctx_for(G, {1: 0, 2: 0, 3: 0, 4: 0, 30: 0})  # 4 star alarms + 1 isolated noise alarm on the path
    nodes, st = rca.cascade(c, hops=2, cluster=False)
    assert set(nodes) == set(range(7)) and st["s0"] > st["s1"]
    nd, _ = rca.cascade(c, hops=2, decompose=False)
    assert 30 in nd
    one, _ = rca.cascade(ctx_for(G, {30: 0}), hops=1, cluster=False)
    assert set(one) == {29, 30, 31}  # nothing has >= 2 alarms: keep the component with the most


def test_cascade_cluster_keeps_alarm_communities():
    G = nx.connected_caveman_graph(6, 8)  # 6 cliques of 8 in a ring
    G.graph["layer"] = {v: 0 for v in G}
    c = ctx_for(G, {v: 0 for v in range(0, 6)})  # alarms inside clique 0
    nodes, st = rca.cascade(c, hops=2)
    assert set(range(8)) <= set(nodes) and st["s2"] < st["s1"]


def test_common_ancestor_is_lowest_covering_node():
    G = tree_twin()  # 0-1-{2-{4,5},3}
    c = ctx_for(G, {4: 0.0, 5: 0.0, 1: 0.0})
    assert np.argmax(rca.common_ancestor(c, frac=0.6)) == 2  # covers 4 and 5 (2/3), deepest such node
    assert np.argmax(rca.common_ancestor(c, frac=1.0)) == 1  # only 1 (and 0) cover all three; 1 is deeper


def test_earliest_alarm_and_in_cascade():
    G = tree_twin()
    c = ctx_for(G, {4: 5.0, 1: 3.0, 3: -10.0})
    assert np.argmax(rca.earliest_alarm(c)) == 3
    s = rca.earliest_in(c, [1, 2, 4])
    assert np.argmax(s) == 1 and s[3] == -np.inf


def test_weighted_is_percentile_average():
    tr = {"ppr": np.array([0.1, 0.3, 0.2, -np.inf]), "alarm_degree": np.array([3.0, 1.0, 2.0, -np.inf]),
          "alarm_closeness": np.array([0.0, 0.0, 1.0, -np.inf])}
    w = rca.weighted(tr, [0, 1, 2], {"ppr": 1, "alarm_degree": 0, "alarm_closeness": 0})
    assert list(np.argsort(w[:3])) == [0, 2, 1] and w[3] == -np.inf
    e = rca.weighted(tr, [0, 1, 2], {k: 1 / 3 for k in tr})
    assert np.argmax(e) == 2
