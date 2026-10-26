"""rcatwin/experiment.py: stored ranks, the exploratory tie-break, the autonomy confidence and the PR cache."""
import networkx as nx
import numpy as np

from rcatwin import rca
from rcatwin.experiment import confidence, run_incident, unique_top
from rcatwin.faults import FaultParams, eligible_roots
from rcatwin.twin import make_twin


def _incidents(kind="mesh", n=200, k=25, **kw):
    G = make_twin(kind, n, 2)
    roots = eligible_roots(G)
    P = FaultParams(**{"p_silent": 0.0, "noise": 0.25, **kw})
    return G, [run_incident(G, int(roots[i % len(roots)]), P, np.random.default_rng(i)) for i in range(k)]


def test_run_incident_ranks_pessimistic_and_tie_break_only_within_ties():
    G, out = _incidents()
    methods = [c[5:] for c in out[0] if c.startswith("rank:")]
    assert {"ppr", "alarm_degree", "alarm_closeness", "equal", "alarm_degree_tb", "global_pagerank"} <= set(methods)
    strict = 0
    for o in out:
        if o["no_alarms"]:
            continue
        for m in methods:
            assert 1 <= o[f"rankopt:{m}"] <= o[f"rank:{m}"] <= G.number_of_nodes()
        strict += o["rank:alarm_degree"] > o["rankopt:alarm_degree"]
        # the closeness tie-break may only reorder nodes inside an alarm-degree tie
        assert o["rankopt:alarm_degree"] <= o["rank:alarm_degree_tb"] <= o["rank:alarm_degree"]
        assert o["rankopt:alarm_degree"] <= o["rankopt:alarm_degree_tb"] <= o["rank:alarm_degree"]
    assert strict > 0  # mesh incidents with a root alarm do produce alarm-degree ties: stored ranks are pessimistic


def test_unique_top_and_confidence():
    assert unique_top(np.array([0.1, 0.9, 0.3])) == 1 and unique_top(np.array([0.9, 0.9, 0.3])) is None
    nodes = [0, 1, 2]
    s = np.array([0.2, 0.9, 0.1])
    trio = {"ppr": np.array([0.1, 0.8, 0.3]), "alarm_degree": np.array([2.0, 2.0, 0.0]), "alarm_closeness": np.array([0.0, 0.5, 0.4])}
    margin, agree, top = confidence(s, nodes, trio)
    assert top == 1 and abs(margin - 0.7) < 1e-12 and agree == 2  # tied alarm-degree top does not count
    assert confidence(np.array([0.9, 0.9, 0.1]), nodes, trio)[1] == 0  # tied ensemble top: no agreement


def test_global_pagerank_cache_is_per_twin():
    a, b = make_twin("mesh", 200, 1), make_twin("mesh", 300, 2)
    pa, pb = rca.global_pagerank(a), rca.global_pagerank(b)
    assert len(pa) == a.number_of_nodes() and len(pb) == b.number_of_nodes()
    assert np.allclose(rca.global_pagerank(a), pa)
    g = nx.path_graph(5)  # no twin_id: computed fresh every time, never served from the cache
    assert len(rca.global_pagerank(g)) == 5 and len(rca.global_pagerank(nx.path_graph(7))) == 7
