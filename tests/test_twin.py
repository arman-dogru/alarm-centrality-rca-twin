import networkx as nx
import numpy as np
import pytest

from rcatwin.twin import KINDS, make_twin, upstream
from rcatwin.faults import FaultParams, eligible_roots, impact, simulate, _deliver
from tests.helpers import ring_twin, tree_twin


@pytest.mark.parametrize("kind", KINDS)
def test_generators_connected_seeded(kind):
    G = make_twin(kind, 300, 3)
    assert nx.is_connected(G) and abs(G.number_of_nodes() - 300) <= 40
    assert all(G.graph["layer"][g] == 0 for g in G.graph["gateways"])
    assert {G.nodes[v]["region"] for v in G} <= {"hierarchical", "star", "mesh"}
    H = make_twin(kind, 300, 3)
    assert sorted(G.edges) == sorted(H.edges)  # same seed -> same twin
    assert sorted(G.edges) != sorted(make_twin(kind, 300, 4).edges)


@pytest.mark.parametrize("kind", KINDS)
def test_upstream_points_towards_gateways(kind):
    G = make_twin(kind, 200, 1)
    L = G.graph["layer"]
    D = upstream(G)
    assert D.number_of_edges() > 0 and all(L[v] == L[u] - 1 for u, v in D.edges)
    assert all(D.out_degree(v) >= 1 for v in G if L[v] > 0)  # every non-gateway has an upstream path


def test_impact_tree_and_ring():
    G = tree_twin()
    disc, deg = impact(G, 2)
    assert disc == {4, 5} and deg == set()
    assert impact(G, 1)[0] == {2, 3, 4, 5}
    R = ring_twin(8)
    disc, deg = impact(R, 1)
    assert disc == set() and deg == {2, 3}  # 2 and 3 now reach the gateway the long way round


def test_simulate_deterministic_full_observability():
    G = tree_twin()
    P = FaultParams(p_silent=0, noise=0, p_link=1, p_unreach=1, p_degrade=1, poll_interval=0, jitter=0, p_trap=0)
    inc = simulate(G, 2, P, np.random.default_rng(0))
    assert set(inc.alarms) == {1, 2, 4, 5}  # root, its upstream neighbour, the two disconnected leaves
    assert inc.kinds[2] == "NODE_DOWN" and inc.kinds[4] == "UNREACHABLE" and inc.kinds[1] == "LINK_DOWN"
    assert inc.alarms[2] < min(inc.alarms[v] for v in (1, 4, 5))  # ideal clocks: the cause alarms first
    silent = simulate(G, 2, FaultParams(**{**P.__dict__, "p_silent": 1.0}), np.random.default_rng(0))
    assert 2 not in silent.alarms and set(silent.alarms) == {1, 4, 5}


def test_noise_alarms_count():
    G = make_twin("mesh", 300, 1)
    P = FaultParams(noise=1.0, p_link=1, p_unreach=1, p_degrade=0, p_silent=0)
    n_spur = [sum(k == "SPURIOUS" for k in simulate(G, 10, P, np.random.default_rng(s)).kinds.values()) for s in range(40)]
    assert 3 < np.mean(n_spur) < 12  # Poisson(noise * #true alarms), true alarms ~ degree + 1


def test_poll_delivery_on_grid():
    G = tree_twin()
    P = FaultParams(p_trap=0, poll_interval=60, jitter=0)
    for v in G:
        t = _deliver(G, v, np.random.default_rng(1), P, extra=7.0)
        phase = G.nodes[v]["poll"] * 60
        assert t >= 7.0 and abs(((t - phase) / 60) - round((t - phase) / 60)) < 1e-9 and t - 7.0 < 60


def test_eligible_roots_excludes_leaves():
    G = tree_twin()
    assert set(eligible_roots(G)) == {0, 1, 2}  # gateway 0 cuts everything; leaves touch only their parent


def test_p_link_and_link_models():
    G = make_twin("mesh", 300, 1)
    root = eligible_roots(G)[5]
    base = dict(p_silent=1.0, noise=0.0, p_unreach=0.0, p_degrade=0.0)
    none = simulate(G, root, FaultParams(p_link=0.0, **base), np.random.default_rng(0))
    assert "LINK_DOWN" not in none.kinds.values()
    allk = simulate(G, root, FaultParams(p_link=1.0, **base), np.random.default_rng(0))
    assert set(allk.alarms) == set(G.neighbors(root))
    L = G.graph["layer"]
    up = simulate(G, root, FaultParams(p_link=1.0, link_model="upstream", **base), np.random.default_rng(0))
    assert set(up.alarms) == {u for u in G.neighbors(root) if L[u] < L[root]}
    half = [len(simulate(G, root, FaultParams(p_link=0.5, **base), np.random.default_rng(s)).alarms) for s in range(200)]
    assert abs(np.mean(half) / G.degree(root) - 0.5) < 0.08


def test_degraded_spurious_window_and_unreachable_depth():
    G = nx.path_graph(5)
    for v in G:
        G.nodes[v].update(tier="x", region="hierarchical")
    from rcatwin.twin import _finish
    G = _finish(G, "hierarchical", [0], np.random.default_rng(0))
    P = FaultParams(p_silent=1, noise=0, p_link=0, p_unreach=1, p_degrade=1, poll_interval=0, jitter=0, p_trap=0)
    inc = simulate(G, 1, P, np.random.default_rng(0))
    assert inc.alarms == {2: 1.0, 3: 2.0, 4: 3.0}  # UNREACHABLE delay grows with hops below the root
    assert inc.degraded == set() and "DEGRADED" not in inc.kinds.values()  # nothing reachable got longer paths
    R = ring_twin(8)
    no_deg = simulate(R, 1, FaultParams(p_silent=1, noise=0, p_link=0, p_degrade=0), np.random.default_rng(0))
    assert no_deg.alarms == {}
    deg = simulate(R, 1, FaultParams(p_silent=1, noise=0, p_link=0, p_degrade=1), np.random.default_rng(0))
    assert set(deg.alarms) == {2, 3}
    M = make_twin("mesh", 300, 1)
    ts = [t for s in range(30) for v, t in simulate(M, 10, FaultParams(noise=1.0), np.random.default_rng(s)).alarms.items()
          if v not in simulate(M, 10, FaultParams(noise=0.0), np.random.default_rng(s)).alarms]
    assert min(ts) < -30 and max(ts) > 250 and min(ts) >= -60 and max(ts) <= 300  # spurious window [-60, 300] s


def test_noise_rate_matches_parameter():
    G = make_twin("mesh", 300, 1)
    P = FaultParams(noise=0.5, p_link=1, p_unreach=1, p_degrade=0, p_silent=0)
    tot, spur = 0, 0
    for s in range(150):
        inc = simulate(G, 10, P, np.random.default_rng(s))
        tot += len(inc.true_alarms); spur += sum(k == "SPURIOUS" for k in inc.kinds.values())
    assert 0.35 < spur / tot < 0.6  # Poisson(0.5 x #true); collisions with true alarms only lower it


def test_eligible_roots_threshold_is_inclusive():
    G = tree_twin()
    assert set(eligible_roots(G, min_blast=3)) == {0, 1, 2}  # node 2 touches exactly 3 others
    assert set(eligible_roots(G, min_blast=4)) == {0, 1}
