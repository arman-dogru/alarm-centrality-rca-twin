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
