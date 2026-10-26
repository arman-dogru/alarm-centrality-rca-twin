"""Fault injection and alarm propagation on a twin.

A root node r fails at t0 = 0. The network is re-evaluated without r:
  * neighbours of r raise LINK_DOWN (probability p_link) after a loss-of-signal hold-down (holddown s);
  * nodes that lose every path to a gateway raise UNREACHABLE (probability p_unreach), detected by the
    poller after the propagation delay (hop_latency per hop below r);
  * nodes that stay reachable but whose gateway distance grows raise DEGRADED (probability p_degrade);
  * r itself raises NODE_DOWN unless the incident is a *silent root* (probability p_silent);
  * spurious alarms: Poisson(noise * #true alarms) on uniformly random nodes at uniform times in [-60, 300] s.
Delivery: each true alarm is a trap (exponential latency, mean trap_latency) with probability p_trap, else it is
seen at the node's next poll (period poll_interval, phase = node attribute poll) plus Gaussian jitter.
UNREACHABLE is always poll-detected (a disconnected device cannot send traps).
"""
from dataclasses import dataclass, field
import numpy as np

from .twin import layers


@dataclass
class FaultParams:
    p_silent: float = 0.0
    noise: float = 0.25
    p_link: float = 0.9
    p_unreach: float = 0.85
    p_degrade: float = 0.3
    p_trap: float = 0.5
    trap_latency: float = 2.0
    poll_interval: float = 60.0
    jitter: float = 5.0
    hop_latency: float = 1.0
    holddown: float = 2.0


@dataclass
class Incident:
    root: int
    alarms: dict                    # node -> first alarm time (s)
    kinds: dict                     # node -> alarm type of the first alarm
    true_alarms: set = field(default_factory=set)   # nodes whose alarm is caused by the fault
    disconnected: set = field(default_factory=set)  # blast radius: nodes cut off from every gateway
    degraded: set = field(default_factory=set)


def impact(G, root):
    """(disconnected, degraded) node sets when `root` fails."""
    L0 = G.graph["layer"]
    L1 = layers(G, G.graph["gateways"], removed=[root])
    others = [v for v in G if v != root]
    disc = {v for v in others if v not in L1}
    deg = {v for v in others if v in L1 and L1[v] > L0[v]}
    return disc, deg


def _deliver(G, v, rng, P, trap_ok=True, extra=0.0):
    if trap_ok and rng.random() < P.p_trap:
        return extra + rng.exponential(P.trap_latency)
    if P.poll_interval <= 0:
        return extra + abs(rng.normal(0, P.jitter))
    phase = G.nodes[v]["poll"] * P.poll_interval
    t = extra
    nxt = phase + P.poll_interval * np.ceil((t - phase) / P.poll_interval)
    return float(nxt + abs(rng.normal(0, P.jitter)))


def simulate(G, root, P, rng):
    disc, deg = impact(G, root)
    L0 = G.graph["layer"]
    alarms, kinds = {}, {}

    def emit(v, t, k):
        if v not in alarms or t < alarms[v]:
            alarms[v], kinds[v] = t, k

    if rng.random() >= P.p_silent:
        emit(root, _deliver(G, root, rng, P), "NODE_DOWN")
    for u in G.neighbors(root):
        if rng.random() < P.p_link:
            emit(u, _deliver(G, u, rng, P, extra=P.holddown), "LINK_DOWN")
    for v in disc:
        if rng.random() < P.p_unreach:
            hops = max(1, L0[v] - L0[root])
            emit(v, _deliver(G, v, rng, P, trap_ok=False, extra=P.hop_latency * hops), "UNREACHABLE")
    for v in deg:
        if rng.random() < P.p_degrade:
            emit(v, _deliver(G, v, rng, P, extra=P.hop_latency), "DEGRADED")
    true = set(alarms)
    nodes = np.array(list(G.nodes))
    for v in rng.choice(nodes, rng.poisson(P.noise * max(1, len(true)))):
        emit(int(v), float(rng.uniform(-60, 300)), "SPURIOUS")
    return Incident(root, alarms, kinds, true, disc, deg)


def eligible_roots(G, min_blast=3):
    """Nodes whose failure touches >= min_blast other nodes (neighbours + disconnected): real incidents."""
    out = []
    for v in G:
        disc, _ = impact(G, v)
        if len(set(G.neighbors(v)) | disc) >= min_blast:
            out.append(v)
    return out
