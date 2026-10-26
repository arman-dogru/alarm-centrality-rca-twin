# Method

## Twin
Four seeded generators in [`rcatwin/twin.py`](../rcatwin/twin.py), each about 1,000 nodes by default:

| kind | structure | gateways |
|---|---|---|
| hierarchical | 4 cores (full mesh) -> aggregation (dual-homed to 2 cores) -> edge (20% dual-homed) -> cells | cores |
| star | hubs on a ring with chords; about 50 single-homed leaves per hub | 2 hubs |
| mesh | random geometric graph, mean degree about 6, stitched to be connected | 4 corner nodes |
| hybrid | hierarchical core/aggregation; under each aggregation node, a 30-node star, mesh or tree access region | cores |

Each node has a `region` label (hierarchical / star / mesh). It is the ground truth for triage: the region of
the root's location.

## Faults and alarms
See [`rcatwin/faults.py`](../rcatwin/faults.py). When the root fails:

- **Neighbours** raise LINK_DOWN (p = 0.9, after a 2 s hold-down).
- **Disconnected nodes** raise UNREACHABLE (p = 0.85). These are nodes with no remaining path to any gateway; they are always found by polling, with a 1 s per-hop delay.
- **Rerouted nodes** raise DEGRADED (p = 0.3). These are nodes whose distance to a gateway grew.
- **The root** raises NODE_DOWN unless it is a silent root.
- **Spurious alarms:** Poisson(noise × #true alarms), on uniform random nodes, at times uniform in [-60, 300] s.
- **Delivery:** an alarm arrives as a trap (50%; exponential latency, mean 2 s) or at the node's next poll (60 s period, random phase, |N(0, 5 s)| jitter).

## Methods
See [`rcatwin/rca.py`](../rcatwin/rca.py).

**Cascade.** Each stage narrows the candidate set:
- Stage 0: the 2-hop neighbourhood of the alarming nodes.
- Stage 1: its connected components; drop those with fewer than 2 alarms.
- Stage 2: Louvain communities inside each kept component; keep the most-alarmed communities until they cover 80% of that component's alarms.

**Alarm-conditioned trio**, computed on the stage-2 subgraph:
- PPR with restart on the alarming nodes.
- Alarm-degree: the number of alarming neighbours, plus 0.5 if the node alarms itself.
- Alarm-closeness: 1 / (1 + mean hop distance to the alarming nodes).

**Ensembles:** percentile-rank averages of the trio. Variants are equal weights, or 0.6 on the member a topology emphasises and 0.2 on the others.

**Triage** (see [`rcatwin/triage.py`](../rcatwin/triage.py)). The agent computes these features: hub share, cycle rank, degree CV, layer depth, diameter and clustering. It then labels the subgraph star, mesh or hierarchical, using either a two-threshold rule or a depth-3 tree (both calibrated on dev). It maps the label to a trio member or weighting using one of two mappings:
- the **blog mapping**: hierarchical -> PPR, star -> alarm-degree, mesh -> alarm-closeness;
- the **dev-tuned mapping**: the best member per label on the dev twins.

**Baselines:**
- earliest alarm (raw, and restricted to the cascade candidates);
- common ancestor (upstream walk, deepest node covering at least half the best coverage);
- static PageRank, degree and closeness on the same candidates;
- global PageRank;
- a random candidate.

**Oracles** (upper bounds, not methods):
- the true topology label with its weighting;
- the best trio member per incident.

## Metrics and statistics
- **Ranking:** the rank of the true root among all twin nodes, with pessimistic ties (a tie counts against the root). A root outside the candidate set is ranked last.
- **Reported metrics:** Top-1, Top-5 and MRR.
- **CIs:** 95% cluster bootstrap (B = 2000). It resamples twins (seeds) with replacement, then incidents within each drawn twin ([`rcatwin/stats.py`](../rcatwin/stats.py)).
- **Paired comparisons:** the planned list is in [`scripts/aggregate.py`](../scripts/aggregate.py). Each one reports:
  - the Top-1 difference, with the bootstrap CI computed on the same resamples for both methods;
  - a two-sided incident-level sign-flip p (5,000 flips), Holm-corrected over the 40 per-topology comparisons.
- **Significance rule:** a difference counts as significant only when Holm p < 0.05 **and** the CI excludes 0. The sign-flip treats the twins as fixed; the bootstrap does not.
