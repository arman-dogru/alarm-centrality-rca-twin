# Pilots and design decisions (dev twins only)

Every decision below was made on **dev twins (seeds 1-5)** or on a single seed-1 twin before the evaluation
grid ran. The evaluation twins (seeds 100-109) were generated only by the final grid. Calibration that uses
data is done by [`scripts/aggregate.py`](../scripts/aggregate.py) on `results/raw/dev/` only, and its output
is committed in [`results/calibration.json`](../results/calibration.json).

1. **PPR walk direction and damping.** The first version walked only *upstream* dependency edges
   (child -> parent), so that probability mass would collect on common ancestors. On a seed-1 twin
   (60 incidents per topology, noise 0.5, silent 0.5) it was far worse than an undirected walk. The upstream
   neighbours of a failed node raise LINK_DOWN alarms themselves, so mass flows past the root to its parent.
   Top-1 accuracy, directed vs undirected (alpha 0.85):

   | topology | directed | undirected |
   |---|---|---|
   | hierarchical | 0.07 | 0.88 |
   | star | 0.72 | 1.00 |
   | mesh | 0.07 | 0.17 |
   | hybrid | 0.10 | 0.40 |

   Degree-normalised PPR was worse everywhere. **Decision:** undirected PPR with alpha 0.85 (networkx's
   default) is the main variant. Directed and alpha 0.5 / 0.95 are kept as ablations in the `ablate` phase.
2. **Temporal baseline.** Spurious alarms are spread over [-60, 300] s, so the raw earliest alarm is almost
   always a spurious alarm that fired *before* the fault. To give the temporal baseline a fair chance,
   `earliest_in_cascade` takes the earliest alarm among the cascade's candidates (both variants are reported).
   A 2 s loss-of-signal hold-down on LINK_DOWN was added so that, with ideal clocks, the root's own alarm
   comes first. The `poll` phase includes an ideal-clock condition (no polling, no jitter, no noise) as the
   temporal baseline's best case.
3. **Common-ancestor baseline.** A plain "most alarms in the downstream cone" score always picks the gateway,
   because every alarm is downstream of it. The baseline was changed to the *deepest* node covering at least
   `frac` of the best coverage. Four fractions are computed online, and the dev twins choose one
   (`calibration.json`; 0.5 was chosen).
4. **Triage is applied offline.** The grid stores the rank under every candidate weighting (`w_hierarchical`,
   `w_star`, `w_mesh`) and the triage features. The triage agent's choice (rules, tree, mapping) is therefore
   applied after the grid, from dev-calibrated parameters, without re-running anything. The triage rules
   use `featctx` (candidate set plus 1 hop) or `feat` (candidate set only), whichever scores better on dev.
5. **Root sampling.** Roots are drawn uniformly from nodes whose failure touches at least 3 other nodes
   (neighbours plus disconnected nodes). Leaves are excluded because their failure is a single trivial alarm.
   In star twins this leaves only the hubs, which is why star is easy for every structural method.
6. **Ties (found after the first evaluation pass, diagnosed on a dev twin).** Alarm-degree is integer-valued
   (+0.5 when the node alarms itself). In the clustered random-geometric meshes, a neighbour of the root
   that sees most of the root's other alarming neighbours *and* the root's own alarm often ties with the
   root. Ties count against the root (pessimistic ranking), so in mesh twins alarm-degree *improves* as the
   root goes silent, because the neighbours lose that one extra alarm. Six seed-1 dev incidents showed root
   ranks of 2-6 at tied scores. In response I made two changes and re-ran the whole grid:
   - every method now also stores its optimistic rank, so `top1_ties_split` (the expected Top-1 under random
     tie-breaking) is reported alongside the pessimistic Top-1;
   - `alarm_degree_tb` (alarm-degree with alarm-closeness as a tie-break) was added. It is labelled
     exploratory: it is not in the planned comparison family or the Holm correction.
