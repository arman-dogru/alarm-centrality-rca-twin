"""One incident -> every method's rank of the true root, plus cascade, triage and confidence diagnostics."""
import time
import numpy as np

from . import rca, triage
from .faults import FaultParams, impact, simulate

TRIO = ("ppr", "alarm_degree", "alarm_closeness")


def confidence(s, nodes, trio_scores):
    """(margin between the top-2 ensemble scores, #trio members whose top-1 equals the ensemble top-1, top-1)."""
    v = np.sort(s[list(nodes)])[::-1]
    top = int(np.argmax(s))
    agree = sum(int(np.argmax(trio_scores[k])) == top for k in TRIO)
    return float(v[0] - v[1]) if len(v) > 1 else 1.0, agree, top


def run_incident(G, root, P, rng, ablations=False, seed=0):
    out = {"root": int(root), "root_tier": G.nodes[root]["tier"], "true_label": triage.true_label(G, root)}
    t0 = time.perf_counter()
    inc = simulate(G, root, P, rng)
    ctx = rca.context(G, inc)
    out.update(n_alarms=len(ctx.A), n_true_alarms=len(inc.true_alarms), root_alarmed=int(root in inc.alarms),
               blast=len(inc.disconnected))
    if len(ctx.A) == 0:  # nothing observable (possible with a silent root and missed alarms)
        out["no_alarms"] = 1
        return out
    out["no_alarms"] = 0
    t1 = time.perf_counter()
    nodes, st = rca.cascade(ctx, seed=seed)
    t2 = time.perf_counter()
    tr = rca.trio(ctx, nodes)
    t3 = time.perf_counter()
    f = triage.features(G, nodes)
    f_ctx = triage.features(G, sorted(rca.neighbourhood(G, nodes, 1)))
    S = dict(tr)
    S["equal"] = rca.weighted(tr, nodes, triage.EQUAL)
    for lab in triage.LABELS:  # the triage agent's choice is applied offline (scripts/aggregate.py) from features
        S[f"w_{lab}"] = rca.weighted(tr, nodes, triage.WEIGHTS[lab])
    S.update(rca.static_centralities(ctx, nodes))
    S["earliest_alarm"] = rca.earliest_alarm(ctx)
    S["earliest_in_cascade"] = rca.earliest_in(ctx, nodes)
    for fr in (0.3, 0.5, 0.7, 0.9):  # frac is chosen on the dev twins (scripts/aggregate.py)
        S[f"common_ancestor_{fr:g}"] = rca.common_ancestor(ctx, fr)
    S["global_pagerank"] = rca.global_pagerank(G)
    S["random_candidate"] = rca.random_candidates(ctx, nodes, rng)
    t4 = time.perf_counter()
    for k, s in S.items():
        out[f"rank:{k}"] = rca.rank_of(s, root)
    out.update({f"stage:{k}": v for k, v in st.items()})
    out["root_in_s0"] = int(root in rca.neighbourhood(G, ctx.A, 2))
    out["root_in_s2"] = int(root in set(nodes))
    out.update({f"feat:{k}": v for k, v in f.items()})
    out.update({f"featctx:{k}": v for k, v in f_ctx.items()})
    for ens in ("equal", *TRIO, *(f"w_{lab}" for lab in triage.LABELS)):
        m, a, top = confidence(S[ens], nodes, tr)
        out[f"conf_margin:{ens}"], out[f"conf_agree:{ens}"] = m, a
        out[f"top1_blast:{ens}"] = len(impact(G, top)[0])
    out.update(ms_simulate=1e3 * (t1 - t0), ms_cascade=1e3 * (t2 - t1), ms_trio=1e3 * (t3 - t2), ms_rest=1e3 * (t4 - t3))
    if ablations:
        abl = {}
        for name, kw in {"no_decompose": dict(decompose=False), "no_cluster": dict(cluster=False),
                         "hops1": dict(hops=1), "hops3": dict(hops=3)}.items():
            nd, _ = rca.cascade(ctx, seed=seed, **kw)
            t = rca.trio(ctx, nd)
            t["equal"] = rca.weighted(t, nd, triage.EQUAL)
            for k, s in t.items():
                abl[f"rank:{k}@{name}"] = rca.rank_of(s, root)
            abl[f"size@{name}"] = len(nd)
        for name, kw in {"ppr_directed": dict(directed=True), "ppr_alpha50": dict(alpha=0.5), "ppr_alpha95": dict(alpha=0.95)}.items():
            abl[f"rank:ppr@{name}"] = rca.rank_of(rca.trio(ctx, nodes, **kw)["ppr"], root)
        out.update(abl)
    return out
