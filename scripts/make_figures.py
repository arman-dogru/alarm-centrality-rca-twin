"""All result figures from results/summary.json and results/<phase>_incidents.csv.gz -> docs/figures/*.png"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # every path below is relative to the repo root, wherever the script is run from
from rcatwin import rca, triage
from rcatwin.faults import FaultParams, eligible_roots, simulate
from rcatwin.twin import make_twin

FIG = "docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("results/summary.json"))
if "partial" in S or not {"main", "plink", "poll", "ablate", "size"} <= set(S.get("phases", {})):
    sys.exit("make_figures.py: refusing a partial or incomplete results/summary.json (re-run aggregate.py on the full grid)")
KINDS = ["hierarchical", "star", "mesh", "hybrid"]
NAMES = {"random_candidate": "random (cascade candidates)", "global_pagerank": "global PageRank (static)",
         "earliest_alarm": "earliest alarm", "earliest_in_cascade": "earliest alarm in cascade",
         "common_ancestor": "common ancestor (upstream walk)", "static_pagerank": "static PageRank (cascade)",
         "static_degree": "static degree (cascade)", "static_closeness": "static closeness (cascade)",
         "ppr": "alarm-seeded PPR", "alarm_degree": "alarm-degree", "alarm_closeness": "alarm-closeness",
         "alarm_degree_tb": "alarm-degree, closeness tie-break*",
         "equal": "trio, equal weights", "triage_blog_select": "triage: blog mapping, select",
         "triage_blog_weighted": "triage: blog mapping, weighted", "triage_dev_select": "triage: dev-tuned mapping",
         "triage_tree_weighted": "triage: learned tree, weighted", "oracle_label_weighted": "oracle topology label, weighted",
         "oracle_incident": "oracle: best trio member per incident"}
GROUP = {m: "baseline" for m in list(NAMES)[:8]}
GROUP.update({m: "alarm-conditioned" for m in ("ppr", "alarm_degree", "alarm_closeness", "alarm_degree_tb", "equal")})
GROUP.update({m: "triage" for m in NAMES if m.startswith("triage")})
GROUP.update({m: "oracle" for m in NAMES if m.startswith("oracle")})
COL = {"baseline": "#8c8c8c", "alarm-conditioned": "#1f77b4", "triage": "#ff7f0e", "oracle": "#2ca02c"}
main = S["phases"]["main"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=120); plt.close(fig); print(name)


# 1. headline: Top-1 per method x topology --------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
meths = list(NAMES)
for ax, k in zip(axes, KINDS):
    g = main[k]["methods"]
    y = np.arange(len(meths))
    for yi, m in zip(y, meths):
        v = g[m]["top1"]
        ax.errorbar(100 * v["mean"], yi, xerr=[[100 * (v["mean"] - v["lo"])], [100 * (v["hi"] - v["mean"])]], fmt="o",
                    color=COL[GROUP[m]], capsize=2, ms=5)
        ax.text(100 * v["hi"] + 2, yi, f"{100 * v['mean']:.0f}", va="center", fontsize=7)
    ax.set_xlim(0, 112); ax.set_title(f"{k} (n={main[k]['n']:,}, {main[k]['twins']} twins)")
    ax.set_xlabel("Top-1 root-cause accuracy (%)"); ax.grid(axis="x", alpha=0.3)
axes[0].set_yticks(y, [NAMES[m] for m in meths]); axes[0].invert_yaxis()
for g, c in COL.items():
    axes[-1].plot([], [], "o", color=c, label=g)
fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.suptitle("Top-1 accuracy over the main grid (4 noise levels x 3 silent-root rates), 95% twin-clustered bootstrap CI; ties count against the root; * exploratory")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{FIG}/top1_by_method.png", dpi=120); plt.close(fig); print("top1_by_method")

# 2. robustness: Top-1 vs noise and vs silent-root rate ----------------------------------------------
LINES = ["earliest_in_cascade", "common_ancestor", "static_closeness", "ppr", "alarm_degree", "alarm_closeness", "equal", "triage_dev_select"]
for axis, label, fname in (("noise", "spurious alarms per true alarm", "top1_vs_noise"),
                           ("silent", "silent-root probability", "top1_vs_silent")):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, KINDS):
        keys = sorted([g for g in main if g.startswith(f"{k}|{axis}=")], key=lambda s: float(s.split("=")[1]))
        xs = [float(g.split("=")[1]) for g in keys]
        for m in LINES:
            v = [main[g]["methods"][m]["top1"] for g in keys]
            ax.plot(xs, [100 * a["mean"] for a in v], "o-", label=NAMES[m], color=plt.cm.tab10(LINES.index(m)), ms=3)
            ax.fill_between(xs, [100 * a["lo"] for a in v], [100 * a["hi"] for a in v], alpha=0.12, color=plt.cm.tab10(LINES.index(m)))
        ax.set_title(k); ax.set_xlabel(label); ax.set_ylim(0, 102); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 (%)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, fname)

# 3. cascade funnel + stage ablations -----------------------------------------------------------------
inc = pd.read_csv("results/main_incidents.csv.gz")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(4)
for i, (col, lab) in enumerate((("stage:s0", "stage 0: alarm 2-hop neighbourhood"), ("stage:s1", "stage 1: + decomposition (WCC)"),
                                ("stage:s2", "stage 2: + Louvain clustering"))):
    med = [inc[inc.kind == k][col].median() for k in KINDS]
    axes[0].bar(x + (i - 1) * 0.27, med, 0.27, label=lab)
axes[0].set_yscale("log"); axes[0].set_xticks(x, KINDS); axes[0].set_ylabel("candidate nodes (median, log)")
axes[0].axhline(1000, color="k", ls=":", lw=0.8); axes[0].text(3.4, 1050, "twin size", fontsize=7, ha="right")
axes[0].legend(fontsize=7); axes[0].set_title("cascade narrows the search space")
ret = [100 * inc[inc.kind == k]["root_in_s2"].mean() for k in KINDS]
axes[0].set_xlabel("root kept after stage 2: " + ", ".join(f"{k} {r:.1f}%" for k, r in zip(KINDS, ret)), fontsize=7)
if "ablate" in S["phases"]:
    A = S["phases"]["ablate"]
    vars_ = [("", "full cascade"), ("@no_cluster", "no clustering"), ("@no_decompose", "no decomposition/clustering"),
             ("@hops1", "1-hop neighbourhood"), ("@hops3", "3-hop neighbourhood")]
    for j, (suf, lab) in enumerate(vars_):
        vals = [A[k]["methods"][f"equal{suf}"]["top1"] for k in KINDS]
        axes[1].bar(x + (j - 2) * 0.16, [100 * v["mean"] for v in vals], 0.16, label=lab,
                    yerr=[[100 * (v["mean"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["mean"]) for v in vals]], capsize=2)
    axes[1].set_xticks(x, KINDS); axes[1].set_ylabel("Top-1 (%), trio equal weights"); axes[1].legend(fontsize=7)
    axes[1].set_title("stage ablations (noise 0.25, silent 0.5)")
save(fig, "cascade_funnel_ablation")

# 4. triage: confusion matrices + mapping comparison --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
labs = list(triage.LABELS)
for ax, col, title in ((axes[0], "triage_rule", "rule-based triage (dev-calibrated)"), (axes[1], "triage_tree", "learned depth-3 tree")):
    C = np.array([[((inc.true_label == t) & (inc[col] == p)).sum() for p in labs] for t in labs], float)
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * Cn[i, j]:.0f}%\n({int(C[i, j])})", ha="center", va="center", fontsize=8, color="w" if Cn[i, j] > 0.6 else "k")
    bal = np.mean(np.diag(Cn))
    ax.set_xticks(range(3), labs); ax.set_yticks(range(3), labs); ax.set_xlabel("predicted"); ax.set_ylabel("true region of the root")
    ax.set_title(f"{title}\nbalanced accuracy {100 * bal:.1f}% (eval twins)")
ms = ["ppr", "alarm_degree", "alarm_closeness", "equal", "triage_blog_select", "triage_blog_weighted", "triage_dev_select", "oracle_incident"]
xx = np.arange(len(KINDS))
for j, m in enumerate(ms):
    vals = [main[k]["methods"][m]["top1"] for k in KINDS]
    axes[2].bar(xx + (j - 3.5) * 0.1, [100 * v["mean"] for v in vals], 0.1, label=NAMES[m], color=plt.cm.tab10(j))
axes[2].set_xticks(xx, KINDS); axes[2].set_ylabel("Top-1 (%)"); axes[2].legend(fontsize=6, ncol=2, loc="lower left")
axes[2].set_title("does topology-aware selection help?")
save(fig, "triage")

# 5. poll interval / timing ---------------------------------------------------------------------------
if "poll" in S["phases"]:
    P = S["phases"]["poll"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, KINDS):
        for s, ls in ((0.0, "-"), (0.5, "--")):
            keys = sorted([g for g in P if g.startswith(f"{k}|poll=") and g.endswith(f"silent={s:g}")], key=lambda g: float(g.split("|")[1].split("=")[1]))
            xs = [float(g.split("|")[1].split("=")[1]) for g in keys]
            xs_plot = [max(x_, 3) for x_ in xs]
            for m, c in (("earliest_in_cascade", "#d62728"), ("common_ancestor", "#7f7f7f"), ("alarm_degree", "#1f77b4"), ("equal", "#9467bd")):
                ax.plot(xs_plot, [100 * P[g]["methods"][m]["top1"]["mean"] for g in keys], ls, marker="o", ms=3, color=c,
                        label=f"{NAMES[m]}, silent {s:g}")
        ideal = [g for g in P if g.startswith(f"{k}|") and g.endswith("ideal")]
        if ideal:
            v = P[ideal[0]]["methods"]["earliest_in_cascade"]["top1"]["mean"]
            ax.axhline(100 * v, color="#d62728", lw=0.8, ls=":"); ax.text(3, 100 * v + 2, "earliest, ideal clocks, no noise", fontsize=6, color="#d62728")
        ax.set_xscale("log"); ax.set_xticks([3, 15, 60, 300], ["0", "15", "60", "300"]); ax.set_xlabel("polling interval (s)")
        ax.set_title(k); ax.set_ylim(0, 102); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 (%)"); axes[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "polling")

# 6. size scaling -------------------------------------------------------------------------------------
if os.path.exists("results/size_incidents.csv.gz"):
    Z = pd.concat([pd.read_csv("results/size_incidents.csv.gz"), inc[(inc.noise == 0.25) & (inc.p_silent == 0.5)]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, c in zip(KINDS, plt.cm.tab10.colors):
        g = Z[Z.kind == k].groupby("n")
        t = g[["ms_cascade", "ms_trio"]].median().sum(1)
        axes[0].plot(t.index, t.values, "o-", color=c, label=k)
        a = g["rank:alarm_degree"].apply(lambda r: 100 * (r == 1).mean())
        axes[1].plot(a.index, a.values, "o-", color=c, label=f"{k}: alarm-degree")
        e = g["rank:equal"].apply(lambda r: 100 * (r == 1).mean())
        axes[1].plot(e.index, e.values, "s--", color=c, alpha=0.6, label=f"{k}: trio equal")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("twin size (nodes)"); axes[0].set_ylabel("median ms per incident (cascade + trio, 1 core)")
    axes[0].legend(fontsize=7); axes[0].set_title("cost per incident")
    axes[1].set_xscale("log"); axes[1].set_xlabel("twin size (nodes)"); axes[1].set_ylabel("Top-1 (%)"); axes[1].set_ylim(0, 102)
    axes[1].legend(fontsize=6, ncol=2); axes[1].set_title("accuracy vs size (noise 0.25, silent 0.5)")
    save(fig, "size_scaling")

# 7. graduated autonomy: risk-coverage ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for m, c in (("equal", "#9467bd"), ("alarm_degree", "#1f77b4"), ("w_star", "#ff7f0e")):
    hit = (inc[f"rank:{m}"] == 1).to_numpy()
    conf = inc[f"conf_agree:{m}"].to_numpy() + inc[f"conf_margin:{m}"].rank(pct=True).to_numpy() * 0.999
    o = np.argsort(-conf)
    cov = np.arange(1, len(o) + 1) / len(o)
    acc = np.cumsum(hit[o]) / np.arange(1, len(o) + 1)
    axes[0].plot(100 * cov, 100 * acc, color=c, label=NAMES.get(m, "trio, alarm-degree-heavy weights"))
axes[0].set_xlabel("coverage: share of incidents acted on (%)"); axes[0].set_ylabel("Top-1 precision on covered incidents (%)")
axes[0].set_title("risk-coverage: confidence = trio agreement, then margin"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
axes[0].set_ylim(40, 101)
# gate: auto-remediate only if all 3 trio members agree AND predicted blast radius <= B
rows = []
for Bmax in (0, 5, 20, 50, 1e9):
    for agree in (3, 2):
        gate = (inc["conf_agree:equal"] >= agree) & (inc["top1_blast:equal"] <= Bmax)
        hit = inc.loc[gate, "rank:equal"] == 1
        rows.append((Bmax, agree, 100 * gate.mean(), 100 * hit.mean() if gate.any() else np.nan,
                     100 * ((inc["rank:equal"] != 1) & gate).mean()))
G_ = pd.DataFrame(rows, columns=["B", "agree", "coverage", "precision", "wrong_auto"])
for agree, mk in ((3, "o-"), (2, "s--")):
    g = G_[G_.agree == agree]
    axes[1].plot(g.coverage, g.precision, mk, label=f"trio agreement >= {agree}/3")
    for _, r in g.iterrows():
        axes[1].annotate("B=" + ("inf" if r.B > 1e6 else f"{int(r.B)}"), (r.coverage, r.precision), fontsize=7, xytext=(3, -9), textcoords="offset points")
axes[1].set_xlabel("auto-remediated incidents (%)"); axes[1].set_ylabel("correct root among them (%)")
axes[1].set_title("graduated-autonomy gate (blast radius <= B nodes)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
G_.to_csv("results/autonomy_gate.csv", index=False)
save(fig, "autonomy")

# 8. forest plot of planned comparisons --------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
for ax, k in zip(axes, KINDS):
    cs = [c for c in main[k]["comparisons"] if c["planned"]]
    for yi, c in enumerate(cs):
        col = ("#2ca02c" if c["diff"] > 0 else "#d62728") if c["significant"] else "#555555"
        ax.errorbar(100 * c["diff"], yi, xerr=[[100 * (c["diff"] - c["lo"])], [100 * (c["hi"] - c["diff"])]], fmt="o", color=col, capsize=2, ms=4,
                    mfc=col if c["significant"] else "white")
        ax.annotate(f"{c['twins_pos']}+/{c['twins_neg']}-", (100 * c["hi"], yi), xytext=(4, -3), textcoords="offset points", fontsize=6, color="#444")
    ax.axvline(0, color="k", lw=0.8); ax.set_title(f"{k} ({main[k]['twins']} twins)"); ax.set_xlabel("Top-1 difference (pp)")
    ax.margins(x=0.15)
axes[0].set_yticks(range(len(cs)), [f"{NAMES[c['a']]}  vs  {NAMES[c['b']]}" for c in cs], fontsize=7); axes[0].invert_yaxis()
fig.suptitle("planned paired comparisons: 95% twin-clustered bootstrap CI; filled = twin-level sign-flip, Holm (40 tests) p < 0.05 AND CI excludes 0; "
             "labels = #twins with positive / negative mean difference", fontsize=9)
save(fig, "comparisons_forest")

# 9. topology gallery + a worked incident -------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, k in zip(axes, KINDS):
    G = make_twin(k, 160, 7)
    pos = nx.kamada_kawai_layout(G) if k != "hierarchical" else nx.multipartite_layout(G, subset_key={l: [v for v in G if G.graph["layer"][v] == l] for l in set(G.graph["layer"].values())}, align="horizontal")
    if k == "hierarchical":
        pos = {v: (p[0], -p[1]) for v, p in pos.items()}
    regc = {"hierarchical": "#1f77b4", "star": "#ff7f0e", "mesh": "#2ca02c"}
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[40 if v in G.graph["gateways"] else 10 for v in G],
                           node_color=[regc[G.nodes[v]["region"]] for v in G])
    ax.set_title(f"{k} (160-node example; big = gateway)"); ax.axis("off")
save(fig, "topologies")

def worked(G, root, P, rng, fname, what):
    incd = simulate(G, root, P, rng)
    ctx = rca.context(G, incd)
    nodes, st = rca.cascade(ctx, seed=3)
    tr = rca.trio(ctx, nodes)
    H = G.subgraph(rca.neighbourhood(G, nodes, 1))
    pos = nx.kamada_kawai_layout(H)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, (("alarm_degree", "alarm-degree"), ("alarm_closeness", "alarm-closeness"), ("ppr", "alarm-seeded PPR"))):
        s = tr[key]
        vals = np.array([s[v] if np.isfinite(s[v]) else np.nan for v in H])
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3)
        sc = ax.scatter([pos[v][0] for v in H], [pos[v][1] for v in H], c=vals, cmap="viridis", s=30, zorder=2)
        al = [v for v in H if v in incd.alarms]
        ax.scatter([pos[v][0] for v in al], [pos[v][1] for v in al], facecolors="none", edgecolors="r", s=80, lw=1.2, zorder=3, label="alarming")
        ax.scatter([pos[root][0]], [pos[root][1]], marker="*", s=300, c="gold", edgecolors="k", zorder=4,
                   label="true root" + (" (silent)" if root not in incd.alarms else " (alarming)"))
        best = max(s[v] for v in H if np.isfinite(s[v]))
        tops = [v for v in H if s[v] == best]
        ax.scatter([pos[v][0] for v in tops], [pos[v][1] for v in tops], marker="s", s=120, facecolors="none", edgecolors="k", lw=1.5, zorder=4,
                   label=f"top score ({len(tops)} tied); root rank {rca.rank_of(s, root)} (ties count against it)")
        plt.colorbar(sc, ax=ax, shrink=0.7); ax.set_title(title); ax.axis("off"); ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"worked incident: {what}, {len(incd.alarms)} alarms ({sum(k == 'SPURIOUS' for k in incd.kinds.values())} spurious); "
                 f"candidates {st['s0']} -> {st['s1']} -> {st['s2']}", fontsize=9)
    save(fig, fname)
    return {k: rca.rank_of(tr[k], root) for k in tr}


G = make_twin("hybrid", 400, 3)
roots = [v for v in eligible_roots(G) if G.nodes[v]["tier"] == "hub"]
worked(G, roots[0], FaultParams(p_silent=1.0, noise=0.5), np.random.default_rng(11), "worked_incident", "hybrid twin, silent star hub fails (an easy case)")
# a typical *hard* case (review n3): an alarming mesh root where alarm-degree does not rank the root first.
# Deterministic search: the first (root index, rng seed) on this twin whose alarm-degree rank is > 1.
G = make_twin("mesh", 400, 3)
er = eligible_roots(G)
for i in range(200):
    r_, seed_ = er[(7 * i) % len(er)], 100 + i
    inc_ = simulate(G, r_, FaultParams(p_silent=0.0, noise=0.25), np.random.default_rng(seed_))
    c_ = rca.context(G, inc_); nd_, _ = rca.cascade(c_, seed=3)
    if r_ in nd_ and rca.rank_of(rca.trio(c_, nd_)["alarm_degree"], r_) > 1:
        break
worked(G, r_, FaultParams(p_silent=0.0, noise=0.25), np.random.default_rng(seed_), "worked_incident_mesh",
       "mesh twin, alarming root (a hard case: neighbours tie or beat the root on alarm-degree)")

# 10. p_link sensitivity (review M1) -------------------------------------------------------------------
if "plink" in S["phases"]:
    PL = S["phases"]["plink"]
    LM = [("alarm_degree", "#1f77b4"), ("alarm_closeness", "#8c564b"), ("ppr", "#d62728"), ("equal", "#9467bd"),
          ("common_ancestor", "#7f7f7f"), ("earliest_in_cascade", "#e377c2"), ("triage_dev_select", "#bcbd22")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey="row")
    for j, k in enumerate(KINDS):
        keys = sorted([g for g in PL if g.startswith(f"{k}|p_link=") and g.endswith("|model=all")], key=lambda g: float(g.split("|")[1][7:]))
        xs = [float(g.split("|")[1][7:]) for g in keys]
        up = f"{k}|p_link=0.9|model=upstream"
        ax = axes[0, j]
        for m, c in LM:
            v = [PL[g]["methods"][m]["top1"] for g in keys]
            ax.plot(xs, [100 * a["mean"] for a in v], "o-", color=c, ms=3, label=NAMES[m])
            ax.fill_between(xs, [100 * a["lo"] for a in v], [100 * a["hi"] for a in v], color=c, alpha=0.12)
            if up in PL:
                ax.plot([0.97], [100 * PL[up]["methods"][m]["top1"]["mean"]], "D", mfc="white", color=c, ms=5)
        ax.set_title(f"{k} ({PL[keys[0]]['twins']} twins)"); ax.set_ylim(0, 102); ax.grid(alpha=0.3)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9, 0.97], ["0.3", "0.5", "0.7", "0.9", "up-\nstream"])
        ax = axes[1, j]
        for sil, mk in (("0", "o-"), ("0.5", "s--")):
            kk = [f"{k}|p_link={x:g}|model=all|silent={sil}" for x in xs]
            cs = [next(c for c in PL[g]["comparisons"] if c["a"] == "alarm_degree" and c["b"] == "equal") for g in kk]
            ax.errorbar(xs, [100 * c["diff"] for c in cs], yerr=[[100 * (c["diff"] - c["lo"]) for c in cs], [100 * (c["hi"] - c["diff"]) for c in cs]],
                        fmt=mk, capsize=2, ms=4, label=f"silent-root prob. {sil}")
            ku = f"{k}|p_link=0.9|model=upstream|silent={sil}"
            if ku in PL:
                cu = next(c for c in PL[ku]["comparisons"] if c["a"] == "alarm_degree" and c["b"] == "equal")
                ax.errorbar([0.97], [100 * cu["diff"]], yerr=[[100 * (cu["diff"] - cu["lo"])], [100 * (cu["hi"] - cu["diff"])]], fmt="D",
                            mfc="white", capsize=2, ms=5, color=ax.lines[-1].get_color())
        ax.axhline(0, color="k", lw=0.8); ax.grid(alpha=0.3); ax.set_xlabel("neighbour LINK_DOWN probability p_link")
        ax.set_xticks([0.3, 0.5, 0.7, 0.9, 0.97], ["0.3", "0.5", "0.7", "0.9", "up-\nstream"])
    axes[0, 0].set_ylabel("Top-1 (%), silent 0 and 0.5 pooled"); axes[1, 0].set_ylabel("alarm-degree minus equal-weight trio (pp)")
    axes[0, -1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5)); axes[1, -1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("sensitivity to the neighbour-alarm model (noise 0.25; 40 incidents x condition x twin; 95% twin-clustered bootstrap CI). "
                 "Hollow diamonds: upstream-only LINK_DOWN (p_link 0.9)", fontsize=9)
    save(fig, "plink_sensitivity")
