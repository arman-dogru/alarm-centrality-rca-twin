"""Calibrate on the dev twins, then score every method on the evaluation twins.

1. Dev calibration (results/raw/dev only) -> results/calibration.json
   * common-ancestor coverage fraction (best dev Top-1)
   * rule-based triage thresholds (grid search, balanced accuracy vs the root's true region type)
   * a depth-3 decision-tree triage (learned) on the same features
   * the dev-best member for each triage label (the blog's mapping is kept as the pre-registered variant)
2. Derived methods on every phase: triage variants applied offline from the stored features and ranks.
3. Summary statistics: Top-1/Top-5/MRR with 95% twin x incident cluster-bootstrap CIs; planned paired
   comparisons (Top-1 difference, bootstrap CI, incident-level sign-flip p, Holm). A comparison counts as
   significant only when Holm p < 0.05 AND the CI excludes 0.
Outputs: results/<phase>_incidents.csv.gz (tidy per-incident ranks), results/summary.json, results/summary.md
"""
import glob, itertools, json, os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcatwin.stats import boot_means, ci, cluster_weights, holm, signflip_p
from rcatwin.triage import LABELS, SELECT

B = int(os.environ.get("BOOT", 2000))
rng = np.random.default_rng(7)
FEATS = ["hub_share", "cycle_rank", "deg_cv", "depth", "diameter", "clustering", "n"]


def load(phase):
    fs = sorted(glob.glob(f"results/raw/{phase}/*.csv.gz"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def balanced_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def rule(df, scope, c_mesh, d_star):
    cyc, cv = df[f"{scope}:cycle_rank"].to_numpy(), df[f"{scope}:deg_cv"].to_numpy()
    return np.where(cyc >= c_mesh, "mesh", np.where(cv >= d_star, "star", "hierarchical"))


# 1. calibration --------------------------------------------------------------------------------------
dev = load("dev")
dev = dev[dev.no_alarms == 0].reset_index(drop=True)
cal = {"n_dev_incidents": int(len(dev)), "dev_seeds": sorted(dev.seed.unique().tolist())}
fr = {c.split("_")[-1]: float((dev[c] == 1).mean()) for c in dev if c.startswith("rank:common_ancestor_")}
cal["common_ancestor_frac"] = {"dev_top1": fr, "chosen": max(fr, key=fr.get)}
y = dev.true_label.to_numpy()
best = None
for scope in ("feat", "featctx"):
    for c_mesh in np.round(np.arange(0.05, 1.01, 0.05), 2):
        for d_star in np.round(np.arange(0.8, 3.01, 0.1), 2):
            acc = balanced_acc(y, rule(dev, scope, c_mesh, d_star))
            if best is None or acc > best[0] + 1e-12:
                best = (acc, scope, float(c_mesh), float(d_star))
cal["rule"] = dict(zip(["dev_balanced_acc", "scope", "cycle_rank_mesh_min", "deg_cv_star_min"], best))
Xcols = [f"{s}:{f}" for s in ("feat", "featctx") for f in FEATS]
tree = DecisionTreeClassifier(max_depth=3, random_state=0, class_weight="balanced").fit(dev[Xcols], y)
cal["tree"] = {"dev_balanced_acc": balanced_acc(y, tree.predict(dev[Xcols])), "rules": export_text(tree, feature_names=Xcols)}
CAND = ["ppr", "alarm_degree", "alarm_closeness", "equal"] + [f"w_{l}" for l in LABELS]
dev["triage_rule"] = rule(dev, cal["rule"]["scope"], cal["rule"]["cycle_rank_mesh_min"], cal["rule"]["deg_cv_star_min"])
cal["dev_best_member"] = {}
for lab in LABELS:
    sub = dev[dev.triage_rule == lab]
    acc = {m: float((sub[f"rank:{m}"] == 1).mean()) for m in CAND} if len(sub) else {m: 0.0 for m in CAND}
    cal["dev_best_member"][lab] = {"chosen": max(acc, key=acc.get), "dev_top1": acc, "n": int(len(sub))}
json.dump(cal, open("results/calibration.json", "w"), indent=1)


# 2. derived methods ----------------------------------------------------------------------------------
def derive(df):
    df = df[df.no_alarms == 0].reset_index(drop=True).copy()
    df["rank:common_ancestor"] = df[f"rank:common_ancestor_{cal['common_ancestor_frac']['chosen']}"]
    r = cal["rule"]
    df["triage_rule"] = rule(df, r["scope"], r["cycle_rank_mesh_min"], r["deg_cv_star_min"])
    df["triage_tree"] = tree.predict(df[Xcols])
    pick = lambda labs, f: np.array([df.at[i, f"rank:{f(l)}"] for i, l in enumerate(labs)])
    df["rank:triage_blog_select"] = pick(df.triage_rule, lambda l: SELECT[l])
    df["rank:triage_blog_weighted"] = pick(df.triage_rule, lambda l: f"w_{l}")
    df["rank:triage_dev_select"] = pick(df.triage_rule, lambda l: cal["dev_best_member"][l]["chosen"])
    df["rank:triage_tree_weighted"] = pick(df.triage_tree, lambda l: f"w_{l}")
    df["rank:oracle_label_weighted"] = pick(df.true_label, lambda l: f"w_{l}")
    df["rank:oracle_incident"] = df[["rank:ppr", "rank:alarm_degree", "rank:alarm_closeness"]].min(1)
    return df


METHODS = ["random_candidate", "global_pagerank", "earliest_alarm", "earliest_in_cascade", "common_ancestor",
           "static_pagerank", "static_degree", "static_closeness", "ppr", "alarm_degree", "alarm_closeness",
           "equal", "triage_blog_select", "triage_blog_weighted", "triage_dev_select", "triage_tree_weighted",
           "oracle_label_weighted", "oracle_incident"]
COMPARISONS = [("ppr", "static_pagerank"), ("alarm_degree", "static_degree"), ("alarm_closeness", "static_closeness"),
               ("equal", "earliest_in_cascade"), ("equal", "common_ancestor"),
               ("triage_blog_weighted", "equal"), ("triage_blog_select", "equal"), ("triage_dev_select", "equal"),
               ("triage_tree_weighted", "equal"), ("alarm_degree", "equal")]


def metrics(df, groups, methods):
    """{group: {method: {top1, top5, mrr: {mean, lo, hi}}, n, twins}} + paired comparisons per group."""
    out = {}
    for key, g in groups:
        W = cluster_weights(g.seed.to_numpy(), B, rng)
        R = np.stack([g[f"rank:{m}"].to_numpy(float) for m in methods])
        vals = {"top1": (R == 1).astype(float), "top5": (R <= 5).astype(float), "mrr": 1.0 / R}
        res = {"n": int(len(g)), "twins": int(g.seed.nunique()), "methods": {}}
        bm = {k: boot_means(v, W) for k, v in vals.items()}
        for i, m in enumerate(methods):
            res["methods"][m] = {k: {"mean": float(vals[k][i].mean()), **dict(zip(("lo", "hi"), ci(bm[k][i])))}
                                 for k in vals}
        comps = []
        for a, b in COMPARISONS:
            if a in methods and b in methods:
                ia, ib = methods.index(a), methods.index(b)
                d = vals["top1"][ia] - vals["top1"][ib]
                lo, hi = ci(bm["top1"][ia] - bm["top1"][ib])
                comps.append({"a": a, "b": b, "diff": float(d.mean()), "lo": lo, "hi": hi, "p": signflip_p(d, 5000, rng)})
        res["comparisons"] = comps
        out[key] = res
    return out


summary = {"calibration": cal, "phases": {}}
for phase in ("main", "poll", "ablate", "size"):
    df = load(phase)
    if df is None:
        continue
    df = derive(df)
    keep = ["phase", "kind", "n", "seed", "incident", "noise", "p_silent", "poll_interval", "root", "root_tier", "true_label",
            "triage_rule", "triage_tree", "n_alarms", "root_alarmed", "blast", "root_in_s0", "root_in_s2"]
    keep += [c for c in df if c.startswith(("rank:", "stage:", "conf_", "top1_blast", "ms_", "size@", "feat:", "featctx:"))]
    if "jitter" in df:
        keep.append("jitter")
    df[keep].to_csv(f"results/{phase}_incidents.csv.gz", index=False, compression="gzip")
    ms = METHODS + ([c[5:] for c in df if c.startswith("rank:") and "@" in c] if phase == "ablate" else [])
    if phase == "main":
        groups = [("all", df)] + [(k, g) for k, g in df.groupby("kind")]
        groups += [(f"{k}|noise={z:g}", g) for (k, z), g in df.groupby(["kind", "noise"])]
        groups += [(f"{k}|silent={s:g}", g) for (k, s), g in df.groupby(["kind", "p_silent"])]
    elif phase == "poll":
        df["cond"] = [f"poll={p:g}|silent={s:g}" + ("|ideal" if j == 0 else "") for p, s, j in
                      zip(df.poll_interval, df.p_silent, df.get("jitter", pd.Series(5.0, index=df.index)).fillna(5.0))]
        groups = [(f"{k}|{c}", g) for (k, c), g in df.groupby(["kind", "cond"])]
    elif phase == "size":
        groups = [(f"{k}|n={n}", g) for (k, n), g in df.groupby(["kind", "n"])]
    else:
        groups = [("all", df)] + [(k, g) for k, g in df.groupby("kind")]
    summary["phases"][phase] = metrics(df, groups, ms)
    print(phase, len(df), "incidents", flush=True)

# Holm over the planned family: per-topology comparisons of the main phase
fam = [(k, c) for k in ("hierarchical", "star", "mesh", "hybrid") if k in summary["phases"].get("main", {})
       for c in summary["phases"]["main"][k]["comparisons"]]
for (k, c), ph in zip(fam, holm([c["p"] for _, c in fam])):
    c["p_holm"] = float(ph)
    c["significant"] = bool(ph < 0.05 and (c["lo"] > 0 or c["hi"] < 0))
json.dump(summary, open("results/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in cal.items() if k != "tree"}, indent=1))
