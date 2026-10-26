"""Calibrate on the dev twins, then score every method on the evaluation twins.

1. Dev calibration (<raw>/dev only; refuses non-dev seeds) -> results/calibration.json
   * common-ancestor coverage fraction (best dev Top-1)
   * rule-based triage thresholds (grid search, balanced accuracy vs the root's true region type)
   * a depth-3 decision-tree triage (learned) on the same features
   * the dev-best member for each triage label (the blog-reading mapping is kept as the pre-registered variant)
2. Derived methods on every phase: triage variants and the per-incident oracle, applied offline from stored ranks.
3. Summary statistics: Top-1/Top-5/MRR (+ ties-split Top-1) with 95% twin-clustered bootstrap CIs (stratified by
   topology for pooled groups). Planned paired comparisons: Top-1 difference with a paired bootstrap CI and a
   TWIN-LEVEL two-sided sign-flip test on per-twin mean differences (twin-level t-test as a secondary check; the
   incident-level sign-flip is kept only as a descriptive number). Holm over the 40 per-topology planned
   comparisons of the main phase. A comparison is significant only when Holm p < 0.05 AND the CI excludes 0.
Outputs: results/<phase>_incidents.csv.gz (tidy per-incident ranks), results/summary.json, results/calibration.json
Usage: python scripts/aggregate.py [--raw results/raw]
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rcatwin.grid import DEV_SEEDS
from rcatwin.rca import expected_top1
from rcatwin.stats import boot_means, ci, cluster_weights, holm, signflip_p, twin_signflip_p, twin_t_p
from rcatwin.triage import LABELS, SELECT, classify

B = int(os.environ.get("BOOT", 2000))
FLIPS = 20000
FEATS = ["hub_share", "cycle_rank", "deg_cv", "depth", "diameter", "clustering", "n"]
XCOLS = [f"{s}:{f}" for s in ("feat", "featctx") for f in FEATS]
CAND = ["ppr", "alarm_degree", "alarm_closeness", "equal"] + [f"w_{l}" for l in LABELS]
METHODS = ["random_candidate", "global_pagerank", "earliest_alarm", "earliest_in_cascade", "common_ancestor",
           "static_pagerank", "static_degree", "static_closeness", "ppr", "alarm_degree", "alarm_closeness",
           "alarm_degree_tb", "equal", "triage_blog_select", "triage_blog_weighted", "triage_dev_select", "triage_tree_weighted",
           "oracle_label_weighted", "oracle_incident"]
COMPARISONS = [("ppr", "static_pagerank"), ("alarm_degree", "static_degree"), ("alarm_closeness", "static_closeness"),
               ("equal", "earliest_in_cascade"), ("equal", "common_ancestor"),
               ("triage_blog_weighted", "equal"), ("triage_blog_select", "equal"), ("triage_dev_select", "equal"),
               ("triage_tree_weighted", "equal"), ("alarm_degree", "equal")]
EXPLORATORY = [("alarm_degree_tb", "alarm_degree")]  # added after the first look at the eval grid; not in the Holm family


def load(raw, phase):
    fs = sorted(glob.glob(os.path.join(raw, phase, "*.csv.gz")))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def balanced_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def rule(df, scope, c_mesh, d_star):
    return classify(df[f"{scope}:cycle_rank"], df[f"{scope}:deg_cv"], c_mesh, d_star)


def calibrate(dev):
    """Fit every tunable choice on dev incidents only. Returns (cal dict, fitted tree)."""
    if not set(dev.seed.unique()) <= set(DEV_SEEDS):
        raise ValueError(f"calibration must use dev seeds {DEV_SEEDS} only, got {sorted(dev.seed.unique())}")
    dev = dev[dev.no_alarms == 0].reset_index(drop=True)
    cal = {"n_dev_incidents": int(len(dev)), "dev_seeds": sorted(int(s) for s in dev.seed.unique())}
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
    tree = DecisionTreeClassifier(max_depth=3, random_state=0, class_weight="balanced").fit(dev[XCOLS], y)
    cal["tree"] = {"dev_balanced_acc": balanced_acc(y, tree.predict(dev[XCOLS])), "rules": export_text(tree, feature_names=XCOLS)}
    lab_dev = rule(dev, cal["rule"]["scope"], cal["rule"]["cycle_rank_mesh_min"], cal["rule"]["deg_cv_star_min"])
    cal["dev_best_member"] = {}
    for lab in LABELS:
        sub = dev[lab_dev == lab]
        acc = {m: float((sub[f"rank:{m}"] == 1).mean()) for m in CAND} if len(sub) else {m: 0.0 for m in CAND}
        cal["dev_best_member"][lab] = {"chosen": max(acc, key=acc.get), "dev_top1": acc, "n": int(len(sub))}
    return cal, tree


def derive(df, cal, tree):
    """Add the chosen common ancestor, triage picks and the per-incident oracle (rank: and rankopt: columns)."""
    df = df[df.no_alarms == 0].reset_index(drop=True).copy()
    for pre in ("rank", "rankopt"):
        df[f"{pre}:common_ancestor"] = df[f"{pre}:common_ancestor_{cal['common_ancestor_frac']['chosen']}"]
    r = cal["rule"]
    df["triage_rule"] = rule(df, r["scope"], r["cycle_rank_mesh_min"], r["deg_cv_star_min"])
    df["triage_tree"] = tree.predict(df[XCOLS]) if tree is not None else df["triage_rule"]
    idx = np.arange(len(df))
    for pre in ("rank", "rankopt"):
        def pick(labs, f):
            return np.array([df.at[i, f"{pre}:{f(l)}"] for i, l in enumerate(labs)])
        df[f"{pre}:triage_blog_select"] = pick(df.triage_rule, lambda l: SELECT[l])
        df[f"{pre}:triage_blog_weighted"] = pick(df.triage_rule, lambda l: f"w_{l}")
        df[f"{pre}:triage_dev_select"] = pick(df.triage_rule, lambda l: cal["dev_best_member"][l]["chosen"])
        df[f"{pre}:triage_tree_weighted"] = pick(df.triage_tree, lambda l: f"w_{l}")
        df[f"{pre}:oracle_label_weighted"] = pick(df.true_label, lambda l: f"w_{l}")
    trio_r = ["rank:ppr", "rank:alarm_degree", "rank:alarm_closeness"]
    best = df[trio_r].to_numpy().argmin(1)  # oracle: the trio member with the best (pessimistic) rank
    df["rank:oracle_incident"] = df[trio_r].to_numpy()[idx, best]
    df["rankopt:oracle_incident"] = df[[c.replace("rank:", "rankopt:") for c in trio_r]].to_numpy()[idx, best]
    return df


def is_significant(c):
    return bool(c["p_holm"] < 0.05 and (c["lo"] > 0 or c["hi"] < 0))


def metrics(groups, methods, rng, B=B, comparisons=COMPARISONS + EXPLORATORY):
    """{group: {n, twins, methods: {m: {metric: {mean, lo, hi}}}, comparisons: [...]}}. Twins are (kind, seed);
    groups spanning several topologies are bootstrapped stratified by topology."""
    out = {}
    for key, g in groups:
        twin = (g.kind.astype(str) + "|" + g.seed.astype(str)).to_numpy()
        W = cluster_weights(twin, B, rng, strata=g.kind.to_numpy())
        R = np.stack([g[f"rank:{m}"].to_numpy(float) for m in methods])
        Ro = np.stack([g[f"rankopt:{m}"].to_numpy(float) for m in methods])
        vals = {"top1": (R == 1).astype(float), "top5": (R <= 5).astype(float), "mrr": 1.0 / R,
                "top1_ties_split": expected_top1(R, Ro)}  # expected Top-1 under random tie-breaking
        res = {"n": int(len(g)), "twins": int(len(np.unique(twin))), "methods": {}}
        bm = {k: boot_means(v, W) for k, v in vals.items()}
        for i, m in enumerate(methods):
            res["methods"][m] = {k: {"mean": float(vals[k][i].mean()), **dict(zip(("lo", "hi"), ci(bm[k][i])))}
                                 for k in vals}
        comps = []
        for a, b in comparisons:
            if a in methods and b in methods:
                ia, ib = methods.index(a), methods.index(b)
                d = vals["top1"][ia] - vals["top1"][ib]
                lo, hi = ci(bm["top1"][ia] - bm["top1"][ib])
                tm = pd.Series(d).groupby(twin).mean()
                comps.append({"a": a, "b": b, "diff": float(d.mean()), "lo": lo, "hi": hi,
                              "p": twin_signflip_p(d, twin, FLIPS, rng), "p_t": twin_t_p(d, twin),
                              "p_incident": signflip_p(d, 5000, rng),
                              "twins_pos": int((tm > 0).sum()), "twins_neg": int((tm < 0).sum()),
                              "planned": (a, b) in COMPARISONS})
        res["comparisons"] = comps
        out[key] = res
    return out


def apply_holm(comps):
    """Holm-adjust the twin-level p of a family of comparison dicts in place and set `significant`."""
    for c, ph in zip(comps, holm([c["p"] for c in comps])):
        c["p_holm"] = float(ph)
        c["significant"] = is_significant(c)


def groups_for(phase, df):
    if phase == "main":
        groups = [("all", df)] + [(k, g) for k, g in df.groupby("kind")]
        groups += [(f"{k}|noise={z:g}", g) for (k, z), g in df.groupby(["kind", "noise"])]
        groups += [(f"{k}|silent={s:g}", g) for (k, s), g in df.groupby(["kind", "p_silent"])]
        return groups
    if phase == "poll":
        df["cond"] = [f"poll={p:g}|silent={s:g}" + ("|ideal" if j == 0 else "") for p, s, j in
                      zip(df.poll_interval, df.p_silent, df.get("jitter", pd.Series(5.0, index=df.index)).fillna(5.0))]
        return [(f"{k}|{c}", g) for (k, c), g in df.groupby(["kind", "cond"])]
    if phase == "plink":
        df["link_model"] = df.get("link_model", pd.Series("all", index=df.index)).fillna("all")
        groups = [(f"{k}|p_link={p:g}|model={m}", g) for (k, p, m), g in df.groupby(["kind", "p_link", "link_model"])]
        groups += [(f"{k}|p_link={p:g}|model={m}|silent={s:g}", g)
                   for (k, p, m, s), g in df.groupby(["kind", "p_link", "link_model", "p_silent"])]
        groups += [(f"all|p_link={p:g}|model={m}", g) for (p, m), g in df.groupby(["p_link", "link_model"])]
        return groups
    if phase == "size":
        return [(f"{k}|n={n}", g) for (k, n), g in df.groupby(["kind", "n"])]
    return [("all", df)] + [(k, g) for k, g in df.groupby("kind")]


def main(raw):
    rng = np.random.default_rng(7)
    res = os.path.join(ROOT, "results")
    cal, tree = calibrate(load(raw, "dev"))
    json.dump(cal, open(os.path.join(res, "calibration.json"), "w"), indent=1)
    summary = {"calibration": cal, "phases": {}}
    for phase in ("main", "plink", "poll", "ablate", "size"):
        df = load(raw, phase)
        if df is None:
            continue
        df = derive(df, cal, tree)
        keep = ["phase", "kind", "n", "seed", "incident", "noise", "p_silent", "poll_interval", "root", "root_tier", "true_label",
                "triage_rule", "triage_tree", "n_alarms", "root_alarmed", "blast", "root_in_s0", "root_in_s2"]
        keep += [c for c in ("jitter", "p_link", "link_model") if c in df]
        keep += [c for c in df if c.startswith(("rank:", "rankopt:", "stage:", "conf_", "top1_blast", "ms_", "size@", "feat:", "featctx:"))]
        df[[c for c in keep if c in df]].to_csv(os.path.join(res, f"{phase}_incidents.csv.gz"), index=False, compression="gzip")
        ms = METHODS + ([c[5:] for c in df if c.startswith("rank:") and "@" in c] if phase == "ablate" else [])
        summary["phases"][phase] = metrics(groups_for(phase, df), ms, rng)
        print(phase, len(df), "incidents", flush=True)
    M = summary["phases"]["main"]
    # primary family: 10 planned comparisons x 4 topologies (main phase); secondary family: the 10 pooled ones
    apply_holm([c for k in ("hierarchical", "star", "mesh", "hybrid") if k in M
                for c in M[k]["comparisons"] if c["planned"]])
    apply_holm([c for c in M.get("all", {"comparisons": []})["comparisons"] if c["planned"]])
    json.dump(summary, open(os.path.join(res, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in cal.items() if k != "tree"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw"))
    main(ap.parse_args().raw)
