"""Single-line mutation check: each mutant is applied to a throw-away copy of rcatwin/ and tests/ (the working
tree is never touched) and `pytest -x tests` must fail. Writes results/mutation_check.json; exit 1 if any
mutant survives. Extra arguments go to pytest (then nothing is written)."""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = [  # (file, original, mutant, what it breaks)
    ("rcatwin/rca.py", "(1 if optimistic else (s == r).sum())", "(1 if optimistic else 1)", "pessimistic rank breaks ties for the root"),
    ("rcatwin/rca.py", "(1 if optimistic else (s == r).sum())", "((s == r).sum() if optimistic else (s == r).sum())", "optimistic rank counts ties against the root"),
    ("rcatwin/rca.py", "return (np.asarray(opt) == 1) / (np.asarray(pess) - np.asarray(opt) + 1.0)", "return (np.asarray(pess) == 1) / (np.asarray(pess) - np.asarray(opt) + 1.0)", "ties-split Top-1 uses the pessimistic rank"),
    ("rcatwin/rca.py", "        return len(s)\n", "        return 1\n", "unrankable root counted as a hit"),
    ("rcatwin/rca.py", "x_new = alpha * (W.T @ x + dangling * p) + (1 - alpha) * p", "x_new = alpha * (W @ x + dangling * p) + (1 - alpha) * p", "PPR transition transposed"),
    ("rcatwin/rca.py", "x_new = alpha * (W.T @ x + dangling * p) + (1 - alpha) * p", "x_new = alpha * (W.T @ x + dangling * p) + (1 - alpha) / len(p)", "PPR restarts uniformly (not alarm-seeded)"),
    ("rcatwin/rca.py", "deg = M @ is_alarm + 0.5 * is_alarm", "deg = np.asarray(M.sum(1)).ravel() + 0.5 * is_alarm", "alarm-degree counts every edge"),
    ("rcatwin/rca.py", "close = 1.0 / (1.0 + D.mean(0))\n    else:", "close = 1.0 / (1.0 + D.min(0))\n    else:", "alarm-closeness uses the nearest alarm only"),
    ("rcatwin/rca.py", "keep = [c for c, k in zip(comps, cnt) if k >= min_alarms] or [comps[int(np.argmax(cnt))]]", "keep = comps", "decomposition keeps every component"),
    ("rcatwin/rca.py", "            if got >= need:\n                break", "            if got >= 0:\n                break", "clustering keeps one community only"),
    ("rcatwin/rca.py", "if L[v] == L[u] - 1 and v not in seen:", "if L[v] == L[u] + 1 and v not in seen:", "common ancestor walks downstream"),
    ("rcatwin/rca.py", "ok = c >= frac * c.max()", "ok = c >= c.max()", "common ancestor ignores frac"),
    ("rcatwin/rca.py", "return _full(ctx.n, ctx.A, -ctx.t)", "return _full(ctx.n, ctx.A, ctx.t)", "earliest alarm picks the latest"),
    ("rcatwin/rca.py", "out[nodes] = (rankdata(v) - 1) / max(1, len(v) - 1)", "out[nodes] = v", "ensemble mixes raw score scales"),
    ("rcatwin/faults.py", "disc = {v for v in others if v not in L1}", "disc = {v for v in others if v in L1}", "blast radius inverted"),
    ("rcatwin/faults.py", "if rng.random() >= P.p_silent:", "if rng.random() >= 0:", "root never alarms"),
    ("rcatwin/faults.py", "emit(v, _deliver(G, v, rng, P, trap_ok=False, extra=P.hop_latency * hops), \"UNREACHABLE\")", "emit(v, _deliver(G, v, rng, P, trap_ok=False, extra=-P.hop_latency * hops), \"UNREACHABLE\")", "effects before cause"),
    ("rcatwin/faults.py", "nxt = phase + P.poll_interval * np.ceil((t - phase) / P.poll_interval)", "nxt = phase + P.poll_interval * np.floor((t - phase) / P.poll_interval)", "poll before the event"),
    ("rcatwin/faults.py", "if len(set(G.neighbors(v)) | disc) >= min_blast:", "if len(disc) >= min_blast:", "eligible roots need disconnection"),
    ("rcatwin/twin.py", "if L[v] == L[u] - 1)", "if L[v] == L[u] + 1)", "upstream graph points downstream"),
    ("rcatwin/stats.py", "np.add.at(W[b], m[rng.integers(0, len(m), len(m))], 1)", "np.add.at(W[b], rng.integers(0, len(clusters), len(m)), 1)", "bootstrap ignores clusters"),
    ("rcatwin/stats.py", "np.abs(null) >= abs(obs) - 1e-12", "null >= obs - 1e-12", "one-sided sign-flip"),
    ("rcatwin/stats.py", "run = max(run, min(1.0, (k - r) * p[i]))", "run = min(1.0, (k - r) * p[i])", "Holm not monotone"),
    ("rcatwin/triage.py", '"cycle_rank": (m - n + comps) / max(1, n)', '"cycle_rank": (m - n) / max(1, n)', "cycle rank ignores components"),
    ("rcatwin/triage.py", 'np.where(cyc >= cycle_mesh_min, "mesh", np.where(cv >= deg_cv_star_min, "star", "hierarchical"))', 'np.where(cyc >= cycle_mesh_min, "star", np.where(cv >= deg_cv_star_min, "mesh", "hierarchical"))', "pipeline triage rule swaps mesh/star"),
    # --- round-1 review: the reviewer's surviving mutants (probes/extra_mutants.py), adapted to the current code ---
    ("rcatwin/rca.py", "deg = M @ is_alarm + 0.5 * is_alarm", "deg = M @ is_alarm + 0.0 * is_alarm", "alarm-degree drops the self-alarm term"),
    ("rcatwin/rca.py", "deg = M @ is_alarm + 0.5 * is_alarm", "deg = M @ is_alarm + 5.0 * is_alarm", "self-alarm dominates alarm-degree"),
    ("rcatwin/rca.py", "D[np.isinf(D)] = len(nodes)\n        close", "D[np.isinf(D)] = 0\n        close", "alarm-closeness: unreachable alarms at distance 0"),
    ("rcatwin/rca.py", "comms = sorted(comms, key=lambda m: (-len(m & Aset), -len(m), min(m)))", "comms = sorted(comms, key=lambda m: (len(m & Aset), -len(m), min(m)))", "Louvain keeps the least-alarmed communities first"),
    ("rcatwin/rca.py", "out[nodes] = (rankdata(v) - 1) / max(1, len(v) - 1)", "out[nodes] = (rankdata(v, method='ordinal') - 1) / max(1, len(v) - 1)", "percentile ties broken by node index"),
    ("rcatwin/rca.py", '    key = G.graph.get("twin_id")', "    key = id(G)", "global PR cache keyed by id(G) (review m4)"),
    ("rcatwin/stats.py", "def ci(boot, lo=2.5, hi=97.5):", "def ci(boot, lo=10, hi=90):", "CI is 80% not 95%"),
    ("rcatwin/stats.py", "return (np.asarray(X, np.float64) @ W.T.astype(np.float64)) / W.sum(1)", "return (np.asarray(X, np.float64) @ W.T.astype(np.float64)) / W.shape[1]", "bootstrap mean divides by N"),
    ("rcatwin/stats.py", "return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (flips + 1))", "return float(np.sum(np.abs(null) >= abs(obs) - 1e-12) / flips)", "sign-flip p can be 0"),
    ("rcatwin/stats.py", "for k in rng.integers(0, len(members), len(members)):", "for k in range(len(members)):", "bootstrap never resamples clusters"),
    ("rcatwin/stats.py", "        groups.append([np.flatnonzero((clusters == c) & (strata == s)) for c in ids])", "        groups.append([np.flatnonzero(clusters == c) for c in np.unique(clusters)])", "stratified bootstrap ignores strata"),
    ("rcatwin/stats.py", "    m = twin_means(d, clusters)\n    if not np.any(m):", "    m = np.asarray(d, float)\n    if not np.any(m):", "twin-level test runs on incidents"),
    ("rcatwin/stats.py", "    return float(_st.ttest_1samp(m, 0.0).pvalue)", "    return float(_st.ttest_1samp(np.asarray(d, float), 0.0).pvalue)", "twin t-test runs on incidents"),
    ("rcatwin/faults.py", "if rng.random() < P.p_link and", "if rng.random() < 1.0 and", "p_link ignored"),
    ("rcatwin/faults.py", '(P.link_model == "all" or L0[u] < L0[root])', '(P.link_model == "all" or L0[u] <= L0[root])', "upstream model admits same-layer peers"),
    ("rcatwin/faults.py", 'emit(int(v), float(rng.uniform(-60, 300)), "SPURIOUS")', 'emit(int(v), float(rng.uniform(0, 300)), "SPURIOUS")', "no pre-fault spurious alarms"),
    ("rcatwin/faults.py", "deg = {v for v in others if v in L1 and L1[v] > L0[v]}", "deg = {v for v in others if v in L1 and L1[v] >= L0[v]}", "every reachable node is degraded"),
    ("rcatwin/faults.py", "if rng.random() < P.p_degrade:", "if rng.random() < 1.0:", "p_degrade ignored"),
    ("rcatwin/faults.py", "rng.poisson(P.noise * max(1, len(true)))", "rng.poisson(P.noise * max(1, len(true)) * 3)", "3x spurious rate"),
    ("rcatwin/faults.py", "if len(set(G.neighbors(v)) | disc) >= min_blast:", "if len(set(G.neighbors(v)) | disc) >= min_blast - 1:", "eligible-root threshold off by one"),
    ("rcatwin/faults.py", "            hops = max(1, L0[v] - L0[root])", "            hops = 1", "UNREACHABLE delay ignores depth"),
    ("rcatwin/experiment.py", 'S["alarm_degree_tb"] = tr["alarm_degree"] + 1e-3 * rca.pct(tr["alarm_closeness"], nodes)', 'S["alarm_degree_tb"] = tr["alarm_degree"] + 1e3 * rca.pct(tr["alarm_closeness"], nodes)', "tie-break dominates alarm-degree"),
    ("rcatwin/experiment.py", '        out[f"rank:{k}"] = rca.rank_of(s, root)', '        out[f"rank:{k}"] = rca.rank_of(s, root, optimistic=True)', "headline rank becomes optimistic"),
    ("rcatwin/experiment.py", "    top = int(np.argmax(s))\n    return top if", "    top = int(np.argmin(s))\n    return top if", "autonomy gate top-1 is the worst node"),
    ("rcatwin/experiment.py", "    return top if int((s == s[top]).sum()) == 1 else None", "    return top", "gate agreement counts tied tops"),
    ("rcatwin/triage.py", "WEIGHTS = {lab: {k: (0.6 if k == SELECT[lab] else 0.2) for k in SELECT.values()} for lab in LABELS}", "WEIGHTS = {lab: {k: (0.2 if k == SELECT[lab] else 0.4) for k in SELECT.values()} for lab in LABELS}", "blog-reading weights inverted"),
    ("rcatwin/triage.py", 'SELECT = {"hierarchical": "ppr", "star": "alarm_degree", "mesh": "alarm_closeness"}', 'SELECT = {"hierarchical": "alarm_closeness", "star": "alarm_degree", "mesh": "ppr"}', "blog-reading mapping swapped"),
    ("scripts/aggregate.py", "best = df[trio_r].to_numpy().argmin(1)", "best = df[trio_r].to_numpy().argmax(1)", "oracle picks the worst trio member"),
    ("scripts/aggregate.py", '    cal, tree = calibrate(load(raw, "dev"))', '    cal, tree = calibrate(load(raw, "main"))', "calibration on eval twins"),
    ("scripts/aggregate.py", "    if not set(dev.seed.unique()) <= set(DEV_SEEDS):", "    if False:", "calibration accepts eval seeds"),
    ("scripts/aggregate.py", '    return bool(c["p_holm"] < 0.05 and (c["lo"] > 0 or c["hi"] < 0))', '    return bool(c["p_holm"] < 0.05)', "significance ignores the CI"),
    ("scripts/aggregate.py", '    for c, ph in zip(comps, holm([c["p"] for c in comps])):', '    for c, ph in zip(comps, [c["p"] for c in comps]):', "no Holm correction"),
    ("scripts/aggregate.py", '                              "p": twin_signflip_p(d, twin, FLIPS, rng), "p_t": twin_t_p(d, twin),', '                              "p": signflip_p(d, 5000, rng), "p_t": twin_t_p(d, twin),', "primary test back to incident level"),
    ("scripts/aggregate.py", '        df[f"{pre}:triage_blog_select"] = pick(df.triage_rule, lambda l: SELECT[l])', '        df[f"{pre}:triage_blog_select"] = pick(df.triage_rule, lambda l: "equal")', "blog select uses equal weights"),
    ("scripts/aggregate.py", '        df[f"{pre}:triage_dev_select"] = pick(df.triage_rule, lambda l: cal["dev_best_member"][l]["chosen"])', '        df[f"{pre}:triage_dev_select"] = pick(df.triage_rule, lambda l: SELECT[l])', "dev-tuned select uses the blog mapping"),
    ("scripts/aggregate.py", "        W = cluster_weights(twin, B, rng, strata=g.kind.to_numpy())", "        W = cluster_weights(np.arange(len(g)), B, rng)", "bootstrap ignores twins"),
    # ---- review round 2 (F3): reviewer's surviving mutants + the rank rewrite once called equivalent ----
    ("rcatwin/rca.py", "return int((s > r).sum() + (1 if optimistic else (s == r).sum()))", "return int((s >= r).sum() + (1 if optimistic else 0))", "rank rewrite (not equivalent: breaks optimistic)"),
    ("rcatwin/stats.py", "    return np.bincount(inv, weights=d) / np.bincount(inv)", "    return np.bincount(inv, weights=d)", "twin_means returns sums"),
    ("scripts/aggregate.py", "        W = cluster_weights(twin, B, rng, strata=g.kind.to_numpy())", "        W = cluster_weights(twin, B, rng)", "pooled bootstrap unstratified"),
    ("scripts/aggregate.py", "        twin = (g.kind.astype(str) + \"|\" + g.seed.astype(str)).to_numpy()", "        twin = g.seed.astype(str).to_numpy()", "twin id = seed only (merges topologies)"),
    ("scripts/aggregate.py", "\"twins_pos\": int((tm > 0).sum())", "\"twins_pos\": int((tm >= 0).sum())", "twins_pos counts zeros"),
    ("scripts/aggregate.py", "    apply_holm([c for c in M.get(\"all\", {\"comparisons\": []})[\"comparisons\"] if c[\"planned\"]])", "    apply_holm([c for c in M.get(\"all\", {\"comparisons\": []})[\"comparisons\"]])", "pooled Holm family includes exploratory"),
    ("rcatwin/experiment.py", "    agree = 0 if top is None else sum(unique_top(trio_scores[k]) == top for k in TRIO)", "    agree = 0 if top is None else sum(int(np.argmax(trio_scores[k])) == top for k in TRIO)", "members' tied tops counted again"),
    ("rcatwin/grid.py", "EVAL_SEEDS = tuple(range(100, 130))", "EVAL_SEEDS = tuple(range(3, 33))", "eval seeds overlap dev"),
    ("rcatwin/grid.py", "P_LINK = (0.3, 0.5, 0.7, 0.9)", "P_LINK = (0.9, 0.9, 0.9, 0.9)", "p_link sweep collapsed"),
    ("scripts/aggregate.py", "        if not allow_partial:\n            raise SystemExit(2)", "        if not allow_partial:\n            pass", "aggregate accepts an incomplete grid"),
    ("scripts/aggregate.py", "        if have != want:", "        if not have:", "grid check only notices empty phases"),
]


def main():
    res = []
    for f, old, new, what in M:
        tmp = tempfile.mkdtemp(prefix="mut_")
        try:
            for d in ("rcatwin", "tests", "scripts"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
            p = os.path.join(tmp, f); src = open(p).read()
            assert src.count(old) == 1, f"{f}: mutation site not unique/found: {old}"
            open(p, "w").write(src.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *sys.argv[1:], "tests"],
                               cwd=tmp, env={**os.environ, "PYTHONPATH": tmp}, capture_output=True, text=True)
            killed = r.returncode != 0
        finally:
            shutil.rmtree(tmp)
        res.append(dict(file=f, mutant=new, breaks=what, killed=killed))
        print("KILLED  " if killed else "SURVIVED", f, "-", what, flush=True)
    n = sum(r["killed"] for r in res)
    print(f"{n}/{len(res)} mutants killed")
    if not sys.argv[1:]:
        json.dump({"killed": n, "total": len(res), "mutants": res}, open(os.path.join(ROOT, "results/mutation_check.json"), "w"), indent=1)
    sys.exit(0 if n == len(res) else 1)


if __name__ == "__main__":
    main()
