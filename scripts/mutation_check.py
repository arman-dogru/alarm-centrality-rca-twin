"""Single-line mutation check: each mutant is applied to a throw-away copy of rcatwin/ and tests/ (the working
tree is never touched) and `pytest -x tests` must fail. Writes results/mutation_check.json; exit 1 if any
mutant survives. Extra arguments go to pytest (then nothing is written)."""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = [  # (file, original, mutant, what it breaks)
    ("rcatwin/rca.py", "return int((s > r).sum() + (s == r).sum())", "return int((s > r).sum() + 1)", "optimistic ties"),
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
    ("rcatwin/triage.py", "    if f[\"hub_share\"] >= hub_share_min:\n        return \"star\"", "    if f[\"hub_share\"] >= hub_share_min:\n        return \"mesh\"", "star rule returns mesh"),
]


def main():
    res = []
    for f, old, new, what in M:
        tmp = tempfile.mkdtemp(prefix="mut_")
        try:
            for d in ("rcatwin", "tests"):
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
