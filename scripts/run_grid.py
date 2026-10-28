"""Run one or more phases of the grid (resumable: one gzip CSV per twin x condition under <out>/<phase>/).

Usage: python scripts/run_grid.py --phase main [--workers 6] [--out results/raw] [--force]
  --out    write somewhere else (e.g. a scratch dir) to recompute from scratch without touching committed files
  --force  recompute condition files even if they already exist (they are overwritten atomically)"""
import argparse, os, sys, time, zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rcatwin.experiment import run_incident
from rcatwin.faults import FaultParams, eligible_roots
from rcatwin.grid import cond_id, phases
from rcatwin.twin import make_twin


def task(args):
    out, force, phase, kind, n, seed, conds, n_inc, abl = args
    path = lambda c: os.path.join(out, phase, f"{kind}_n{n}_s{seed}_{cond_id(c)}.csv.gz")
    todo = [c for c in conds if force or not os.path.exists(path(c))]
    if not todo:
        return 0
    t0 = time.time()
    G = make_twin(kind, n, seed)
    roots = eligible_roots(G)
    for c in todo:
        rows = []
        cid = cond_id(c)
        for i in range(n_inc):
            rng = np.random.default_rng([seed, zlib.crc32(f"{kind}|{n}|{cid}".encode()), i])
            root = int(rng.choice(roots))
            r = run_incident(G, root, FaultParams(**c), rng, ablations=abl, seed=seed)
            rows.append({"phase": phase, "kind": kind, "n": n, "seed": seed, "incident": i, **c, **r})
        p = path(c)
        pd.DataFrame(rows).to_csv(p + ".tmp", index=False, compression="gzip")
        os.replace(p + ".tmp", p)  # atomic: a crash never leaves a partial checkpoint
    print(f"{phase} {kind} n={n} seed={seed}: {len(todo)} conditions in {time.time() - t0:.0f}s", flush=True)
    return len(todo)


if __name__ == "__main__":
    P = phases()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", nargs="+", required=True, choices=sorted(P))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for ph in a.phase:
        os.makedirs(os.path.join(a.out, ph), exist_ok=True)
        tasks = [(a.out, a.force, ph, *t) for t in P[ph]]
        tasks.sort(key=lambda t: -t[4])  # big twins first
        with Pool(a.workers) as pool:
            done = sum(pool.imap_unordered(task, tasks))
        print(f"phase {ph}: {done} new condition files", flush=True)
