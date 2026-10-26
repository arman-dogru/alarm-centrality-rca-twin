"""Run one or more phases of the grid (resumable: one gzip CSV per twin x condition under results/raw/<phase>/).
Usage: python scripts/run_grid.py --phase main [--workers 6]"""
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcatwin.experiment import run_incident
from rcatwin.faults import FaultParams, eligible_roots
from rcatwin.grid import cond_id, phases
from rcatwin.twin import make_twin

def task(args):
    phase, kind, n, seed, conds, n_inc, abl = args
    todo = [c for c in conds if not os.path.exists(f"results/raw/{phase}/{kind}_n{n}_s{seed}_{cond_id(c)}.csv.gz")]
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
        path = f"results/raw/{phase}/{kind}_n{n}_s{seed}_{cid}.csv.gz"
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip")
        os.replace(path + ".tmp", path)  # atomic: a crash never leaves a partial checkpoint
    print(f"{phase} {kind} n={n} seed={seed}: {len(todo)} conditions in {time.time() - t0:.0f}s", flush=True)
    return len(todo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    P = phases()
    for ph in a.phase:
        os.makedirs(f"results/raw/{ph}", exist_ok=True)
        tasks = [(ph, *t) for t in P[ph]]
        tasks.sort(key=lambda t: -t[2])  # big twins first
        with Pool(a.workers) as pool:
            done = sum(pool.imap_unordered(task, tasks))
        print(f"phase {ph}: {done} new condition files", flush=True)
