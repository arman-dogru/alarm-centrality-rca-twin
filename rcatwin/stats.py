"""Cluster (twin) x incident bootstrap, paired sign-flip test and Holm correction."""
import numpy as np


def cluster_weights(clusters, B, rng):
    """(B, N) bootstrap multiplicities: resample clusters (twins) with replacement, then incidents within each
    drawn cluster with replacement."""
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in ids]
    W = np.zeros((B, len(clusters)), np.float32)
    for b in range(B):
        for k in rng.integers(0, len(ids), len(ids)):
            m = members[k]
            np.add.at(W[b], m[rng.integers(0, len(m), len(m))], 1)
    return W


def boot_means(X, W):
    """X: (M, N) per-incident values for M methods; returns (M, B) bootstrap means."""
    return (np.asarray(X, np.float64) @ W.T.astype(np.float64)) / W.sum(1)


def ci(boot, lo=2.5, hi=97.5):
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def signflip_p(d, flips, rng):
    """Two-sided Monte-Carlo sign-flip test of mean(d) = 0; p = (#{|null| >= |obs|} + 1) / (flips + 1)."""
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    obs = d.mean()
    null = np.empty(flips)
    for s in range(0, flips, 1000):
        k = min(1000, flips - s)
        null[s:s + k] = (rng.choice([-1.0, 1.0], (k, len(d))) * d).mean(1)
    return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (flips + 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    k, run, out = len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * p[i]))
        out[i] = run
    return out
