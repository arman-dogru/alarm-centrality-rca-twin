"""Twin-clustered bootstrap, twin-level paired tests, an incident-level sign-flip (descriptive only) and Holm.

Incidents are not independent: they share a twin (and often a root), so every *test* is done on per-twin means
(twin_signflip_p, twin_t_p); the bootstrap resamples twins (optionally within topology strata) and then incidents."""
import numpy as np
from scipy import stats as _st


def cluster_weights(clusters, B, rng, strata=None):
    """(B, N) bootstrap multiplicities: resample clusters (twins) with replacement (within each stratum when
    `strata` is given, so every replicate keeps the same number of twins per topology), then incidents within each
    drawn cluster with replacement."""
    clusters = np.asarray(clusters)
    strata = np.zeros(len(clusters), int) if strata is None else np.asarray(strata)
    groups = []
    for s in np.unique(strata):
        ids = np.unique(clusters[strata == s])
        groups.append([np.flatnonzero((clusters == c) & (strata == s)) for c in ids])
    W = np.zeros((B, len(clusters)), np.float32)
    for b in range(B):
        for members in groups:
            for k in rng.integers(0, len(members), len(members)):
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


def twin_means(d, clusters):
    """Per-cluster (twin) means of per-incident values d, in sorted cluster order."""
    d, clusters = np.asarray(d, float), np.asarray(clusters)
    ids, inv = np.unique(clusters, return_inverse=True)
    return np.bincount(inv, weights=d) / np.bincount(inv)


def twin_signflip_p(d, clusters, flips, rng):
    """Two-sided sign-flip test on the per-twin mean differences (the twin is the unit of inference). Exact when
    2^#twins <= flips, otherwise Monte Carlo with p = (#{|null| >= |obs|} + 1) / (flips + 1)."""
    m = twin_means(d, clusters)
    if not np.any(m):
        return 1.0
    k, obs = len(m), abs(m.mean())
    if 2 ** k <= flips:
        signs = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1)
        return float(np.mean(np.abs((signs * m).mean(1)) >= obs - 1e-12))
    return signflip_p(m, flips, rng)


def twin_t_p(d, clusters):
    """Two-sided one-sample t-test of the per-twin mean differences (a secondary, parametric twin-level test)."""
    m = twin_means(d, clusters)
    if np.allclose(m, m[0]):
        return 1.0 if m[0] == 0 else 0.0
    return float(_st.ttest_1samp(m, 0.0).pvalue)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    k, run, out = len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * p[i]))
        out[i] = run
    return out
