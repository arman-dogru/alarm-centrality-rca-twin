"""RCA methods: baselines, the alarm-conditioned centrality trio, and the decompose -> cluster -> rank cascade.

Every method returns a score per node of the twin (higher = more likely root cause). Nodes a method cannot
rank get -inf. `rank_of` turns scores into the 1-based rank of the true root with pessimistic ties.
"""
from dataclasses import dataclass
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

NEG = -np.inf


def rank_of(scores, root, optimistic=False):
    """1-based rank of root; ties count against it (optimistic=True: ties count for it); a -inf root is ranked
    last among all nodes. With both, the expected Top-1 under random tie-breaking is
    [opt == 1] / (pess - opt + 1)."""
    s = np.asarray(scores, float)
    r = s[root]
    if r == NEG:
        return len(s)
    return int((s > r).sum() + (1 if optimistic else (s == r).sum()))


def expected_top1(pess, opt):
    """Expected Top-1 under uniformly random tie-breaking, from pessimistic and optimistic ranks."""
    return (np.asarray(opt) == 1) / (np.asarray(pess) - np.asarray(opt) + 1.0)


# --- context shared by all methods for one incident -------------------------------------------------
@dataclass
class Context:
    G: nx.Graph
    A: np.ndarray            # alarming node ids
    t: np.ndarray            # first alarm time per alarming node (aligned with A)
    n: int


def context(G, incident):
    A = np.array(sorted(incident.alarms), dtype=int)
    t = np.array([incident.alarms[a] for a in A], float)
    return Context(G, A, t, G.number_of_nodes())


def _full(n, idx, vals):
    s = np.full(n, NEG)
    s[np.asarray(idx, int)] = vals
    return s


# --- stage 0..2: candidate set ----------------------------------------------------------------------
def neighbourhood(G, A, hops):
    nodes = set(int(a) for a in A)
    frontier = set(nodes)
    for _ in range(hops):
        frontier = {u for v in frontier for u in G.neighbors(v)} - nodes
        nodes |= frontier
    return nodes


def cascade(ctx, hops=2, min_alarms=2, decompose=True, cluster=True, cover=0.8, seed=0):
    """Return (candidate node list, stats). Stage 0: alarm h-hop neighbourhood. Stage 1 (decompose): weakly
    connected components of it; drop components with < min_alarms alarms (unless none has that many).
    Stage 2 (cluster): Louvain inside each kept component; keep the communities with the most alarms until they
    cover `cover` of that component's alarms."""
    G, Aset = ctx.G, set(ctx.A.tolist())
    S0 = neighbourhood(G, ctx.A, hops)
    stats = {"s0": len(S0)}
    if not decompose:
        return sorted(S0), {**stats, "s1": len(S0), "s2": len(S0)}
    comps = [c for c in nx.connected_components(G.subgraph(S0))]
    cnt = [len(c & Aset) for c in comps]
    keep = [c for c, k in zip(comps, cnt) if k >= min_alarms] or [comps[int(np.argmax(cnt))]]
    S1 = set().union(*keep)
    stats["s1"] = len(S1)
    if not cluster:
        return sorted(S1), {**stats, "s2": len(S1)}
    S2 = set()
    for c in keep:
        H = G.subgraph(c)
        if len(c) <= 3:
            S2 |= set(c); continue
        comms = nx.community.louvain_communities(H, seed=seed)
        comms = sorted(comms, key=lambda m: (-len(m & Aset), -len(m), min(m)))
        need, got = cover * len(c & Aset), 0
        for m in comms:
            S2 |= set(m); got += len(m & Aset)
            if got >= need:
                break
    stats["s2"] = len(S2)
    return sorted(S2), stats


# --- centralities on a candidate subgraph ------------------------------------------------------------
def _adj(G, nodes):
    idx = {v: i for i, v in enumerate(nodes)}
    H = G.subgraph(nodes)
    r, c = zip(*[(idx[u], idx[v]) for u, v in H.edges]) if H.number_of_edges() else ((), ())
    m = len(nodes)
    M = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(m, m)).tocsr()
    return (M + M.T).tocsr(), idx


def _upstream_adj(G, nodes, idx):
    """Directed adjacency restricted to `nodes`: u -> v when v is one hop closer to a gateway."""
    L = G.graph["layer"]
    r, c = [], []
    for u in nodes:
        for v in G.neighbors(u):
            if v in idx and L[v] == L[u] - 1:
                r.append(idx[u]); c.append(idx[v])
    m = len(nodes)
    return sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(m, m)).tocsr()


def ppr(M, pers, alpha=0.85, iters=100, tol=1e-10):
    """Personalized PageRank by power iteration on adjacency M (row = source). Dangling mass and the restart
    both go to `pers`."""
    m = M.shape[0]
    p = pers / pers.sum() if pers.sum() > 0 else np.full(m, 1.0 / m)
    out = np.asarray(M.sum(1)).ravel()
    W = sp.diags(np.divide(1.0, out, out=np.zeros(m), where=out > 0)) @ M
    x = p.copy()
    for _ in range(iters):
        dangling = x[out == 0].sum()
        x_new = alpha * (W.T @ x + dangling * p) + (1 - alpha) * p
        if np.abs(x_new - x).sum() < tol:
            return x_new
        x = x_new
    return x


def trio(ctx, nodes, alpha=0.85, directed=False):
    """Alarm-conditioned PPR, alarm-degree and alarm-closeness on the candidate subgraph `nodes`."""
    nodes = list(nodes)
    M, idx = _adj(ctx.G, nodes)
    a_loc = np.array([idx[a] for a in ctx.A if a in idx], int)
    is_alarm = np.zeros(len(nodes)); is_alarm[a_loc] = 1
    # 1. PPR seeded on alarming nodes (undirected walk on the candidate subgraph; directed=True walks only
    #    upstream dependency edges, an ablation that was worse on the dev twins, see docs/pilots.md)
    Mw = _upstream_adj(ctx.G, nodes, idx) if directed else M
    if directed:  # nodes with no upstream edge inside the subgraph fall back to their undirected links
        no_up = np.asarray(Mw.sum(1)).ravel() == 0
        Mw = (Mw + sp.diags(no_up.astype(float)) @ M).tocsr()
    pr = ppr(Mw, is_alarm, alpha)
    # 2. degree counted only to alarming neighbours (+0.5 if the node alarms itself: tie-break)
    deg = M @ is_alarm + 0.5 * is_alarm
    # 3. closeness to alarming nodes only (unreachable distance = number of nodes)
    if len(a_loc):
        D = shortest_path(M, unweighted=True, indices=a_loc, directed=False)
        D[np.isinf(D)] = len(nodes)
        close = 1.0 / (1.0 + D.mean(0))
    else:
        close = np.zeros(len(nodes))
    n = ctx.n
    return {"ppr": _full(n, nodes, pr), "alarm_degree": _full(n, nodes, deg), "alarm_closeness": _full(n, nodes, close)}


def static_centralities(ctx, nodes):
    """Alarm-agnostic PageRank, degree and closeness on the same candidate subgraph (the 'static' control)."""
    nodes = list(nodes)
    M, _ = _adj(ctx.G, nodes)
    pr = ppr(M, np.ones(len(nodes)))
    deg = np.asarray(M.sum(1)).ravel()
    D = shortest_path(M, unweighted=True, directed=False)
    D[np.isinf(D)] = len(nodes)
    close = 1.0 / (1.0 + D.mean(0))
    n = ctx.n
    return {"static_pagerank": _full(n, nodes, pr), "static_degree": _full(n, nodes, deg),
            "static_closeness": _full(n, nodes, close)}


# --- baselines -------------------------------------------------------------------------------------
def earliest_alarm(ctx):
    """Temporal correlation: the earliest alarm is the cause."""
    return _full(ctx.n, ctx.A, -ctx.t)


def earliest_in(ctx, nodes):
    """Temporal correlation restricted to the cascade's candidate set (spurious alarms elsewhere are ignored)."""
    keep = set(nodes)
    m = np.array([a in keep for a in ctx.A], bool)
    return _full(ctx.n, ctx.A[m], -ctx.t[m])


def global_pagerank(G, cache={}):
    """Static PageRank over the whole twin (identical for every incident). Cached per twin under
    G.graph["twin_id"] = (kind, n, seed); graphs without a twin_id are never cached. (Keying on id(G) was a bug:
    a freed twin's id can be reused by the next twin in the same worker.)"""
    key = G.graph.get("twin_id")
    if key is None or key not in cache:
        nodes = list(G.nodes)
        M, _ = _adj(G, nodes)
        val = _full(len(nodes), nodes, ppr(M, np.ones(len(nodes))))
        if key is None:
            return val
        cache.clear(); cache[key] = val
    return cache[key]


def common_ancestor(ctx, frac=0.5):
    """Classic upstream-walk alarm correlation ("lowest common ancestor"): walk every alarm upstream along the
    dependency graph; a node's coverage is the share of alarms in its downstream cone (itself included). The
    deepest node (largest gateway layer) covering >= frac of the best coverage ranks first; then by coverage."""
    G, L = ctx.G, ctx.G.graph["layer"]
    count = {}
    for a in ctx.A:
        seen, frontier = {int(a)}, [int(a)]
        while frontier:  # walk upstream
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if L[v] == L[u] - 1 and v not in seen:
                        seen.add(v); nxt.append(v)
            frontier = nxt
        for v in seen:
            count[v] = count.get(v, 0) + 1
    idx = list(count)
    c = np.array([count[v] for v in idx], float)
    lay = np.array([L[v] for v in idx], float)
    ok = c >= frac * c.max()
    return _full(ctx.n, idx, np.where(ok, 1e3 + lay + 1e-3 * c / c.max(), c / c.max()))


def random_candidates(ctx, nodes, rng):
    return _full(ctx.n, list(nodes), rng.random(len(nodes)))


# --- ensembles -------------------------------------------------------------------------------------
def pct(s, nodes):
    """Percentile rank (0..1, ties averaged) of each candidate's score; -inf elsewhere."""
    from scipy.stats import rankdata
    nodes = list(nodes)
    v = s[nodes]
    out = np.full(len(s), NEG)
    out[nodes] = (rankdata(v) - 1) / max(1, len(v) - 1)
    return out


def weighted(trio_scores, nodes, w):
    nodes = list(nodes)
    out = np.full(len(next(iter(trio_scores.values()))), NEG)
    acc = np.zeros(len(nodes))
    for k, wk in w.items():
        if wk:
            acc += wk * pct(trio_scores[k], nodes)[nodes]
    out[nodes] = acc
    return out
