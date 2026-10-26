import networkx as nx
import numpy as np

from rcatwin.stats import boot_means, cluster_weights, holm, signflip_p
from rcatwin import triage


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.6]), [1, 1])


def test_signflip_two_sided_and_null():
    d = np.random.default_rng(0).uniform(0.1, 1, 30)
    assert signflip_p(d, 2000, np.random.default_rng(1)) == signflip_p(-d, 2000, np.random.default_rng(1)) < 0.01
    assert signflip_p(np.zeros(10), 100, np.random.default_rng(0)) == 1.0


def test_cluster_weights_resample_whole_clusters():
    cl = np.repeat([0, 1, 2, 3], 5)
    W = cluster_weights(cl, 200, np.random.default_rng(0))
    assert np.all(W.sum(1) == 20)  # equal-size clusters: always N draws
    per = np.stack([W[:, cl == c].sum(1) for c in range(4)], 1)
    assert np.all(per % 5 == 0)  # each cluster contributes whole multiples of its size
    x = np.where(cl < 2, 1.0, 0.0)[None]
    b = boot_means(x, W)[0]
    assert set(np.round(b, 6)) <= {0.0, 0.25, 0.5, 0.75, 1.0} and 0.3 < b.mean() < 0.7  # only cluster-level noise


def test_features_and_rules():
    star = nx.star_graph(10); star.graph["layer"] = {v: (0 if v == 0 else 1) for v in star}
    f = triage.features(star, list(star))
    assert f["hub_share"] == 1.0 and f["cycle_rank"] == 0 and f["depth"] == 1 and f["diameter"] == 2
    grid = nx.grid_2d_graph(5, 5); grid = nx.convert_node_labels_to_integers(grid)
    grid.graph["layer"] = nx.single_source_shortest_path_length(grid, 0)
    g = triage.features(grid, list(grid))
    assert g["cycle_rank"] == (40 - 25 + 1) / 25 and g["diameter"] == 8
    assert triage.classify(g["cycle_rank"], g["deg_cv"], 0.3, 2.6) == "mesh"
    assert abs(f["deg_cv"] - 1.4230249470757708) < 1e-12  # star_graph(10): degrees 10, 1 x 10
    assert triage.classify(f["cycle_rank"], f["deg_cv"], 0.3, 1.4) == "star"
    assert triage.classify(f["cycle_rank"], f["deg_cv"], 0.3, 1.5) == "hierarchical"


def test_ci_is_95_percent():
    from rcatwin.stats import ci
    assert ci(np.arange(1001.0)) == (25.0, 975.0)


def test_bootstrap_unequal_clusters_mean_and_resampling():
    cl = np.repeat([0, 1, 2], [2, 10, 30])
    W = cluster_weights(cl, 300, np.random.default_rng(0))
    assert np.allclose(boot_means(np.ones((1, len(cl))), W), 1.0)  # a constant stays constant whatever the sizes
    per = np.stack([W[:, cl == c].sum(1) for c in range(3)], 1)
    assert (per == 0).any(0).all()  # every cluster is sometimes left out: clusters really are resampled
    S = cluster_weights(np.array([0, 0, 1, 1, 2, 2, 3, 3]), 200, np.random.default_rng(1), strata=np.array([0] * 4 + [1] * 4))
    assert np.all(S[:, :4].sum(1) == 4) and np.all(S[:, 4:].sum(1) == 4)  # stratified: 2 twins per stratum every time


def test_signflip_p_never_zero_and_twin_level():
    from rcatwin.stats import twin_signflip_p, twin_t_p
    d = np.ones(50)
    assert signflip_p(d, 999, np.random.default_rng(0)) == 1 / 1000
    cl = np.repeat(np.arange(5), 10)
    assert twin_signflip_p(d, cl, 10000, np.random.default_rng(0)) == 2 / 32  # exact, 5 twins
    assert twin_signflip_p(np.where(cl < 2, 1.0, -1.0), cl, 10000, np.random.default_rng(0)) > 0.5
    from scipy.stats import ttest_1samp
    d3 = np.r_[np.full(10, .2), np.full(10, .1), np.full(10, .3)]
    assert abs(twin_t_p(d3, np.repeat([0, 1, 2], 10)) - ttest_1samp([.2, .1, .3], 0).pvalue) < 1e-12  # 3 twins, df = 2


def test_pipeline_triage_rule_and_blog_reading():
    lab = triage.classify([0.5, 0.1, 0.1, 0.39], [3.0, 3.0, 1.0, 2.59], 0.4, 2.6)
    assert list(lab) == ["mesh", "star", "hierarchical", "hierarchical"]
    assert triage.SELECT == {"hierarchical": "ppr", "star": "alarm_degree", "mesh": "alarm_closeness"}
    assert triage.WEIGHTS["mesh"] == {"ppr": 0.2, "alarm_degree": 0.2, "alarm_closeness": 0.6}
    assert all(abs(sum(w.values()) - 1) < 1e-12 for w in triage.WEIGHTS.values())
