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
    assert triage.classify(g, cycle_mesh_min=0.3) == "mesh"
    assert triage.classify(f, hub_share_min=0.5) == "star"
