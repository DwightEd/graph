import numpy as np

from experiments.mechanism_validation.evaluation import cluster_bootstrap


def test_cluster_bootstrap_resamples_source_clusters_not_individual_tokens():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([.1, .9, .2, .8])
    sources = np.array(["a", "a", "b", "b"])

    result = cluster_bootstrap(labels, scores, sources, n_resamples=5, seed=3)

    assert result["auroc"].shape == (5,)
    assert result["valid_replicates"] <= 5
