import numpy as np

from experiments.grounded_route.graph_effectiveness.upper_bound import source_group_folds


def test_outer_folds_are_source_disjoint_and_complete():
    source = np.repeat([f"source-{index}" for index in range(10)], 4)
    label = np.tile([0, 0, 1, 0], 10)
    fold_id = source_group_folds(label, source, folds=5, seed=17)

    assert set(fold_id.tolist()) == set(range(5))
    for fold in range(5):
        test_sources = set(source[fold_id == fold])
        train_sources = set(source[fold_id != fold])
        assert test_sources.isdisjoint(train_sources)
