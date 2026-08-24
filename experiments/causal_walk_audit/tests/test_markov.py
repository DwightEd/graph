import numpy as np

from experiments.causal_walk_audit.markov import NestedMarkovModel


def test_higher_order_model_beats_shuffled_path_blocks():
    rng = np.random.default_rng(7)
    rows = 800
    base = rng.normal(size=(rows, 4)).astype(np.float32)
    one = rng.normal(size=(rows, 3)).astype(np.float32)
    multi = rng.normal(size=(rows, 2)).astype(np.float32)
    order1 = base
    order2 = np.concatenate((base, one), axis=1)
    order3 = np.concatenate((base, one, multi), axis=1)
    target = (
        0.2 * base[:, :2]
        + 1.5 * one[:, :2]
        + 0.8 * multi
        + 0.02 * rng.normal(size=(rows, 2))
    ).astype(np.float32)

    model = NestedMarkovModel.fit(
        order1[:600],
        order2[:600],
        order3[:600],
        target[:600],
        alpha=0.1,
        seed=11,
    )
    summary = model.validation_summary(
        order1[600:],
        order2[600:],
        order3[600:],
        target[600:],
        seed=13,
    )
    assert summary["order2_gain"] > 0.2
    assert summary["order3_gain"] > 0.05
    assert summary["order2_path_gain"] > 0.2
    assert summary["order3_path_gain"] > 0.05
