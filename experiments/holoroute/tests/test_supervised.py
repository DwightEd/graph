import numpy as np

from experiments.holoroute.supervised import LinearProbe, fit_linear_probe


def test_linear_probe_recovers_a_separable_direction():
    random = np.random.default_rng(7)
    negative = random.normal(loc=-0.8, scale=0.5, size=(80, 12)).astype(np.float32)
    positive = random.normal(loc=0.8, scale=0.5, size=(40, 12)).astype(np.float32)
    feature = np.concatenate((negative, positive), axis=0)
    label = np.concatenate(
        (
            np.zeros(len(negative), dtype=np.int64),
            np.ones(len(positive), dtype=np.int64),
        )
    )

    probe = fit_linear_probe(feature, label, seed=9)
    score = probe.decision_function(feature)
    restored = LinearProbe.from_arrays(probe.arrays())

    assert score[label == 1].mean() > score[label == 0].mean()
    assert np.allclose(score, restored.decision_function(feature))
    assert probe.positive_tokens == len(positive)
    assert probe.negative_tokens == len(negative)
