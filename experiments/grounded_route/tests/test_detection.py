import inspect

import numpy as np

from experiments.grounded_route.detection import (
    PCAKNNConfig,
    PCAWhitenedKNN,
    fit,
    load_reference,
    save_reference,
)


def test_detector_consumes_only_embedding_and_is_deterministic(tmp_path):
    random = np.random.default_rng(7)
    embedding = random.normal(size=(80, 12)).astype(np.float32)
    config = PCAKNNConfig(
        components=6,
        neighbors=4,
        max_reference=20,
        seed=11,
    )

    first = fit(embedding, config)
    second = fit(embedding, config)
    assert set(inspect.signature(fit).parameters) == {"embedding", "config"}
    assert first.reference.shape == (20, 6)
    assert np.array_equal(first.reference, second.reference)
    assert np.array_equal(first.basis, second.basis)

    query = np.vstack((embedding[:5], np.full((1, 12), 20.0, dtype=np.float32)))
    score = first.score(query)
    assert score.shape == (6,)
    assert score[-1] > score[:-1].max()

    path = tmp_path / "detector.npz"
    save_reference(path, first, checkpoint_sha256="a" * 64)
    restored = load_reference(path)
    assert isinstance(restored, PCAWhitenedKNN)
    assert np.allclose(score, restored.score(query))
