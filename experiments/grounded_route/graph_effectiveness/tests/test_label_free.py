import numpy as np

from experiments.grounded_route.graph_effectiveness.label_free import (
    procrustes_embedding_sensitivity,
)


def test_embedding_comparison_removes_global_rotation_shift_and_scale():
    random = np.random.default_rng(23)
    reference = random.normal(size=(40, 8))
    orthogonal, _ = np.linalg.qr(random.normal(size=(8, 8)))
    candidate = 3.0 * (reference @ orthogonal) + 7.0

    cosine, rmse = procrustes_embedding_sensitivity(reference, candidate)

    assert np.allclose(cosine, 1.0, atol=1e-5)
    assert np.allclose(rmse, 0.0, atol=1e-5)
