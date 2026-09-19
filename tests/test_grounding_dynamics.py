import numpy as np

from experiments.ragtruth_flow.grounding_dynamics import (
    LinearAccumulator,
    CovarianceAccumulator,
    transition_xy,
    save_reference_checkpoint,
    load_reference_checkpoint,
)


def test_grounding_transition_predictors_keep_head_identity():
    routes = np.zeros((3, 2, 4, 4), dtype=float)
    routes[0, :, :, 3] = 1
    routes[1, :, :, 0] = 2
    routes[1, :, :, 3] = 3
    routes[2, :, :, 2] = 4
    x, y = transition_xy(routes)
    assert x.shape == (2, 2, 16)
    assert y.shape == (2, 2, 4)
    np.testing.assert_array_equal(x[0, 0, :4], np.ones(4))
    np.testing.assert_array_equal(x[0, 0, 4:8], np.full(4, 2.0))
    np.testing.assert_array_equal(y[0, 0], np.full(4, 2.0))


def test_linear_accumulator_recovers_simple_unlabelled_dynamics():
    x = np.arange(40, dtype=float).reshape(10, 4)
    y = x[:, :2] * 2 + 3
    accumulator = LinearAccumulator(4, 2)
    accumulator.add(x, y, 1.0)
    weight, intercept = accumulator.fit(1e-8)
    prediction = x @ weight + intercept
    np.testing.assert_allclose(prediction, y, atol=1e-5)


def test_covariance_reference_is_finite_with_ridge():
    values = np.array([[1., 2.], [2., 3.], [3., 4.]])
    accumulator = CovarianceAccumulator(2)
    accumulator.add(values, 1.0)
    mean, precision = accumulator.fit(1e-3)
    assert np.isfinite(mean).all()
    assert np.isfinite(precision).all()


def test_reference_checkpoint_roundtrip(tmp_path):
    raw = [CovarianceAccumulator(2) for _ in range(2)]
    contrast = [CovarianceAccumulator(2) for _ in range(2)]
    for layer in range(2):
        values = np.array([[1. + layer, 2.], [3., 4. + layer]])
        raw[layer].add(values, .5)
        contrast[layer].add(values - values.mean(axis=1, keepdims=True), .5)

    path = tmp_path / "reference_checkpoint.npz"
    ids = ["a", "b", "c"]
    save_reference_checkpoint(path, raw, contrast, [0.1, 0.2], 2, ids)
    loaded_raw, loaded_contrast, jumps, next_index = load_reference_checkpoint(
        path, (2, 2), ids
    )

    assert next_index == 2
    assert jumps == [0.1, 0.2]
    for original, restored in zip(raw, loaded_raw):
        assert original.weight == restored.weight
        np.testing.assert_allclose(original.sum, restored.sum)
        np.testing.assert_allclose(original.outer, restored.outer)
    for original, restored in zip(contrast, loaded_contrast):
        assert original.weight == restored.weight
        np.testing.assert_allclose(original.sum, restored.sum)
        np.testing.assert_allclose(original.outer, restored.outer)


def test_reference_checkpoint_rejects_changed_training_ids(tmp_path):
    raw = [CovarianceAccumulator(2)]
    contrast = [CovarianceAccumulator(2)]
    path = tmp_path / "reference_checkpoint.npz"
    save_reference_checkpoint(path, raw, contrast, [], 0, ["a"])
    try:
        load_reference_checkpoint(path, (1, 2), ["b"])
    except ValueError as error:
        assert "response IDs changed" in str(error)
    else:
        raise AssertionError("changed TRAIN IDs must invalidate the checkpoint")
