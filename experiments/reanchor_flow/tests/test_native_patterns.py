import numpy as np

from experiments.reanchor_flow.native_patterns import NativePatternModel


def _trace(seed=0, rows=32, layers=2, heads=3, rank=2):
    rng = np.random.default_rng(seed)
    return {
        "head_sketch": rng.normal(size=(layers, heads, rows, rank)).astype(np.float32),
        "mlp_sketch": rng.normal(size=(layers, rows, rank)).astype(np.float32),
        "row_position": np.arange(10, 10 + rows),
        "response_start": 11,
    }


def _fit(trace, **kwargs):
    head = trace["head_sketch"]
    return NativePatternModel.fit(
        NativePatternModel.vectorize(trace), (*head.shape[:2], head.shape[-1]),
        (head.shape[0], head.shape[-1]), **kwargs,
    )


def _prefix(trace, stop):
    return {**trace, "head_sketch": trace["head_sketch"][:, :, :stop],
            "mlp_sketch": trace["mlp_sketch"][:, :stop],
            "row_position": trace["row_position"][:stop]}


def test_layout_preserves_every_signed_head_coordinate_and_mlp():
    trace = _trace()
    matrix = NativePatternModel.vectorize(trace)
    for row in (0, 7, 31):
        np.testing.assert_array_equal(matrix[row, :12], trace["head_sketch"][:, :, row].ravel())
        np.testing.assert_array_equal(matrix[row, 12:], trace["mlp_sketch"][:, row].ravel())
    trace["hallucination_label"] = np.ones(32)
    np.testing.assert_array_equal(NativePatternModel.encode(trace), matrix)


def test_equal_norm_opposite_writes_are_separate_modes_without_head_averaging():
    trace = _trace(rows=20, layers=1, heads=2, rank=1)
    trace["head_sketch"][0, 0, :, 0] = np.r_[np.ones(10), -np.ones(10)]
    trace["head_sketch"][0, 1, :, 0] = -trace["head_sketch"][0, 0, :, 0]
    trace["mlp_sketch"][:] = 0
    model = _fit(trace, n_components=1, n_patterns=2)
    result = model.transform(trace)
    assert result["pattern_id"][0] != result["pattern_id"][-1]
    assert np.unique(np.linalg.norm(trace["head_sketch"], axis=-1)).size == 1
    np.testing.assert_array_equal(trace["head_sketch"].sum(axis=1), 0)
    centers = model.inverse_pattern_centers()["head_sketch"]
    np.testing.assert_allclose(centers[:, 0, 0], -centers[:, 0, 1], atol=1e-6)


def test_train_layer_scaling_preserves_relative_head_strength_and_row_amplitude():
    trace = _trace()
    trace["head_sketch"][:, 1] *= 10
    model = _fit(trace)
    scales = model.scale[:12].reshape(2, 3, 2)
    np.testing.assert_array_equal(scales, np.broadcast_to(scales[:, :1, :1], scales.shape))
    expected = np.sqrt(np.mean(trace["head_sketch"] ** 2, axis=(1, 2, 3)))
    np.testing.assert_allclose(scales[:, 0, 0], expected, rtol=1e-6)
    row = _prefix(trace, 1)
    doubled = {**row, "head_sketch": row["head_sketch"] * 2,
               "mlp_sketch": row["mlp_sketch"] * 2}
    first = model.transform(row)["coords"]
    second = model.transform(doubled)["coords"]
    assert not np.allclose(first, second)


def test_prefix_invariance_gap_mask_and_future_labels_cannot_change_scores():
    trace = _trace(seed=1)
    trace["row_position"][12:] += 4
    model = _fit(_trace(seed=2))
    whole, short = model.transform(trace), model.transform(_prefix(trace, 17))
    for key in whole:
        np.testing.assert_allclose(whole[key][:17], short[key], rtol=1e-6, atol=1e-6)
    assert whole["transition"][0] == whole["transition"][12] == 0
    assert not whole["has_previous"][0] and not whole["has_previous"][12]
    trace["hallucination_label"] = np.ones(32)
    trace["final_logit_margin"] = np.arange(32) * 1000
    trace["full_response_tokens"] = 1000000
    for key, value in model.transform(trace).items():
        np.testing.assert_array_equal(value, whole[key])


def test_test_transform_never_updates_training_statistics_or_parameters():
    model = _fit(_trace())
    before = {key: value.copy() for key, value in vars(model).items() if isinstance(value, np.ndarray)}
    shifted = _trace(seed=3)
    shifted["head_sketch"] += 100
    model.transform(shifted)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(model, key), value)


def test_full_basis_reconstructs_centers_and_exposes_original_units_loadings():
    trace = _trace(rows=60)
    model = _fit(trace, n_components=16, n_patterns=3)
    assert np.isclose(model.explained_variance_ratio.sum(), 1, atol=1e-6)
    assert model.transform(trace)["reconstruction_error"].max() < 3e-6
    centers = model.inverse_pattern_centers()
    packed = np.concatenate((centers["head_sketch"].reshape(3, -1),
                             centers["mlp_sketch"].reshape(3, -1)), axis=1)
    np.testing.assert_allclose((packed / model.scale - model.mean) @ model.components.T,
                               model.centers, atol=3e-6)
    loadings = model.component_loadings()
    assert loadings["head_sketch"].shape == (16, 2, 3, 2)
    delta = NativePatternModel.vectorize(trace)[1] - NativePatternModel.vectorize(trace)[0]
    packed_loading = np.concatenate((loadings["head_sketch"].reshape(16, -1),
                                     loadings["mlp_sketch"].reshape(16, -1)), axis=1)
    coords = model.transform(trace)["coords"]
    np.testing.assert_allclose(packed_loading @ delta, coords[1] - coords[0], atol=2e-6)


def test_discarded_direction_is_reported_separately_from_latent_novelty():
    trace = _trace(rows=64)
    model = _fit(trace, n_components=1)
    assert model.explained_variance_ratio.sum() < 0.5
    test = _prefix(trace, 1)
    null = np.ones(16, dtype=np.float32)
    null -= (null @ model.components.T) @ model.components
    raw_delta = null * model.scale * 100
    changed = {**test, "head_sketch": test["head_sketch"] + raw_delta[:12].reshape(2, 3, 1, 2),
               "mlp_sketch": test["mlp_sketch"] + raw_delta[12:].reshape(2, 1, 2)}
    a, b = model.transform(test), model.transform(changed)
    np.testing.assert_allclose(a["coords"], b["coords"], atol=5e-5)
    assert b["reconstruction_error"][0] > 20 * a["reconstruction_error"][0]


def test_constant_bank_and_pickle_free_roundtrip(tmp_path):
    trace = _trace()
    trace["head_sketch"][:] = 0
    trace["mlp_sketch"][:] = 0
    model = _fit(trace)
    assert len(model.centers) == 1
    np.testing.assert_array_equal(model.explained_variance_ratio, 0)
    path = tmp_path / "patterns.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    restored = NativePatternModel.load(path)
    for key, value in model.transform(_trace(seed=9)).items():
        np.testing.assert_array_equal(restored.transform(_trace(seed=9))[key], value)
