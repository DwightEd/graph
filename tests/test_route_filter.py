"""Causality, physical-head identity, baseline parity and compact-cache reuse."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from state_audit.storage import read_arrays, read_json, write_json
from test_native_support import Tokenizer, response
from test_native_support import model as _model

from experiments.native_support.filter_evaluation import recovery_rows
from experiments.native_support.filter_features import cached_features
from experiments.native_support.filtering import (
    causal_filter,
    filter_scores,
    head_profile,
)
from experiments.native_support.optimize import (
    DIRECTORY,
    evaluate_optimization,
    run_optimization,
)
from experiments.native_support.pipeline import capture_response
from experiments.native_support.routes import route_scores
from experiments.native_support.run import main

model = _model


def test_profile_separates_native_source_history_self_and_controls():
    magnitude = np.array([[[2., 3., 5., 7., 11.]]])
    groups = np.array([2, 0, 0, 1, 1])
    profile = head_profile(magnitude, groups, 3, np.array([False, True, False]))
    np.testing.assert_allclose(profile[0, 0], np.array([3, 7, 11, 7, 0]) / 28)
    first = head_profile(magnitude[..., :3], groups[:3], 3, np.array([False, True, True]))
    np.testing.assert_allclose(first[0, 0], [.3, 0, 0, .7, 0])
    scaled = head_profile(magnitude * 17, groups, 3, np.array([False, True, False]))
    np.testing.assert_allclose(scaled, profile)
    empty = head_profile(np.zeros_like(magnitude), groups, 3, None)
    np.testing.assert_array_equal(empty[0, 0], [0, 0, 0, 0, 1])


def test_constant_states_reduce_exactly_to_causal_mean():
    values = np.array([.1, .4, -.2, 1., -.7])
    profile = np.tile([.2, .3, .1, .4, 0], (len(values), 2, 3, 1)) / 3
    result = causal_filter(values, profile, 3)
    expected = np.array([values[max(0, t - 2):t + 1].mean() for t in range(len(values))])
    np.testing.assert_allclose(result["route_state_filter"], expected)
    np.testing.assert_allclose(result["route_mean"], expected)
    np.testing.assert_allclose(result["filter_effective_tokens"], [1, 2, 3, 3, 3])
    np.testing.assert_allclose(causal_filter(values, profile, 1)["route_state_filter"], values)


def test_future_profiles_and_future_scores_cannot_change_current_estimate():
    random = np.random.default_rng(13)
    profile = random.dirichlet(np.ones(5), (30, 2, 4)) / 4
    values = random.normal(size=30)
    before = causal_filter(values, profile)
    altered, altered_values = profile.copy(), values.copy()
    altered[8:] = random.dirichlet(np.ones(5), (22, 2, 4)) / 4
    altered_values[8:] += 1000
    after = causal_filter(altered_values, altered)
    prefix = causal_filter(values[:8], profile[:8])
    for name in before:
        np.testing.assert_array_equal(before[name][:8], after[name][:8])
        np.testing.assert_array_equal(before[name][:8], prefix[name])


def test_changed_routing_state_reduces_stale_high_scores():
    profile = np.zeros((8, 1, 1, 5))
    profile[:, 0, 0, 1] = 1
    profile[-1, 0, 0] = [1, 0, 0, 0, 0]
    scores = np.r_[np.ones(7), -1.]
    result = filter_scores({"head_profile": profile, "route": scores}, "route", 8)
    assert result["filter_current_weight"][-1] == 1
    assert result["route_state_filter"][-1] == -1
    assert result["route_mean"][-1] == .75
    assert result["route_pooled_filter"][-1] == -1


def test_head_identity_changes_weights_without_changing_marginal_routing_score():
    random = np.random.default_rng(22)
    profile = random.dirichlet(np.ones(8), (8, 1)).reshape(8, 1, 2, 4)
    profile = np.pad(profile, ((0, 0), (0, 0), (0, 0), (0, 1)))
    scores = (profile[..., 1] + profile[..., 2] - profile[..., 0]).sum(-1).mean(-1)
    original = filter_scores({"head_profile": profile, "route": scores}, "route", 8)
    changed = profile.copy()
    changed[-1] = profile[-1, :, ::-1]
    swapped = filter_scores({"head_profile": changed, "route": scores}, "route", 8)
    assert original["route_state_filter"][-1] != pytest.approx(swapped["route_state_filter"][-1])
    np.testing.assert_allclose(original["route_pooled_filter"], swapped["route_pooled_filter"])
    permuted = causal_filter(scores, profile[:, :, ::-1], 8)
    np.testing.assert_allclose(original["route_state_filter"], permuted["route_state_filter"])


def test_joint_profile_retains_the_original_functional_route_budget():
    magnitude = np.random.default_rng(14).random((3, 4, 6))
    groups = np.array([2, 0, 0, 1, 1, 1])
    evidence = np.array([False, True, False])
    profile = head_profile(magnitude, groups, 3, evidence)
    np.testing.assert_allclose(profile.sum((-1, -2)), 1)
    reconstructed = (profile[..., 1] + profile[..., 2] - profile[..., 0]).sum(-1).mean()
    expected = route_scores(magnitude, magnitude, groups, 3, evidence)["routing_imbalance"]
    assert reconstructed == pytest.approx(expected)


def test_recovery_ranks_use_real_adjacent_normal_positions_only():
    record = {"id": "a", "target": np.array([0, 1, 2, 4, 5]),
              "labels": np.array([0, 1, 1, 0, 0]), "scores": np.array([[0], [1], [1], [1], [0]])}
    rows = [{"response_id": "a", "target": i, "token": "x"} for i in range(6)]
    assert recovery_rows([record], rows, {"route": "route"}) == []
    record["target"] = np.arange(5)
    recovered = recovery_rows([record], rows, {"route": "route"})
    assert len(recovered) == 1 and recovered[0]["target"] == 3
    assert recovered[0]["within_answer_normal_percentile"] == pytest.approx(5 / 6)


def test_cache_workflow_keeps_baseline_and_old_results_and_never_uses_labels(model, tmp_path):
    item = response()
    raw = tmp_path / "responses/0000"
    capture_response(model, Tokenizer(), item, raw, 3)
    settings = {"model": "unused", "responses": [item]}
    write_json(tmp_path / "settings.json", settings)
    write_json(tmp_path / "route_comparison_v2/evaluation.json", {"previous": True})
    annotations = {"sample": {"token_ids": item["token_ids"][7:], "labels": [0, 1, 1, 0, 0, 0]}}
    write_json(tmp_path / "annotations.json", annotations)
    original = (raw / "token_000000.npz").read_bytes()
    with patch("state_audit.model.load_model", side_effect=AssertionError("model load forbidden")):
        first = run_optimization(tmp_path, settings, window=4)
    directory = tmp_path / DIRECTORY / "w4"
    saved = read_arrays(directory / "responses/0000/scores.npz")
    for target in range(6):
        native = read_arrays(raw / f"token_{target:06d}.npz")
        expected = route_scores(native["attention"], np.sqrt(native["edge_value_energy"]), native["group_ids"], 7, None)
        assert saved["prompt_routing_imbalance"][target] == expected["prompt_routing_imbalance"]
    assert first["candidate_method"] == "route_state_filter"
    assert first["evaluation"]["methods"]["route_state_filter"]["all_error"]["positives"] == 2
    assert first["comparisons"]["all_error"][0]["candidate"] == "route_mean"
    assert read_json(tmp_path / "route_comparison_v2/evaluation.json") == {"previous": True}
    assert (raw / "token_000000.npz").read_bytes() == original
    annotations["sample"]["labels"] = [0, 0, 0, 1, 1, 0]
    write_json(tmp_path / "annotations.json", annotations)
    with patch("experiments.native_support.filter_features.extract_features", side_effect=AssertionError("raw cache reread forbidden")):
        run_optimization(tmp_path, settings, window=4)
    changed = read_arrays(directory / "responses/0000/scores.npz")
    for name in saved:
        np.testing.assert_array_equal(saved[name], changed[name])
    valid = (directory / "evaluation.json").read_bytes()
    missing = evaluate_optimization(tmp_path, tmp_path / "missing.json", window=4)
    assert missing["status"] == "unavailable"
    assert (directory / "evaluation.json").read_bytes() == valid
    altered = deepcopy(item)
    altered["token_ids"][0] += 1
    with pytest.raises(ValueError, match="token IDs differ"):
        cached_features(altered, raw, tmp_path / DIRECTORY / "features/0000.npz", None)


def test_cli_window_validation(tmp_path):
    with pytest.raises(SystemExit):
        main(["--stage", "optimize", "--window", "0", "--output", str(tmp_path)])
