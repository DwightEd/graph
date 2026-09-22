"""Probability recursion and cache integration; no natural performance claims."""

from copy import deepcopy
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from test_native_support import response

from experiments.native_support.routes import ROUTE_SCORES
from experiments.native_support.run import main
from experiments.native_support.state_inputs import reference_moments
from experiments.native_support.state_model import (
    DIRECTORY,
    evaluate_state_model,
    run_state_model,
)
from experiments.native_support.switching import (
    predictive_logpdf,
    prior_state,
    switching_filter,
    update_state,
)


def test_predictive_density_matches_independent_student_distribution():
    mean = np.array([.1, -.2, .5])
    covariance = np.array([[.2, .1, 0], [.1, .3, .02], [0, .02, .4]])
    state = update_state(np.array([.3, -.1, 1]), prior_state(mean, covariance))
    value = np.array([-.4, .2, .3])
    degrees = state["degrees"][0] - 3 + 1
    count = state["count"][0]
    shape = state["scale"][0] * (count + 1) / (count * degrees)
    expected = multivariate_t.logpdf(value, loc=state["mean"][0], shape=shape, df=degrees)
    assert predictive_logpdf(value, state)[0] == pytest.approx(expected)


def enumerated_posterior(values, mean, covariance, hazard):
    lengths, joint_logs = [], []
    for changes in product((False, True), repeat=len(values) - 1):
        state = prior_state(mean, covariance)
        log_joint, length = 0., 0
        for index, value in enumerate(values):
            if index:
                changed = changes[index - 1]
                log_joint += np.log(hazard if changed else 1 - hazard)
                if changed:
                    state, length = prior_state(mean, covariance), 0
            log_joint += predictive_logpdf(value, state)[0]
            state = update_state(value, state)
            length += 1
        lengths.append(length)
        joint_logs.append(log_joint)
    probabilities = np.exp(joint_logs - logsumexp(joint_logs))
    return np.bincount(lengths, weights=probabilities, minlength=len(values) + 1)[1:]


def test_forward_filter_matches_all_possible_segmentations():
    values = np.random.default_rng(2).normal(size=(6, 3))
    mean, covariance = np.zeros(3), np.eye(3)
    result = switching_filter(values, mean, covariance, 4)
    for target in range(len(values)):
        exact = enumerated_posterior(values[:target + 1], mean, covariance, .25)
        np.testing.assert_allclose(result["run_posterior"][target, :target + 1], exact, atol=1e-12)
    np.testing.assert_allclose(result["run_posterior"].sum(1), 1)
    np.testing.assert_allclose(result["reset_contribution"] + result["continuation_contribution"], result["state_mean"][:, 0])


def test_causal_prefix_and_affine_observation_units():
    values = np.random.default_rng(3).normal(size=(25, 3))
    before = switching_filter(values, np.zeros(3), np.eye(3))
    changed = values.copy()
    changed[8:] += 100
    after = switching_filter(changed, np.zeros(3), np.eye(3))
    prefix = switching_filter(values[:8], np.zeros(3), np.eye(3))
    for name in before:
        np.testing.assert_array_equal(before[name][:8], after[name][:8])
        expected = before[name][:8, :8] if name == "run_posterior" else before[name][:8]
        np.testing.assert_allclose(expected, prefix[name])
    scale = np.diag([1, 10, .1])
    transformed = switching_filter(values @ scale, np.zeros(3), scale @ scale)
    np.testing.assert_allclose(before["run_posterior"], transformed["run_posterior"], atol=1e-12)
    np.testing.assert_allclose(before["state_mean"][:, 0], transformed["state_mean"][:, 0])


def test_auxiliary_views_change_memory_without_inventing_risk_direction():
    values = np.zeros((30, 3))
    values[-1] = [.2, 3, 3]
    joint = switching_filter(values, np.zeros(3), np.eye(3))
    route = switching_filter(values[:, :1], np.zeros(1), np.eye(1))
    assert joint["reset_probability"][-1] > route["reset_probability"][-1]
    assert joint["state_mean"][-1, 0] > route["state_mean"][-1, 0]
    values[:, 0] = 0
    no_route_signal = switching_filter(values, np.zeros(3), np.eye(3))
    np.testing.assert_array_equal(no_route_signal["state_mean"][:, 0], 0)
    values[:, :2] = .8
    values[:, 2] = 0  # Confident sustained routing does not need high entropy.
    confident = switching_filter(values, np.zeros(3), np.eye(3))
    assert confident["state_mean"][-1, 0] > .7
    independent = switching_filter(values, np.zeros(3), np.eye(3), expected_run=1)
    np.testing.assert_allclose(independent["state_mean"], values / 2)
    np.testing.assert_allclose(independent["reset_probability"], 1)


def test_reference_excludes_whole_source_and_weights_sources_equally():
    records = [{"response": {"id": str(i), "source_id": source}, "observations": np.full((count, 3), value)}
               for i, (source, count, value) in enumerate((("held", 9, 100), ("held", 7, -100), ("a", 2, 1), ("b", 100, 3)))]
    prior = reference_moments(records, "held")
    assert prior["source_ids"] == ["a", "b"]
    np.testing.assert_allclose(prior["mean"], 2)
    records[0]["observations"][:] = 10000
    assert reference_moments(records, "held") == prior
    with pytest.raises(ValueError, match="independent reference source"):
        reference_moments(records[:2], "held")


def write_compact_fixture(output):
    responses, annotations = [], {}
    random = np.random.default_rng(4)
    for index in range(3):
        item = {**response(), "id": str(index), "source_id": f"source{index}"}
        values = {name: random.uniform(-.6, .6, 6) for name in ROUTE_SCORES}
        values.update(entropy=random.uniform(0, 3, 6), target=np.arange(6), query=np.arange(6, 12),
                      token_id=np.array(item["token_ids"][7:]), all_token_ids=np.array(item["token_ids"]),
                      source_mask=np.empty(0, bool), ledger_error=np.zeros(6))
        write_arrays(output / "route_filter_v3/features" / f"{index:04d}.npz", **values)
        responses.append(item)
        annotations[item["id"]] = {"token_ids": item["token_ids"][7:], "labels": [0, 1, 1, 0, 0, 0]}
    settings = {"model": "unused", "responses": responses}
    write_json(output / "settings.json", settings)
    write_json(output / "annotations.json", annotations)
    return settings, annotations


def test_cli_uses_compact_cache_and_labels_only_after_scoring(tmp_path, capsys):
    settings, annotations = write_compact_fixture(tmp_path)
    old = tmp_path / "route_filter_v3/w16/evaluation.json"
    write_json(old, {"keep": True})
    with patch("experiments.native_support.filter_features.extract_features", side_effect=AssertionError("raw cache forbidden")), \
         patch("state_audit.model.load_model", side_effect=AssertionError("model loading forbidden")):
        main(["--stage", "model", "--output", str(tmp_path)])
    destination = tmp_path / DIRECTORY / "w16"
    summary = read_json(destination / "summary.json")
    assert summary["scored_tokens"] == 18
    assert summary["primary_baseline"] == "route_mean"
    assert summary["evaluation"]["methods"]["joint_state"]["all_error"]["tokens"] == 18
    saved = read_arrays(destination / "responses/0000/scores.npz")
    for item in annotations.values():
        item["labels"] = [1, 0, 0, 0, 1, 0]
    write_json(tmp_path / "annotations.json", annotations)
    run_state_model(tmp_path, settings)
    changed = read_arrays(destination / "responses/0000/scores.npz")
    for name in saved:
        np.testing.assert_array_equal(saved[name], changed[name])
    valid = (destination / "evaluation.json").read_bytes()
    unavailable = evaluate_state_model(tmp_path, tmp_path / "missing.json")
    assert unavailable["status"] == "unavailable"
    assert (destination / "evaluation.json").read_bytes() == valid
    assert read_json(old) == {"keep": True}
    main(["--stage", "evaluate", "--output", str(tmp_path)])
    assert "joint_state" in capsys.readouterr().out


def test_external_reference_is_frozen_and_checks_model_identity(tmp_path):
    settings, _ = write_compact_fixture(tmp_path / "target")
    reference, _ = write_compact_fixture(tmp_path / "reference")
    result = run_state_model(tmp_path / "target", settings, reference_output=tmp_path / "reference")
    assert result["reference_mode"] == "external_reference_with_overlapping_sources_excluded"
    assert result["priors_by_source"]["source0"]["source_ids"] == ["source1", "source2"]
    incompatible = deepcopy(reference)
    incompatible["model"] = "different_observer"
    write_json(tmp_path / "reference/settings.json", incompatible)
    with pytest.raises(ValueError, match="observer models differ"):
        run_state_model(tmp_path / "target", settings, reference_output=tmp_path / "reference")
