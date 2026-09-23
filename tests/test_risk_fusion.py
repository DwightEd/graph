"""Direct risk readout, causal cache reuse, and honest paired evaluation."""

from unittest.mock import patch

import numpy as np
import pytest
from state_audit.storage import read_arrays, read_json, write_json
from test_joint_state import write_compact_fixture

from experiments.native_support.evaluate import ranking
from experiments.native_support.fusion import DIRECTORY, evaluate_fusion, run_fusion
from experiments.native_support.fusion_audit import paired_audit
from experiments.native_support.risk_envelope import (
    apply_envelope,
    fit_distribution,
    fit_envelope,
    percentile,
    reference_threshold,
    source_weights,
)
from experiments.native_support.run import main
from experiments.native_support.state_model import run_state_model


def reference_fixture():
    reference = {name: np.arange(20, dtype=float) for name in ("route", "route_state", "attention", "entropy")}
    sources = np.repeat(["a", "b"], 10)
    return reference, sources


def test_source_weighting_midrank_and_tied_threshold():
    sources = np.array(["a", "b", "b", "b"])
    weights = source_weights(sources)
    np.testing.assert_allclose(weights, [.5, 1 / 6, 1 / 6, 1 / 6])
    distribution = fit_distribution(np.array([0., 1., 1., 1.]), weights)
    np.testing.assert_allclose(percentile(np.array([-1., 0., .5, 1., 2.]), distribution), [0, .25, .5, .75, 1])
    tied = reference_threshold(np.ones(4), weights)
    assert tied["threshold"] == 1
    assert tied["reference_weighted_alarm_rate"] == 0


def test_direct_uncertainty_persistent_risk_and_duplicate_invariance():
    reference, sources = reference_fixture()
    fitted, _, _ = fit_envelope(reference, sources, "route", "attention")
    target = {"route": np.array([1., 19., 1.]), "route_state": np.array([1., 19., 19.]),
              "attention": np.array([1., 1., 19.]), "entropy": np.array([19., 1., 1.])}
    scores = apply_envelope(target, fitted, "route", "attention")
    # High entropy can now raise a low-route score; low entropy cannot erase sustained route risk.
    np.testing.assert_allclose(scores["risk_envelope"], [.975, .975, .975])
    assert scores["dominant_entropy"][0]
    assert scores["dominant_route_state"][1]
    assert scores["dominant_attention"][2] and scores["dominant_route_state"][2]
    duplicate = {**target, "attention": target["route_state"]}
    np.testing.assert_array_equal(apply_envelope(duplicate, fitted, "route", "attention")["risk_envelope"], scores["risk_envelope"])


def test_reference_ranks_are_monotone_unit_invariant_and_prefix_causal():
    reference, sources = reference_fixture()
    fitted, _, _ = fit_envelope(reference, sources, "route", "attention")
    target = {name: np.array([1., 4., 9., 16.]) for name in reference}
    before = apply_envelope(target, fitted, "route", "attention")
    changed = {name: values.copy() for name, values in target.items()}
    changed["entropy"][2:] = 100
    after = apply_envelope(changed, fitted, "route", "attention")
    np.testing.assert_array_equal(before["risk_envelope"][:2], after["risk_envelope"][:2])
    transformed = {name: (values + 1) ** 3 for name, values in reference.items()}
    fitted_again, _, _ = fit_envelope(transformed, sources, "route", "attention")
    target_again = {name: (values + 1) ** 3 for name, values in target.items()}
    np.testing.assert_array_equal(before["risk_envelope"], apply_envelope(target_again, fitted_again, "route", "attention")["risk_envelope"])


def test_paired_audit_reports_added_false_alarms_and_exact_auc_delta():
    labels = np.array([0, 1, 1, 0, 0])
    baseline = np.array([.1, .2, .9, .7, .1])
    candidate = np.array([.9, .9, .9, .2, .1])
    joined = {"labels": labels, "onsets": np.array([0, 1, 0, 0, 0], bool),
              "firsts": np.array([0, 1, 0, 0, 0], bool), "scores": np.column_stack((baseline, candidate))}
    thresholds = {name: {"threshold": .5} for name in ("route", "risk_envelope")}
    result, _, _, _ = paired_audit(joined, ("route", "risk_envelope"), "route", thresholds)
    all_errors = result["all_error"]
    assert all_errors["recovered_error_tokens"] == 1
    assert all_errors["added_normal_alarms"] == 1
    assert all_errors["removed_normal_alarms"] == 1
    expected = ranking(labels, candidate)["auroc"] - ranking(labels, baseline)["auroc"]
    assert all_errors["delta_auroc_from_error_ranks"] == pytest.approx(expected)
    joined["labels"][:] = 1
    joined["firsts"][:] = False
    absent_normal = paired_audit(joined, ("route", "risk_envelope"), "route", thresholds)[0]
    assert absent_normal["all_error"]["delta_auroc_from_error_ranks"] is None


def disjoint_fixture(tmp_path):
    output, reference = tmp_path / "target", tmp_path / "reference"
    settings, annotations = write_compact_fixture(output)
    reference_settings, _ = write_compact_fixture(reference)
    for response in reference_settings["responses"]:
        response["source_id"] = "reference_" + response["source_id"]
    write_json(reference / "settings.json", reference_settings)
    # A corrupt annotation file must be irrelevant to reference fitting.
    (reference / "annotations.json").write_text("not JSON")
    run_state_model(output, settings, reference_output=reference)
    return output, reference, settings, annotations


def test_cli_reuses_cache_freezes_scores_before_labels_and_preserves_controls(tmp_path, capsys):
    output, _reference, _settings, annotations = disjoint_fixture(tmp_path)
    old_path = output / "joint_state_v4/w16/responses/0000/scores.npz"
    original_bytes = old_path.read_bytes()
    with patch("experiments.native_support.filter_features.extract_features", side_effect=AssertionError("raw cache forbidden")), \
         patch("state_audit.model.load_model", side_effect=AssertionError("model forbidden")):
        main(["--stage", "fuse", "--output", str(output)])
    destination = output / DIRECTORY / "w16"
    scores = read_arrays(destination / "responses/0000/scores.npz")
    old_scores = read_arrays(old_path)
    for name in read_json(output / "joint_state_v4/w16/scoring_protocol.json")["methods"]:
        np.testing.assert_array_equal(scores[name], old_scores[name])
    assert old_path.read_bytes() == original_bytes
    assert "run_posterior" not in scores
    assert (destination / "report.html").is_file()
    summary = read_json(destination / "summary.json")
    assert summary["scored_tokens"] == 18
    assert summary["complementarity"]["status"] == "evaluated"
    assert summary["primary_baseline"] == "route_state"
    assert "percentile_route_state" in summary["evaluation"]["methods"]
    thresholds = summary["alarm_thresholds"]
    for item in annotations.values():
        item["labels"] = [1, 0, 0, 1, 0, 0]
    write_json(output / "annotations.json", annotations)
    run_fusion(output)
    after = read_arrays(destination / "responses/0000/scores.npz")
    for name in scores:
        np.testing.assert_array_equal(scores[name], after[name])
    assert read_json(destination / "scoring_protocol.json")["alarm_thresholds"] == thresholds
    main(["--stage", "evaluate", "--output", str(output)])
    assert "risk_envelope" in capsys.readouterr().out
    evaluation = (destination / "evaluation.json").read_bytes()
    assert evaluate_fusion(output, tmp_path / "missing.json")["status"] == "unavailable"
    assert (destination / "evaluation.json").read_bytes() == evaluation


def test_rejects_reference_overlap_prior_mismatch_and_stale_alignment(tmp_path):
    output, reference, settings, _ = disjoint_fixture(tmp_path)
    reference_settings = read_json(reference / "settings.json")
    reference_settings["responses"][0]["source_id"] = settings["responses"][0]["source_id"]
    write_json(reference / "settings.json", reference_settings)
    with pytest.raises(ValueError, match="entirely source-disjoint"):
        run_fusion(output)
    reference_settings["responses"][0]["source_id"] = "reference_source0"
    write_json(reference / "settings.json", reference_settings)
    protocol_path = output / "joint_state_v4/w16/scoring_protocol.json"
    protocol = read_json(protocol_path)
    protocol["priors_by_source"]["source0"]["mean"][0] += 1
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="reference differs"):
        run_fusion(output)
    protocol["priors_by_source"]["source0"]["mean"][0] -= 1
    # Recreate the exact floating point values rather than undoing a rounded sum.
    run_state_model(output, settings, reference_output=reference)
    settings["responses"][0]["token_ids"][-1] += 1
    write_json(output / "settings.json", settings)
    with pytest.raises(ValueError, match="prediction alignment"):
        run_fusion(output)
