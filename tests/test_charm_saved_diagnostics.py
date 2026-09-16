"""Saved-score attribution, without model weights, GPU or retraining."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.charm_structure_audit.diagnose_saved import (
    answer_rankings,
    boundary_changes,
    compare_control,
    context_rankings,
    diagnose,
    onset_rows,
    ranking,
    read_predictions,
    rewiring_summary,
    token_roles,
    validate_predictions,
)


def sample(identity="1", labels=(0, 1, 1, 0, 1, 1, 0), scores=None):
    labels = np.asarray(labels, bool)
    positions = np.arange(len(labels))
    onset = labels & ~np.r_[False, labels[:-1]]
    boundaries = np.diff(np.r_[False, labels, False].astype(int))
    spans = np.stack((np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)), axis=1)
    if scores is None:
        scores = np.linspace(.1, .9, len(labels))
    return dict(id=identity, source_id="s" + identity, task="QA", generator="fixture",
                gold=labels, onset=onset, spans=spans, score=np.asarray(scores, float),
                offsets=np.stack((positions, positions + 1), axis=1),
                response="abcdefghijklmnopqrstuvwxyz"[:len(labels)], diagnostic={})


def test_first_and_later_onsets_are_not_continuation():
    value = sample()
    masks = token_roles(value)
    assert np.flatnonzero(masks["first_error"]).tolist() == [1]
    assert np.flatnonzero(masks["later_onset"]).tolist() == [4]
    assert np.flatnonzero(masks["continuation"]).tolist() == [2, 5]
    np.testing.assert_array_equal(
        masks["first_error"] | masks["later_onset"] | masks["continuation"], value["gold"]
    )


def test_normals_partition_clean_before_after():
    mixed = sample()
    clean = sample("2", (0, 0, 0))
    for value in (mixed, clean):
        masks = token_roles(value)
        parts = [masks[name] for name in ("normal_clean_answer", "normal_before_first", "normal_after_first")]
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in parts), masks["normal"].astype(int))


def test_role_auc_mixture_matches_direct_ranking():
    values = [sample(), sample("2", (0, 0, 0), (.1, .3, .7))]
    matrix = context_rankings(values, .8)
    numerator = 0
    count = 0
    for role in ("first_error", "later_onset", "continuation"):
        result = matrix[role]["normal"]
        numerator += result["positives"] * result["auroc"]
        count += result["positives"]
    assert numerator / count == pytest.approx(matrix["all_error"]["normal"]["auroc"])


def test_within_answer_is_distinct_from_pooled_and_pair_identity_holds():
    first = sample("1", (0, 1), (.1, .2))
    second = sample("2", (0, 1), (.8, .9))
    summary, rows = answer_rankings([first, second], .5)
    assert summary["macro_within_answer_auroc"] == 1
    assert summary["pooled"]["auroc"] == .75
    within = summary["within_answer_pairs"]
    cross = summary["cross_answer_pairs"]
    reconstructed = (within * summary["pair_weighted_within_auroc"] + cross * summary["cross_answer_auroc"]) / (within + cross)
    assert reconstructed == pytest.approx(summary["pooled"]["auroc"])
    assert len(rows) == 2


def test_constant_per_answer_shift_does_not_change_within_ranking():
    values = [sample("1", (0, 1), (.1, .2)), sample("2", (0, 1), (.1, .2))]
    before = answer_rankings(values, .5)[0]
    values[1]["score"] += .6
    after = answer_rankings(values, .5)[0]
    assert before["macro_within_answer_auroc"] == after["macro_within_answer_auroc"]
    assert before["pooled"]["auroc"] != after["pooled"]["auroc"]


def test_prefix_baseline_is_on_common_subset_not_full_population():
    value = sample(scores=(.1, .2, .9, .3, .8, .95, .85))
    value["score_prefix"] = np.array([.7, .1, np.nan, np.nan, .9, np.nan, np.nan])
    result = compare_control([value], "score_prefix", .8)
    measured = result["rankings"]["all_error"]
    assert result["common_tokens"] == 3
    assert measured["baseline"]["auroc"] == 1
    assert measured["control"]["auroc"] == .5
    assert measured["delta"]["auroc"] == -.5
    assert ranking(value["gold"], value["score"], .8)["auroc"] != measured["baseline"]["auroc"]
    assert result["changes"]["first_error"]["tokens"] == 1


def test_missing_prefix_observations_are_not_misses_or_normal_zeros():
    value = sample()
    value["score_prefix"] = np.full(7, np.nan)
    result = compare_control([value], "score_prefix", .8)
    assert result["common_tokens"] == 0
    assert result["rankings"]["all_error"]["baseline"]["auroc"] is None
    assert result["changes"]["first_error"]["lost_alarms"] == 0


def test_rewiring_reports_rr_denominator():
    value = sample()
    value["structure_in_rr"] = np.array([0, 0, 2, 0, 0, 0, 0])
    value["diagnostic"]["rewire_in"] = dict(changed_edges=1, edges=10)
    result = rewiring_summary([value])
    assert result["changed_fraction_all"] == .1
    assert result["changed_fraction_rr"] == .5


def test_ties_and_single_class_are_explicit():
    result = ranking([0, 1, 0, 1], [.8] * 4, .8)
    assert result["auroc"] == .5 and result["ap"] == .5
    assert result["true_positives"] == 0
    assert ranking([0, 0], [.1, .3], .8)["ap"] is None
    assert ranking([1, 1], [.1, .3], .8)["auroc"] is None


def test_all_onsets_exported_with_fixed_threshold_and_missing_controls():
    value = sample(scores=(.1, .2, .9, .3, .8, .95, .85))
    value["score_prefix"] = np.array([np.nan, .1, np.nan, np.nan, np.nan, np.nan, np.nan])
    rows = onset_rows([value], .8, ["score_prefix"])
    assert [row["role"] for row in rows] == ["first_error", "later_onset"]
    assert not rows[1]["hit"]  # Strict >, not >=.
    assert rows[1]["score_prefix"] is None
    assert rows[0]["score_prefix_delta"] == pytest.approx(-.1)


def test_boundary_score_changes_separate_entry_and_exit():
    value = sample(labels=(0, 1, 1, 0), scores=(.1, .2, .9, .8))
    result = boundary_changes([value], .85)
    assert result["NH"]["mean_delta"] == pytest.approx(.1)
    assert result["HN"]["mean_delta"] == pytest.approx(-.1)
    assert result["HH"]["gained_alarms"] == 1
    assert result["NN"]["tokens"] == 0


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "test"
    folder = root / "samples"
    folder.mkdir(parents=True)
    values = [sample(), sample("2", (0, 0, 0), (.1, .2, .3))]
    for value in values:
        record = {key: value[key] for key in ("id", "source_id", "task", "generator")}
        arrays = {key: value[key] for key in ("score", "gold", "onset", "spans", "offsets", "response")}
        # The reader must not load embeddings, including a non-numeric legacy payload.
        arrays["embedding"] = np.array([{"not_loaded": True}], dtype=object)
        np.savez_compressed(folder / (value["id"] + ".npz"), **arrays,
                            record_json=np.asarray(json.dumps(record)), diagnostic_json=np.asarray("{}"))
    settings = dict(threshold=dict(value=.8, origin="already_calibrated"), records=["1", "2"])
    (root / "prediction_settings.json").write_text(json.dumps(settings))
    (root / "predictions.json").write_text(json.dumps(["old/server/samples/1.npz", "old/server/samples/2.npz"]))
    (folder / "3.partial").write_text("incomplete")
    return root


def test_actual_saved_schema_and_separate_outputs_do_not_modify_inputs(saved):
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in saved.rglob("*") if path.is_file()}
    samples, settings = read_predictions(saved)
    result = diagnose(samples, settings, saved / "saved_diagnostics")
    assert result["responses"] == 2
    assert result["unique_onset_tokens"] == 2
    assert result["threshold"] == settings["threshold"]
    assert all(key not in samples[0] for key in ("embedding", "labels"))
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    assert (saved / "saved_diagnostics/onsets.csv").exists()


def test_completed_preview_ignores_partial_files(saved):
    (saved / "predictions.json").unlink()
    values, _ = read_predictions(saved, completed_only=True)
    assert len(values) == 2


def test_cli_runs_without_loading_torch_or_transformers(saved):
    code = "from experiments.charm_structure_audit.diagnose_saved import main; import sys; main(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code, "--predictions", str(saved)],
                            cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "later_onset" in result.stdout
    assert (saved / "saved_diagnostics/diagnosis.json").is_file()


def test_invalid_alignment_is_not_silently_padded():
    value = sample()
    value["score_prefix"] = np.zeros(3)
    with pytest.raises(ValueError, match="aligned"):
        validate_predictions([value])


def test_duplicate_ids_not_counted_twice():
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions([sample(), sample()])
