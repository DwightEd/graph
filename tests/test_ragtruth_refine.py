"""Ranking preservation, label isolation, resumable CPU scoring and portable reproduction."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest

from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from experiments.native_support.evaluate import ranking
from experiments.native_support.message_carriers.token_representation import aggregate_units
from experiments.native_support.ragtruth_benchmark.selection import development_sources
from experiments.native_support.ragtruth_refine import data, report, selection
from experiments.native_support.ragtruth_refine.run import main
from experiments.native_support.ragtruth_refine.scoring import (
    CHANNELS, PRIMARY, SELECTION_POOL, features, fit_scales, refine, score_answer,
)
from experiments.native_support.unified.calibration import fit_distribution


def test_tie_refinement_preserves_strict_anchor_order_and_exact_auc_decomposition():
    # Equal anchors occur in different units too: accounting must include those pairs.
    units = [dict(start=start, stop=start + 2) for start in range(0, 8, 2)]
    anchor = np.repeat([0., 0., 1e-20, 1e20], 2)
    observation = np.array([.1, .9, .8, .2, .4, .6, .7, .3])
    scale = fit_distribution(anchor[::2], ["a", "b", "c", "d"])
    refined = refine(anchor, observation, units, scale)["tie"]
    strict = anchor[:, None] < anchor[None, :]
    assert np.all((refined[:, None] < refined[None, :])[strict])
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    methods, columns = [], []
    for name in ("local", "full", "pair"):
        methods.append(f"source_{name}_unit_mean")
        columns.append(anchor)
        for channel in CHANNELS:
            methods.append(f"{name}_{channel}_tie")
            columns.append(refined)
    records = [dict(labels=labels, scores=np.column_stack(columns))]
    diagnostics = report.tie_accounting(records, methods)
    gain = ranking(labels, refined)["auroc"] - ranking(labels, anchor)["auroc"]
    assert diagnostics[PRIMARY]["predicted_auroc_gain"] == pytest.approx(gain)
    assert diagnostics[PRIMARY]["tied_positive_negative_pairs"] == 6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_baseline_roundoff_does_not_split_equal_anchor_blocks(dtype):
    rng = np.random.default_rng(7)
    units = [dict(start=0, stop=37), dict(start=37, stop=64)]
    observed = {name: rng.normal(size=64).astype(dtype) for name in
                ("source_local", "source_full", "raw_route", "raw_attention", "entropy")}
    observed["token_id"] = np.arange(64)
    row = dict(record=dict(source_id="one"), observed=observed, response=dict(units=units),
               features=features(observed, units, 16))
    scales = fit_scales([row])
    scores = score_answer(row, scales)
    pair = .5 * (aggregate_units(observed["source_local"], units) + aggregate_units(observed["source_full"], units))
    np.testing.assert_array_equal(scores["source_pair_unit_mean"], pair)
    np.testing.assert_array_equal(row["features"]["pair_anchor"], pair[[0, 37]])
    assert all(np.isfinite(value).all() for value in scores.values())


def make_cache(root):
    rng = np.random.default_rng(11)
    records, truth = [], {}
    for task in ("QA", "Summary", "Data2txt"):
        for source in range(7):
            for generator in ("a", "b"):
                identity = f"{task}-{source}-{generator}"
                record = dict(id=identity, source_id=f"{task}-{source}", task=task,
                    split="train" if source < 5 else "test", generator=generator,
                    directory=f"responses/{len(records):06d}", tokens=8)
                records.append(record)
                labels = [0, 0, 1, 1, 0, 1, 0, 0] if generator == "b" else [0] * 8
                observed = {name: rng.normal(size=8) for name in
                            ("source_local", "source_full", "raw_route", "raw_attention", "entropy")}
                observed["source_local"] += np.array(labels) * .5
                observed["token_id"] = np.arange(8) + 100
                units = [dict(start=0, stop=4), dict(start=4, stop=8)]
                observed["previous_selected"] = .5 * (aggregate_units(observed["source_local"], units)
                    + aggregate_units(observed["source_full"], units))
                directory = root / record["directory"]
                write_arrays(directory / "observations.npz", **observed)
                write_json(directory / "response.json", dict(answer_ids=observed["token_id"].tolist(), units=units))
                truth[identity] = dict(source_id=record["source_id"], token_ids=observed["token_id"].tolist(), labels=labels)
    development = {task: development_sources(records, task) for task in ("QA", "Summary", "Data2txt")}
    write_json(root / "manifest.json", dict(portable_refinement_cache=True, model="synthetic",
        records=records, selected_answers=len(records), development_sources=development, previous_selected=True,
        original_scope=dict(answers=len(records))))
    write_json(root / "coverage.json", dict(status="complete"))
    write_json(root / "annotations.json", truth)
    return records, truth


def test_full_three_task_pipeline_resume_and_test_labels_cannot_affect_selection(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    original_records, truth = make_cache(source)
    args = ["--input", str(source), "--output", str(output), "--select-on-train"]
    source_bytes = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with patch.object(selection, "read_annotations", side_effect=AssertionError("Scoring read labels")), \
            patch.object(report, "read_annotations", side_effect=AssertionError("Scoring evaluated")):
        main([*args, "--stage", "score"])
    selected_manifest = read_json(output / "manifest.json")
    assert len(selected_manifest["records"]) == 18  # 20% development sources + all test.
    read_truth = selection.read_annotations
    def train_only(reader, manifest, records):
        assert all(record["split"] == "train" for record in records)
        assert read_json(output / "coverage.json")["status"] == "complete"
        return read_truth(reader, manifest, records)
    with patch.object(selection, "read_annotations", train_only):
        main(["--output", str(output), "--stage", "select"])
    main(["--output", str(output), "--stage", "evaluate"])
    choices = read_json(output / "selection.json")
    summary = read_json(output / "summary.json")
    assert {row["task"] for row in summary["test_by_dataset"]} == {"QA", "Summary", "Data2txt"}
    assert len(read_json(output / "evaluation.json")["groups"]) == 9
    assert all(path.read_bytes() == source_bytes[str(path.relative_to(source))]
               for path in source.rglob("*") if path.is_file())
    timestamps = {path: path.stat().st_mtime_ns for path in output.rglob("scores.npz")}
    for record in original_records:
        if record["split"] == "test":
            truth[record["id"]]["labels"] = [1 - value for value in truth[record["id"]]["labels"]]
    write_json(source / "annotations.json", truth)
    with patch.object(data, "load_row", side_effect=AssertionError("Resume recomputed scores")):
        main([*args, "--resume"])
    assert read_json(output / "selection.json") == choices
    assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in timestamps.items())
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--window", "8"])


def test_exported_task_cache_reproduces_scores_selection_and_metrics(tmp_path):
    source, output, restored = tmp_path / "input", tmp_path / "output", tmp_path / "restored"
    make_cache(source)
    main(["--input", str(source), "--output", str(output), "--select-on-train", "--tasks", "QA"])
    archive = tmp_path / "output_QA_cache.zip"
    with ZipFile(archive) as bundle:
        assert "annotations.json" in bundle.namelist()
        assert not any(name.endswith("scores.npz") for name in bundle.namelist())
    main(["--input", str(archive), "--output", str(restored), "--select-on-train", "--tasks", "QA"])
    assert read_json(restored / "selection.json") == read_json(output / "selection.json")
    assert read_json(restored / "evaluation.json") == read_json(output / "evaluation.json")
    for path in output.rglob("scores.npz"):
        original = read_arrays(path)
        reproduced = read_arrays(restored / path.relative_to(output))
        for name in original:
            np.testing.assert_array_equal(original[name], reproduced[name])


def test_partial_score_resume_recomputes_only_missing_answers(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    make_cache(source)
    args = ["--input", str(source), "--output", str(output), "--tasks", "QA", "--stage", "score"]
    main(args)
    paths = list(output.rglob("scores.npz"))
    paths[0].unlink()
    kept = {path: path.stat().st_mtime_ns for path in paths[1:]}
    original_score = data.score_answer
    calls = []
    def counted(row, scales):
        calls.append(row["record"]["id"])
        return original_score(row, scales)
    with patch.object(data, "score_answer", counted):
        main([*args, "--resume"])
    assert len(calls) == 1
    assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in kept.items())


def test_summary_only_input_fails_before_creating_output(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    make_cache(source)
    for path in source.rglob("observations.npz"):
        path.unlink()
    with pytest.raises(ValueError, match="summaries only"):
        main(["--input", str(source), "--output", str(output)])
    assert not output.exists()


def test_development_ties_prefer_baseline_using_exact_pair_counts(tmp_path):
    labels = np.repeat([0, 1, 0, 1, 1, 0, 1], 3)
    anchor = np.repeat(np.arange(7), 3).astype(float)
    record = dict(id="one", source_id="source", task="QA", split="train", directory="responses/one", tokens=21)
    # Break only same-class ties, leaving every positive-negative comparison unchanged.
    refined = anchor + np.tile([-.1, 0., .1], 7)
    values = {name: refined for name in SELECTION_POOL}
    values[SELECTION_POOL[0]] = anchor
    write_arrays(tmp_path / record["directory"] / "scores.npz", token_id=np.arange(21), **values)
    manifest = dict(records=[record], development_sources=dict(QA=["source"]))
    truth = dict(one=dict(token_ids=np.arange(21), labels=labels))
    with patch.object(selection, "read_annotations", return_value=truth):
        selected = selection.choose_task(tmp_path, None, manifest, "QA")
    assert selected["method"] == SELECTION_POOL[0]
    assert len({trial["concordant_pairs"] for trial in selected["trials"]}) == 1
