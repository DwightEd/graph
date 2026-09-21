import csv
import json

import numpy as np
import pytest

from experiments.unsupervised_token_graph.fixed_graph.tests.test_fixed_graph import (
    write_cache,
)
from experiments.unsupervised_token_graph.head_geometry.run import main


def fixture_arguments(tmp_path, output, tasks=("QA",)):
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    rows = []
    for task_index, task in enumerate(tasks):
        for split, identities in [("train", range(1, 7)), ("test", range(10, 12))]:
            for identity in identities:
                name = str(1000 * task_index + identity)
                write_cache(tmp_path / split / f"{name}.npz", identity=name,
                            task=task, split=split, seed=identity)
                rows.append(dict(id=name, source_id="source_" + name, split=split,
                                 response="abcdefghijkl", model="fixture",
                                 labels=[dict(start=2, end=6)]))
    (dataset / "response.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return ["--train-cache", str(tmp_path / "train"), "--test-cache", str(tmp_path / "test"),
            "--dataset", str(dataset), "--output", str(output), "--special-token-ids", "10",
            "--dimensions", "8", "--window", "4", "--bank-size", "64",
            "--tokens-per-source", "4", "--bootstrap", "4", "--threads", "1"]


def score_arrays(output):
    freeze = json.loads((output / "predictions/freeze.json").read_text())
    arrays = []
    for row in freeze["records"]:
        with np.load(output / "predictions" / row["file"]) as saved:
            arrays.append({name: saved[name] for name in freeze["methods"]})
    return arrays


def test_full_cli_fit_score_evaluate_and_resume(tmp_path):
    output = tmp_path / "run"
    argv = fixture_arguments(tmp_path, output)
    report = main(argv)
    assert report["primary"] == "all__conditional_energy"
    assert "all__independent_energy" in report["groups"]["ALL"]["primary_minus_control"]["all_error"]
    availability = report["groups"]["ALL"]["availability"]
    assert availability["scored_tokens"] == 20
    assert availability["full_window_tokens"] > 0
    metric = report["groups"]["ALL"]["views"]["all_error"]["all__log_moment"]
    assert metric["evaluated_tokens"] == 20
    assert (output / "predictions/metrics.csv").is_file()
    summary = (output / "predictions/task_summary.md").read_text()
    assert "|QA|已评估|2|" in summary
    assert "|Summary|未评估|—|" in summary
    before = score_arrays(output)
    freeze = json.loads((output / "predictions/freeze.json").read_text())
    with np.load(output / "predictions" / freeze["records"][0]["file"]) as saved:
        assert saved["conditional_energy__per_head"].shape[0] == len(saved["embedding_positions"])
    main([*argv, "--resume"])
    after = score_arrays(output)
    for first, second in zip(before, after):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    with pytest.raises(ValueError, match="identical geometry"):
        main([*argv, "--resume", "--window", "8", "--phase", "fit"])


def test_npz_labels_never_read_during_prepare_fit_or_score(tmp_path, monkeypatch):
    output = tmp_path / "run"
    argv = fixture_arguments(tmp_path, output)
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(archive, key):
        if key in ("labels", "hallucination_labels"):
            raise AssertionError("Natural labels entered detector computation")
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    for phase in ("prepare", "fit", "score"):
        main([*argv, "--phase", phase, *(["--resume"] if phase != "prepare" else [])])
    assert (output / "predictions/freeze.json").is_file()


def test_annotation_changes_leave_fitted_geometry_and_scores_identical(tmp_path):
    output = tmp_path / "first"
    argv = fixture_arguments(tmp_path, output)
    main(argv)
    before = score_arrays(output)
    annotation = tmp_path / "dataset/response.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    for row in rows:
        row["labels"] = []
    annotation.write_text("\n".join(map(json.dumps, rows)))
    other = tmp_path / "second"
    changed = argv.copy()
    changed[changed.index("--output") + 1] = str(other)
    report = main(changed)
    assert report["groups"]["ALL"]["views"]["all_error"]["all__raw"]["eligible_positives"] == 0
    for first, second in zip(before, score_arrays(other)):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    with np.load(output / "reference/0/geometry.npz") as first:
        with np.load(other / "reference/0/geometry.npz") as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])


def test_changed_test_observations_do_not_affect_reference_fit(tmp_path):
    output = tmp_path / "first"
    argv = fixture_arguments(tmp_path, output)
    for phase in ("prepare", "fit"):
        main([*argv, "--phase", phase, *(["--resume"] if phase == "fit" else [])])
    write_cache(tmp_path / "test/10.npz", identity="10", split="test", seed=999)
    other = tmp_path / "second"
    argv[argv.index("--output") + 1] = str(other)
    for phase in ("prepare", "fit"):
        main([*argv, "--phase", phase, *(["--resume"] if phase == "fit" else [])])
    for file in ("geometry.npz", "bank.npz"):
        with np.load(output / "reference/0" / file) as first:
            with np.load(other / "reference/0" / file) as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])


def test_overlapping_sources_cannot_enter_train_and_test(tmp_path):
    output = tmp_path / "run"
    argv = fixture_arguments(tmp_path, output)
    write_cache(tmp_path / "test/10.npz", identity="1", split="test")
    with pytest.raises(ValueError, match="source occurs in train and test"):
        main([*argv, "--phase", "prepare"])


def test_separate_task_reports_and_explicit_head_selection(tmp_path):
    tasks = ("QA", "Summary", "Data2txt")
    output = tmp_path / "middle"
    argv = fixture_arguments(tmp_path, output, tasks)
    report = main([*argv, "--tasks", *tasks, "--layers", "1", "--heads", "0"])
    assert report["head_selection"] == {"layers": [1], "heads": [0]}
    assert report["groups"]["ALL"]["answers"] == 6
    reference = json.loads((output / "reference/settings.json").read_text())
    source_groups = []
    for task in tasks:
        assert report["groups"][task]["answers"] == 2
        path = output / "predictions/tasks" / task / "metrics.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert {row["task"] for row in rows} == {task}
        assert {row["group"] for row in rows} == {task, task + "|fixture"}
        assert all(row["dataset"] == "RAGTruth" for row in rows)
        configured = reference["groups"][task + "|fixture"]
        source_groups.append(set(configured["reference_sources"]))
    assert not (source_groups[0] & source_groups[1] | source_groups[1] & source_groups[2]
                | source_groups[0] & source_groups[2])
