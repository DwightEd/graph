import json
import tarfile

import numpy as np
import pytest

from experiments.unsupervised_token_graph.head_geometry.run import main
from experiments.unsupervised_token_graph.head_geometry.tests.test_pipeline import (
    fixture_arguments,
    score_arrays,
)


def test_reuse_three_tasks_freeze_bridge_archive_and_resume(tmp_path, monkeypatch):
    old = tmp_path / "old"
    tasks = ("QA", "Summary", "Data2txt")
    argv = fixture_arguments(tmp_path, old, tasks)
    main([*argv, "--tasks", *tasks])
    output = tmp_path / "cross"
    changed = [
        *argv,
        "--output",
        str(output),
        "--suite",
        "cross_terms",
        "--observations-from",
        str(old),
        "--save-embeddings",
    ]
    for split in ("train", "test"):
        (tmp_path / split).rename(tmp_path / (split + "_unavailable"))

    def no_tokenizer(*args):
        pytest.fail("Reused observations should not load a tokenizer")

    monkeypatch.setattr(
        "experiments.unsupervised_token_graph.head_geometry.run.special_ids", no_tokenizer
    )
    report = main(changed)
    assert report["primary"] == "all__pair_full"
    assert report["head_selection"] == {"layers": [0, 1], "heads": [0, 1]}
    assert output.joinpath("observations").resolve() == old.joinpath("observations").resolve()
    old_settings = json.loads((old / "reference/settings.json").read_text())
    new_settings = json.loads((output / "reference/settings.json").read_text())
    for task in tasks:
        assert report["groups"][task]["answers"] == 2
        group = task + "|fixture"
        assert (
            old_settings["groups"][group]["bank_rows"] == new_settings["groups"][group]["bank_rows"]
        )
        directory = new_settings["groups"][group]["directory"]
        with np.load(old / "reference" / directory / "bank.npz") as first:
            with np.load(output / "reference" / directory / "bank.npz") as second:
                for key in first.files:
                    if key.startswith(("all__raw___", "all__moment___")):
                        np.testing.assert_array_equal(first[key], second[key])
        differences = report["groups"][task]["primary_minus_control"]
        assert "all__pair_state_smooth" in differences["full_window"]
        assert (
            "all__pair_cross_minus_all__pair_state" in report["groups"][task]["planned_comparisons"]
        )
    with np.load(output / "reference/0/geometry.npz") as saved:
        assert "precision" not in saved.files
    archive = tmp_path / "cross_review.tar.gz"
    with tarfile.open(archive) as saved:
        names = saved.getnames()
        assert "cross/observations/manifest.json" in names
        assert "cross/predictions/comparisons.csv" in names
        assert not any(name.endswith("bank.npz") for name in names)
        assert not any(
            name.startswith("cross/observations/") and name.endswith(".npz") for name in names
        )
    before = score_arrays(output)
    main([*changed, "--resume"])
    for first, second in zip(before, score_arrays(output)):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


def test_cross_terms_fit_and_scores_are_independent_of_natural_labels(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    argv = fixture_arguments(tmp_path, prepared)
    main([*argv, "--phase", "prepare"])
    original = np.lib.npyio.NpzFile.__getitem__

    def reject_labels(archive, key):
        if key in ("labels", "hallucination_labels"):
            pytest.fail("Natural labels entered reference fit or scoring")
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", reject_labels)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        changed = [
            *argv,
            "--suite",
            "cross_terms",
            "--observations-from",
            str(prepared),
            "--output",
            str(output),
        ]
        main([*changed, "--phase", "fit"])
        main([*changed, "--phase", "score", "--resume"])
        (tmp_path / "dataset/response.jsonl").write_text("labels are unavailable before evaluation")
    for first, second in zip(score_arrays(outputs[0]), score_arrays(outputs[1])):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    for file in ("geometry.npz", "bank.npz"):
        with np.load(outputs[0] / "reference/0" / file) as first:
            with np.load(outputs[1] / "reference/0" / file) as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])
