import json
import sys

import numpy as np

import experiments.source_reuse_contrast.main as cli
from experiments.source_reuse_contrast import evaluation, experiment
from experiments.source_reuse_contrast.artifacts import load_npz
from experiments.source_reuse_contrast.predictability import write_predictability_gate

from .helpers import SyntheticDataset, sequence_sample, tiny_config


def _dataset(
    prefix: str,
    labels: bool,
    task_types: tuple[str, ...] = ("QA", "QA", "QA", "QA"),
) -> SyntheticDataset:
    samples = []
    for index, task_type in enumerate(task_types):
        samples.append(
            sequence_sample(
                sample_id=f"{prefix}-{index}",
                source_id=f"group-{index}",
                task_type=task_type,
                labels=[0, 1, 1, 0, 0] if labels else None,
            )
        )
    return SyntheticDataset(samples)


def test_cli_passes_task_type_to_train_and_score(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "train_model", lambda **arguments: calls.append(arguments))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source-reuse",
            "train",
            "--train-split",
            "train",
            "--output-dir",
            "output",
            "--task-type",
            "QA",
        ],
    )
    cli.main()
    assert calls[-1]["task_type"] == "QA"

    monkeypatch.setattr(cli, "score_split", lambda **arguments: calls.append(arguments))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source-reuse",
            "score",
            "--split-root",
            "test",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "output",
            "--task-type",
            "QA",
        ],
    )
    cli.main()
    assert calls[-1]["task_type"] == "QA"


def test_train_score_evaluate_uses_validation_and_keeps_labels_out(
    tmp_path,
    monkeypatch,
):
    train_dataset = _dataset("train", labels=False)
    test_dataset = _dataset("test", labels=True)
    monkeypatch.setattr(
        experiment,
        "_open_dataset",
        lambda split_root, device: train_dataset
        if str(split_root) == "train"
        else test_dataset,
    )
    monkeypatch.setattr(evaluation, "_open_dataset", lambda split_root: test_dataset)

    checkpoint = experiment.train_model(
        train_split="train",
        output_dir=tmp_path / "train",
        device="cpu",
        config=tiny_config(epochs=1, score_rounds=2),
    )
    score_path = experiment.score_split(
        split_root="test",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "score",
        device="cpu",
    )

    arrays = load_npz(score_path)
    assert not bool(arrays["labels_included"].item())
    assert "endpoint_nll" in arrays
    assert "margin" in arrays
    assert "positive_logit" in arrays
    assert "query_embedding" in arrays
    assert np.isfinite(arrays["endpoint_nll"]).all()

    training_path = tmp_path / "train" / "training.json"
    manifest_path = tmp_path / "score" / "manifest.json"
    training = json.loads(training_path.read_text())
    assert training["labels_read"] is False
    assert training["validation_samples"] > 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["diagnostics"]["unique_endpoint_nll_1e6"] > 1

    gate_path = tmp_path / "predictability_gate.json"
    write_predictability_gate(
        training_paths={"dynamic": training_path},
        manifest_paths={"dynamic": manifest_path},
        output_path=gate_path,
    )
    gate = json.loads(gate_path.read_text())
    assert gate["labels_read"] is False

    evaluation.evaluate_scores(
        split_root="test",
        score_paths={"dynamic": score_path},
        output_dir=tmp_path / "evaluation",
        bootstrap_replicates=10,
        onset_window=1,
        seed=3,
    )
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["labels_read"] is True
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()
    assert (tmp_path / "evaluation" / "coverage.csv").is_file()


def test_train_and_score_filter_task_type_and_record_selection(tmp_path, monkeypatch):
    task_types = ("Summary", "QA", "Data2txt", "QA", "QA")
    train_dataset = _dataset("train", labels=False, task_types=task_types)
    test_dataset = _dataset("test", labels=True, task_types=task_types)
    monkeypatch.setattr(
        experiment,
        "_open_dataset",
        lambda split_root, device: train_dataset
        if str(split_root) == "train"
        else test_dataset,
    )

    checkpoint = experiment.train_model(
        train_split="train",
        output_dir=tmp_path / "train",
        device="cpu",
        config=tiny_config(epochs=1, score_rounds=1),
        task_type="QA",
    )
    score_path = experiment.score_split(
        split_root="test",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "score",
        device="cpu",
        task_type="QA",
    )

    training = json.loads((tmp_path / "train" / "training.json").read_text())
    manifest = json.loads((tmp_path / "score" / "manifest.json").read_text())
    arrays = load_npz(score_path)
    assert training["task_type_filter"] == "QA"
    assert training["fit_samples"] + training["validation_samples"] == 3
    assert manifest["task_type_filter"] == "QA"
    assert manifest["samples"] == 3
    assert set(arrays["task_type"].tolist()) == {"QA"}
    assert set(arrays["sample_id"].tolist()) == {"test-1", "test-3", "test-4"}


def test_source_disjoint_split_has_no_group_overlap():
    dataset = _dataset("split", labels=False)
    fit, validation = experiment.source_disjoint_split(
        dataset,
        dataset.sample_ids,
        validation_fraction=0.25,
        seed=7,
    )
    fit_groups = {dataset[sample_id].source_id for sample_id in fit}
    validation_groups = {dataset[sample_id].source_id for sample_id in validation}
    assert fit_groups.isdisjoint(validation_groups)
