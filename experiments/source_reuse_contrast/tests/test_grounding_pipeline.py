import json

import numpy as np

from experiments.source_reuse_contrast import grounding_evaluation, grounding_experiment
from experiments.source_reuse_contrast.artifacts import load_npz

from .grounding_helpers import SyntheticDataset, sequence_sample, tiny_config


def test_grounding_train_score_evaluate_keeps_labels_out(tmp_path, monkeypatch):
    train_dataset = SyntheticDataset(
        [sequence_sample(sample_id=f"train-{index}") for index in range(4)]
    )
    test_dataset = SyntheticDataset(
        [
            sequence_sample(sample_id="test-a", labels=[0, 1, 1, 0]),
            sequence_sample(sample_id="test-b", labels=[0, 0, 1, 0]),
        ]
    )
    monkeypatch.setattr(
        grounding_experiment,
        "_open_dataset",
        lambda split_root, device: train_dataset
        if str(split_root) == "train"
        else test_dataset,
    )
    monkeypatch.setattr(
        grounding_evaluation,
        "_open_dataset",
        lambda split_root: test_dataset,
    )

    checkpoint = grounding_experiment.train_grounding_model(
        train_split="train",
        output_dir=tmp_path / "train",
        device="cpu",
        config=tiny_config(epochs=1, score_rounds=1),
    )
    score_path = grounding_experiment.score_grounding_split(
        split_root="test",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "score",
        device="cpu",
    )
    arrays = load_npz(score_path)
    assert not bool(arrays["labels_included"].item())
    assert "closure" in arrays
    assert "embedding" in arrays
    assert len(arrays["closure"]) == 8
    assert np.isfinite(arrays["closure"]).all()

    grounding_evaluation.evaluate_grounding_scores(
        split_root="test",
        score_path=score_path,
        output_dir=tmp_path / "evaluation",
        bootstrap_replicates=10,
        onset_window=1,
        seed=5,
    )
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["labels_read"] is True
    assert report["tokens"] == 8
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()
