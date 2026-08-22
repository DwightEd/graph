import json

import numpy as np

from experiments.graph_structure_audit import evaluate, experiment
from experiments.graph_structure_audit.artifacts import load_npz
from .helpers import dataset, tiny_config


def test_train_score_evaluate_keeps_labels_out(tmp_path, monkeypatch):
    train = dataset("train", 4)
    test = dataset("test", 3, labels=True)

    monkeypatch.setattr(
        experiment,
        "_open_dataset",
        lambda split_root, device: train if str(split_root) == "train" else test,
    )
    monkeypatch.setattr(evaluate, "_open_dataset", lambda split_root: test)

    checkpoint = experiment.train_recovery_model(
        train_split="train",
        output_dir=tmp_path / "train",
        device="cpu",
        config=tiny_config(),
    )
    scores = experiment.score_recovery_split(
        split_root="test",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "score",
        device="cpu",
        save_graphs=True,
    )
    arrays = load_npz(scores)
    assert not bool(arrays["labels_included"].item())
    assert arrays["embedding"].shape[0] == 12
    assert np.isfinite(arrays["recovery"]).all()
    assert list((tmp_path / "score" / "graphs").glob("*.npz"))

    evaluate.evaluate_recovery_scores(
        split_root="test",
        score_path=scores,
        output_dir=tmp_path / "evaluation",
        bootstrap_replicates=10,
        seed=3,
    )
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["labels_read"] is True
    assert (tmp_path / "evaluation" / "structure_gates.csv").is_file()
