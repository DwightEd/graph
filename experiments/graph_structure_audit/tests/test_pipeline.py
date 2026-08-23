import gc
import json
import weakref

import numpy as np
import torch

from experiments.graph_structure_audit import evaluate, experiment
from experiments.graph_structure_audit.artifacts import load_npz
from experiments.graph_structure_audit.model import LayeredGraphRecovery
from .helpers import dataset, tiny_config


def test_train_score_evaluate_keeps_labels_out(tmp_path, monkeypatch):
    train = dataset("train", 4)
    test = dataset("test", 3, labels=True)
    grad_modes = []
    original_forward = LayeredGraphRecovery.forward

    def record_grad_mode(self, *args, **kwargs):
        grad_modes.append((self.training, torch.is_grad_enabled()))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(LayeredGraphRecovery, "forward", record_grad_mode)

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
    assert (True, True) in grad_modes
    assert all(training == grad_enabled for training, grad_enabled in grad_modes)
    assert all(sample.release_calls >= 1 for sample in train.items.values())
    assert all(sample._attention is None for sample in train.items.values())

    load_graph = experiment.load_multiplex_graph
    previous_graph = None

    def assert_previous_graph_released(*args, **kwargs):
        nonlocal previous_graph
        gc.collect()
        assert previous_graph is None or previous_graph() is None
        graph = load_graph(*args, **kwargs)
        previous_graph = weakref.ref(graph)
        return graph

    monkeypatch.setattr(experiment, "load_multiplex_graph", assert_previous_graph_released)
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
    assert all(sample.release_calls == 1 for sample in test.items.values())
    assert all(sample._attention is None for sample in test.items.values())

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
