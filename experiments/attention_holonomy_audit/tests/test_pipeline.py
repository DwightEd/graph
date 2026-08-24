from pathlib import Path

import numpy as np
import torch

from experiments.attention_holonomy_audit.config import (
    AuditConfig,
    EvaluationConfig,
    GraphConfig,
    ReferenceConfig,
    TransportConfig,
)
from experiments.attention_holonomy_audit.evaluation import evaluate_scores
from experiments.attention_holonomy_audit.experiment import fit_reference, score_split
from experiments.attention_holonomy_audit.tests.helpers import FakeDataset, make_sample


def test_fit_score_evaluate_pipeline(tmp_path: Path):
    train_samples = [
        make_sample(f"train-{index}", f"source-{index}", 0.01 * index)
        for index in range(8)
    ]
    test_samples = [
        make_sample("test-0", "test-source-0", 0.0),
        make_sample("test-1", "test-source-1", 0.15),
    ]
    train = FakeDataset(tmp_path / "train", "train", train_samples)
    labels = {
        "test-0": torch.tensor([0, 0, 0, 0], dtype=torch.long),
        "test-1": torch.tensor([0, 1, 1, 0], dtype=torch.long),
    }
    test = FakeDataset(tmp_path / "test", "test", test_samples, labels)
    config = AuditConfig(
        graph=GraphConfig(max_relay_predecessors=8, max_query_events=16),
        transport=TransportConfig(ridge_alpha=1e-3, minimum_pairs=1),
        reference=ReferenceConfig(
            calibration_fraction=0.25,
            reservoir_rows=1000,
            position_degree=2,
            seed=11,
        ),
        evaluation=EvaluationConfig(bootstrap_replicates=10, seed=11),
    )
    reference = tmp_path / "reference.npz"
    scores = tmp_path / "scores.npz"
    fit = fit_reference(train, reference, config=config, task_type="QA")
    assert fit["labels_read"] is False
    scored = score_split(test, reference, scores, task_type="QA")
    assert scored["labels_read"] is False
    report = evaluate_scores(
        test,
        scores,
        tmp_path / "evaluation",
        bootstrap_replicates=10,
        seed=11,
    )
    assert report["labels_read"] is True
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()
    with np.load(scores, allow_pickle=False) as arrays:
        assert not bool(arrays["labels_included"].item())
        assert arrays["standardized_primary"].shape[1] == 6
