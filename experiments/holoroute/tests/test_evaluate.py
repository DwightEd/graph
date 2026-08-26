from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.holoroute.artifacts import SCORE_SCHEMA, save_npz, sha256
from experiments.holoroute.evaluate import evaluate


MODEL_TYPE = "routing_fingerprint_linear_probe"


class Attention:
    token_ids = torch.tensor([1, 2, 10, 11])
    response_idx = 2


class Sample:
    source_id = "source"
    task_type = "QA"

    def attention(self):
        return Attention()

    def release_attention(self):
        pass


class LabelStore:
    def response_labels(self, sample):
        return torch.tensor([0, 1])


class Dataset:
    def __init__(self):
        self.labels_opened = False

    def __getitem__(self, sample_id):
        if sample_id != "sample":
            raise KeyError(sample_id)
        return Sample()

    def prepare_evaluation_labels(self):
        self.labels_opened = True
        return LabelStore()


def score_artifact(
    tmp_path: Path,
    *,
    source_id="source",
    response_token_id=(10, 11),
) -> Path:
    checkpoint = tmp_path / "method.pt"
    reference = tmp_path / "reference.npz"
    probe = tmp_path / "probe.npz"
    checkpoint.write_bytes(b"method")
    reference.write_bytes(b"reference")
    probe.write_bytes(b"probe")
    path = tmp_path / "scores.npz"
    save_npz(
        path,
        schema=np.asarray(SCORE_SCHEMA),
        model_type=np.asarray(MODEL_TYPE),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(checkpoint)),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        reference_path=np.asarray(str(reference)),
        reference_sha256=np.asarray(sha256(reference)),
        probe_path=np.asarray(str(probe)),
        probe_sha256=np.asarray(sha256(probe)),
        sample_id=np.asarray(["sample", "sample"]),
        source_id=np.asarray([source_id, source_id]),
        task_type=np.asarray(["QA", "QA"]),
        token_index=np.asarray([0, 1], dtype=np.int32),
        response_length=np.asarray([2, 2], dtype=np.int32),
        response_token_id=np.asarray(response_token_id, dtype=np.int64),
        score=np.asarray([0.1, 0.9], dtype=np.float32),
        residual=np.asarray([[0.1], [0.9]], dtype=np.float32),
        standardized=np.asarray([[0.1], [0.9]], dtype=np.float32),
        coverage=np.ones((2, 1), dtype=np.float32),
        condition=np.zeros((2, 1), dtype=np.float32),
        residual_names=np.asarray(["linear_probe_logit"]),
        condition_names=np.asarray(["unused"]),
    )
    return path


def test_evaluate_accepts_existing_supervised_scores(tmp_path):
    data = Dataset()
    score_path = score_artifact(tmp_path)

    report = evaluate(data, score_path, tmp_path / "evaluation", bootstrap_replicates=2)

    assert data.labels_opened
    assert report["labels_used_during"] == "train_probe_fit_and_posthoc_test_evaluation"
    assert report["same_token"]["auroc"] == 1.0


def test_row_mismatch_is_rejected_before_labels(tmp_path):
    data = Dataset()
    score_path = score_artifact(tmp_path, source_id="wrong")

    with pytest.raises(ValueError, match="source IDs"):
        evaluate(data, score_path, tmp_path / "evaluation", bootstrap_replicates=2)

    assert not data.labels_opened


def test_token_mismatch_is_rejected_before_labels(tmp_path):
    data = Dataset()
    score_path = score_artifact(tmp_path, response_token_id=(10, 12))

    with pytest.raises(ValueError, match="response tokens"):
        evaluate(data, score_path, tmp_path / "evaluation", bootstrap_replicates=2)

    assert not data.labels_opened
