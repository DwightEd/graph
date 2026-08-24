from types import SimpleNamespace

import numpy as np
import torch

from experiments.causal_walk_audit import evaluation
from experiments.causal_walk_audit.artifacts import read_json, save_npz, write_json
from experiments.causal_walk_audit.config import WalkAuditConfig


class _LabelStore:
    def response_labels(self, sample):
        return sample.labels


class _Attention:
    def __init__(self, token_ids):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.response_idx = 0


class _Dataset:
    def __init__(self, samples):
        self.samples = {item.sample_id: item for item in samples}

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return _LabelStore()


def test_end_to_end_label_late_evaluation(tmp_path, monkeypatch):
    score_dir = tmp_path / "score"
    sample_dir = score_dir / "samples"
    sample_dir.mkdir(parents=True)
    names = tuple(evaluation.SCORE_DIRECTION)
    rows = []
    samples = []
    for index in range(3):
        sample_id = f"sample-{index}"
        labels = (
            np.array([0, 0, 1, 1, 0], dtype=np.int8)
            if index == 0
            else np.zeros(5, dtype=np.int8)
        )
        token_ids = np.arange(100, 105, dtype=np.int32)
        sample = SimpleNamespace(
            sample_id=sample_id,
            source_id=f"source-{index}",
            task_type="QA",
            labels=torch.tensor(labels),
            attention=lambda token_ids=token_ids: _Attention(token_ids),
            release_attention=lambda: None,
        )
        samples.append(sample)
        matrix = np.zeros((5, len(names)), dtype=np.float32)
        for column in range(len(names)):
            matrix[:, column] = np.linspace(0.1, 0.5, 5)
        matrix[:, names.index("lock_in")] += labels * 0.5
        matrix[:, names.index("anchor_js_peak")] += labels * 0.4
        matrix[:, names.index("anchor_js_excess")] += labels * 0.2
        matrix[:, names.index("evidence_escape")] -= labels * 0.3
        filename = f"{sample_id}.npz"
        save_npz(
            sample_dir / filename,
            schema=np.asarray("causal-walk-audit-score-v1"),
            labels_included=np.asarray(False),
            sample_id=np.asarray(sample_id),
            source_id=np.asarray(sample.source_id),
            response_token_ids=token_ids,
            score_names=np.asarray(names, dtype=str),
            scores=matrix,
            token_index=np.arange(5, dtype=np.int32),
            valid_rows=np.ones(5, dtype=np.int16),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": sample.source_id,
                "task_type": "QA",
                "score_path": f"samples/{filename}",
                "tokens": 5,
                "anchor_mode": "manifest",
            }
        )

    config = WalkAuditConfig(
        bootstrap_replicates=10,
        permutation_replicates=10,
        show_progress=False,
    )
    write_json(
        score_dir / "manifest.json",
        {
            "schema": "causal-walk-audit-manifest-v1",
            "labels_read": False,
            "config": config.to_dict(),
            "validation": {
                "order2_gain": 0.1,
                "order2_path_gain": 0.1,
                "order3_gain": 0.05,
                "order3_path_gain": 0.05,
            },
            "samples": rows,
        },
    )
    monkeypatch.setattr(evaluation, "_open_dataset", lambda root: _Dataset(samples))
    output = tmp_path / "evaluation"
    evaluation.evaluate_walk_audit(
        split_root=tmp_path,
        score_dir=score_dir,
        output_dir=output,
        bootstrap_replicates=10,
        permutation_replicates=10,
    )
    report = read_json(output / "evaluation.json")
    assert report["labels_read"] is True
    assert report["artifact_validation_passed"] is True
    assert report["positive_tokens"] == 2
    assert (output / "decision_table.csv").exists()
