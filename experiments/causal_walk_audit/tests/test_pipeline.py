from types import SimpleNamespace

import numpy as np

from experiments.causal_walk_audit import experiment
from experiments.causal_walk_audit.artifacts import load_npz, read_json
from experiments.causal_walk_audit.config import WalkAuditConfig

from .helpers import routing_state


class _Dataset:
    def __init__(self, prefix, count):
        self.sample_ids = [f"{prefix}-{index}" for index in range(count)]
        self.items = {
            sample_id: SimpleNamespace(
                sample_id=sample_id,
                source_id=f"source-{index}",
                task_type="QA",
            )
            for index, sample_id in enumerate(self.sample_ids)
        }

    def __getitem__(self, sample_id):
        return self.items[str(sample_id)]


def test_fit_and_score_pipeline(tmp_path, monkeypatch):
    train = _Dataset("train", 6)
    test = _Dataset("test", 2)
    monkeypatch.setattr(
        experiment,
        "_open_dataset",
        lambda root, device: train if str(root).endswith("train") else test,
    )
    monkeypatch.setattr(
        experiment,
        "_routing_for_sample",
        lambda sample, config: (
            routing_state(),
            np.array([10, 11, 12], dtype=np.int32),
        ),
    )
    config = WalkAuditConfig(
        max_anchors=2,
        prompt_chunk_tokens=1,
        train_reservoir_rows=500,
        anchor_shuffle_replicates=2,
        bootstrap_replicates=5,
        permutation_replicates=5,
        show_progress=False,
    )
    train_output = tmp_path / "train_output"
    model = experiment.fit_walk_audit(
        train_split=tmp_path / "train",
        output_dir=train_output,
        config=config,
        task_type="QA",
    )
    assert model.exists()
    training = read_json(train_output / "training.json")
    assert training["labels_read"] is False
    assert training["validation_samples"] == 1

    score_output = tmp_path / "score_output"
    experiment.score_walk_audit(
        split_root=tmp_path / "test",
        model_path=model,
        output_dir=score_output,
        task_type="QA",
    )
    manifest = read_json(score_output / "manifest.json")
    assert manifest["labels_read"] is False
    assert len(manifest["samples"]) == 2
    arrays = load_npz(score_output / manifest["samples"][0]["score_path"])
    assert arrays["scores"].shape[0] == 3
    assert arrays["anchor_js_map"].shape[:2] == (3, 3)
