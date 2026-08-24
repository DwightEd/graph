import json
from types import SimpleNamespace

import torch

from experiments.causal_walk_audit import experiment
from experiments.causal_walk_audit.config import AuditConfig, CalibrationConfig
from experiments.causal_walk_audit.evaluation import evaluate_scores


class Sample:
    def __init__(self, sample_id, source_id, scale):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.scale = scale
        self._attention = None

    def attention(self):
        if self._attention is None:
            self._attention = SimpleNamespace(
                response_idx=2,
                num_response_tokens=5,
                num_layers=2,
                num_heads=2,
                num_tokens=7,
                response_values=torch.ones(1),
                attention_diagonal=torch.full((2, 2, 7), 0.2),
                attention_floor=0.01,
            )
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        source, target, layer, head, weight = [], [], [], [], []
        for current_layer in range(2):
            for current_head in range(2):
                for query in range(5):
                    current_target = 2 + query
                    source.append(query % 2)
                    target.append(current_target)
                    layer.append(current_layer)
                    head.append(current_head)
                    weight.append(0.35 * self.scale)
                    if query:
                        source.append(current_target - 1)
                        target.append(current_target)
                        layer.append(current_layer)
                        head.append(current_head)
                        weight.append(0.25 * self.scale)
        yield SimpleNamespace(
            source=torch.tensor(source),
            target=torch.tensor(target),
            layer=torch.tensor(layer),
            head=torch.tensor(head),
            weight=torch.tensor(weight),
        )

    def release_attention(self):
        self._attention = None


class Labels:
    def response_labels(self, sample):
        value = torch.zeros(5, dtype=torch.long)
        if sample.sample_id.endswith("-0"):
            value[2] = 1
        return value


class Dataset:
    def __init__(self, root, split, count):
        self.root = root
        self.device = "cpu"
        self.manifest = {"split": split, "num_layers": 2, "num_heads": 2}
        (root / "manifest.json").write_text(json.dumps(self.manifest))
        self.sample_ids = [f"{split}-{index}" for index in range(count)]
        self.samples = {
            sample_id: Sample(sample_id, f"group-{index}", 1.0 + 0.01 * index)
            for index, sample_id in enumerate(self.sample_ids)
        }

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return Labels()


def test_fit_score_evaluate_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment,
        "canonical_source_group",
        lambda sample: sample.source_id,
    )
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    train_root.mkdir()
    test_root.mkdir()
    train = Dataset(train_root, "train", 15)
    test = Dataset(test_root, "test", 4)
    config = AuditConfig(
        calibration=CalibrationConfig(
            channel_fraction=0.2,
            fusion_fraction=0.2,
            reservoir_rows=200,
            topology_min_changed_fraction=0.0,
            seed=7,
        )
    )

    reference = tmp_path / "reference.npz"
    scores = tmp_path / "scores.npz"
    fit = experiment.fit_reference(train, reference, config=config)
    scored = experiment.score_split(test, reference, scores)
    report = evaluate_scores(
        test,
        scores,
        tmp_path / "evaluation",
        bootstrap_replicates=10,
    )

    assert fit["labels_read"] is False
    assert scored["labels_read"] is False
    assert report["labels_read"] is True
    assert report["primary_detector"] == "score"
