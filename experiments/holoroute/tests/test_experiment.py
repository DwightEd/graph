from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.holoroute.config import (
    DensityConfig,
    HoloRouteConfig,
    MaskConfig,
    ModelConfig,
    TrainConfig,
)
from experiments.holoroute.experiment import score_split, train_reference
from experiments.holoroute.tests.helpers import synthetic_graph


class FakeSample:
    def __init__(self, sample_id, source_id):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"

    def release_attention(self):
        pass


class FakeDataset:
    def __init__(self, root: Path, split: str, count: int):
        root.mkdir(parents=True)
        self.root = root
        self.device = "cpu"
        self.manifest = {"split": split, "num_layers": 3, "num_heads": 4}
        (root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.samples = {
            f"{split}-{index}": FakeSample(f"{split}-{index}", f"source-{index}")
            for index in range(count)
        }
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]


def test_train_and_score_artifacts_are_label_free(tmp_path, monkeypatch):
    def build(sample):
        return replace(
            synthetic_graph(),
            sample_id=sample.sample_id,
            source_id=sample.source_id,
        )

    monkeypatch.setattr("experiments.holoroute.experiment.build_attention_event_graph", build)
    train = FakeDataset(tmp_path / "train", "train", 6)
    test = FakeDataset(tmp_path / "test", "test", 2)
    config = HoloRouteConfig(
        model=ModelConfig(hidden_dim=16, head_encoder_heads=4, transport_rank=2, message_blocks=1),
        masking=MaskConfig(event_fraction=0.4, relay_fraction=0.25, score_rounds=1),
        train=TrainConfig(epochs=1, validation_fraction=0.2, calibration_fraction=0.2),
        density=DensityConfig(reservoir_rows=128),
    )
    checkpoint = tmp_path / "model.pt"
    density = tmp_path / "density.npz"
    score = tmp_path / "scores.npz"
    report = train_reference(train, checkpoint, density, config=config)
    assert report["labels_read"] is False
    score_report = score_split(test, checkpoint, density, score)
    assert score_report["labels_read"] is False
    with np.load(score, allow_pickle=False) as arrays:
        assert not bool(arrays["labels_included"])
        assert arrays["mechanism_feature"].shape[1] == 6
