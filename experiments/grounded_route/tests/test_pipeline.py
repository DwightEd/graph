from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.grounded_route.config import TrainConfig
from experiments.grounded_route import pipeline
from experiments.grounded_route.artifacts import GraphSpec, load_scores
from experiments.grounded_route.tests.helpers import make_graph


PACKAGE = Path(__file__).parents[1]


def test_label_firewall_is_visible_in_the_pipeline_source():
    pipeline = (PACKAGE / "pipeline.py").read_text(encoding="utf-8")
    detector = (PACKAGE / "detection.py").read_text(encoding="utf-8")
    evaluator = (PACKAGE / "evaluate.py").read_text(encoding="utf-8")
    runner = (PACKAGE / "run.py").read_text(encoding="utf-8")

    assert "prepare_evaluation_labels" not in pipeline
    assert "retain_embedded_labels" not in pipeline
    assert "prepare_evaluation_labels" not in detector
    assert "response_labels" not in detector
    assert "retain_embedded_labels" not in detector
    assert "FrozenEvaluation" in evaluator
    assert "validate_source_audit" in evaluator
    assert "retain_embedded_labels=True" in runner


class Sample:
    def __init__(self, dataset, sample_id, source_id):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = source_id

    def release_attention(self):
        pass


class Dataset:
    def __init__(self, groups: int = 12):
        self.sample_ids = tuple(f"sample-{group}" for group in range(groups))
        self.samples = {
            sample_id: Sample(self, sample_id, f"source-{group}")
            for group, sample_id in enumerate(self.sample_ids)
        }

    def __getitem__(self, sample_id):
        return self.samples[sample_id]


def test_encoder_validation_and_detector_sources_are_disjoint():
    dataset = Dataset()
    split = pipeline.source_split(
        dataset,
        dataset.sample_ids,
        TrainConfig(validation_fraction=0.2, detector_fraction=0.2, seed=31),
    )

    fit = set(split["fit_source_ids"])
    validation = set(split["validation_source_ids"])
    detector = set(split["calibration_source_ids"])
    assert fit.isdisjoint(validation)
    assert fit.isdisjoint(detector)
    assert validation.isdisjoint(detector)
    assert fit | validation | detector == {
        f"source-{group}" for group in range(12)
    }


def test_checkpoint_geometry_must_match_the_graph_spec():
    spec = GraphSpec(
        dataset_root="/data/test",
        dataset_manifest_sha256="a" * 64,
        split="test",
        task="QA",
        sample_ids=("sample",),
        layer_count=3,
        head_count=4,
        graph_config={"block_rows": 128, "numerical_tolerance": 4e-3},
    )
    pipeline.validate_checkpoint_geometry(
        spec,
        {"layer_count": 3, "head_count": 4},
    )

    for checkpoint in (
        {"layer_count": 2, "head_count": 4},
        {"layer_count": 3, "head_count": 5},
    ):
        with pytest.raises(ValueError, match="layer/head geometry"):
            pipeline.validate_checkpoint_geometry(spec, checkpoint)


class SentinelSample:
    def __init__(self, dataset, sample_id: str, source_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.graph = replace(
            make_graph(),
            sample_id=sample_id,
            source_id=source_id,
        )

    def release_attention(self):
        pass


class LabelSentinelDataset:
    def __init__(self, root: Path, split: str, groups: int):
        self.root = root
        self.device = "cpu"
        self.manifest = {
            "split": split,
            "num_layers": 3,
            "num_heads": 4,
        }
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(self.manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.sample_ids = tuple(f"{split}-sample-{group}" for group in range(groups))
        self.samples = {
            sample_id: SentinelSample(
                self,
                sample_id,
                f"{split}-source-{group}",
            )
            for group, sample_id in enumerate(self.sample_ids)
        }
        self.label_accesses = 0

    def __getitem__(self, sample_id):
        return self.samples[sample_id]

    def prepare_evaluation_labels(self):
        self.label_accesses += 1
        raise AssertionError("label API crossed the representation firewall")


def test_build_fit_encode_detect_handoff_never_opens_labels(tmp_path, monkeypatch):
    train = LabelSentinelDataset(tmp_path / "train", "train", groups=6)
    test = LabelSentinelDataset(tmp_path / "test", "test", groups=2)
    datasets = {
        train.root.resolve(): train,
        test.root.resolve(): test,
    }

    def open_dataset(root, *, device="cpu"):
        dataset = datasets[Path(root).resolve()]
        dataset.device = device
        return dataset

    monkeypatch.setattr(pipeline, "open_research_dataset", open_dataset)
    monkeypatch.setattr(pipeline, "build_graph", lambda sample, config: sample.graph)

    train_spec = tmp_path / "train_graph.json"
    checkpoint = tmp_path / "model.pt"
    calibration = tmp_path / "calibration"
    test_spec = tmp_path / "test_graph.json"
    encoded_test = tmp_path / "encoded_test"
    reference = tmp_path / "detector.npz"
    scores = tmp_path / "scores.npz"

    pipeline.build(train.root, train_spec)
    pipeline.fit(
        train_spec,
        checkpoint,
        train_config=TrainConfig(
            epochs=1,
            validation_fraction=0.2,
            detector_fraction=0.2,
            seed=41,
        ),
    )
    pipeline.encode(
        train_spec,
        checkpoint,
        calibration,
        scope="calibration",
    )
    pipeline.build(test.root, test_spec)
    pipeline.encode(test_spec, checkpoint, encoded_test, scope="all")
    report = pipeline.detect(
        calibration / "index.npz",
        encoded_test / "index.npz",
        reference,
        scores,
    )

    frozen = load_scores(scores)
    assert report["labels_read"] is False
    assert report["variant"] == "real"
    assert not bool(frozen["labels_included"].item())
    assert frozen["variant"].item() == "real"
    assert float(frozen["changed_fraction"].item()) == 0.0
    assert "embedding" not in frozen
    assert train.label_accesses == 0
    assert test.label_accesses == 0
