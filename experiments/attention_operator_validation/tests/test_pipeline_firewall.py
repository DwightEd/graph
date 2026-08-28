from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiment_protocol import file_sha256
from experiments.attention_operator_validation.artifacts import save_feature_table
from experiments.attention_operator_validation import pipeline

from .helpers import make_table


class _Sample:
    def __init__(self, sample_id, source_id, response_length, events, phase):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.response_length = response_length
        self.events = events
        self.phase = phase

    def attention(self):
        self.events.append(f"{self.phase}:{self.sample_id}")
        return SimpleNamespace(num_response_tokens=self.response_length)

    def release_attention(self):
        pass


class _Labels:
    def response_labels(self, sample):
        sample.attention()
        value = torch.zeros(sample.response_length, dtype=torch.long)
        if sample.sample_id == "sample-b":
            value[-1] = 1
        return value


class _Dataset:
    def __init__(self, root, retain_labels, events, *, bad_source=False):
        self.root = root
        self.manifest = {"split": "test"}
        self.sample_ids = ["sample-a", "sample-b"]
        phase = "label" if retain_labels else "bind"
        self.samples = {
            "sample-a": _Sample(
                "sample-a",
                "wrong-source" if bad_source else "source-a",
                2,
                events,
                phase,
            ),
            "sample-b": _Sample("sample-b", "source-b", 3, events, phase),
        }
        self.retain_labels = retain_labels
        self.events = events

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self, sample_ids):
        if not self.retain_labels:
            raise AssertionError("label API opened on the binding dataset")
        self.events.append("labels-opened")
        return _Labels()


def _artifact(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"split":"test"}\n', encoding="utf-8")
    table = make_table(manifest_sha256=file_sha256(manifest))
    table = replace(
        table,
        metadata={**table.metadata, "data_root": str(tmp_path.resolve())},
    )
    path = tmp_path / "features.npz"
    save_feature_table(path, table)
    return path


def test_evaluation_opens_labels_only_after_label_free_binding(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    events = []

    def open_dataset(root, *, retain_embedded_labels=False, **kwargs):
        events.append(f"dataset-retain={retain_embedded_labels}")
        return _Dataset(root, retain_embedded_labels, events)

    monkeypatch.setattr(pipeline, "open_research_dataset", open_dataset)
    pipeline.evaluate_features(
        tmp_path,
        artifact,
        tmp_path / "report.json",
        bootstrap_replicates=0,
        cv_folds=2,
    )

    assert events.index("bind:sample-b") < events.index("dataset-retain=True")
    assert events.index("dataset-retain=True") < events.index("labels-opened")


def test_binding_failure_never_opens_labeled_dataset(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    events = []

    def open_dataset(root, *, retain_embedded_labels=False, **kwargs):
        events.append(f"dataset-retain={retain_embedded_labels}")
        return _Dataset(
            root,
            retain_embedded_labels,
            events,
            bad_source=not retain_embedded_labels,
        )

    monkeypatch.setattr(pipeline, "open_research_dataset", open_dataset)
    with pytest.raises(ValueError, match="source IDs"):
        pipeline.evaluate_features(
            tmp_path,
            artifact,
            tmp_path / "report.json",
            bootstrap_replicates=0,
            cv_folds=2,
        )

    assert events == ["dataset-retain=False", "bind:sample-a"]
