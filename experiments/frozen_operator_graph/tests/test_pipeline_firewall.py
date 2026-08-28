from dataclasses import replace
import json
from pathlib import Path

from .. import pipeline
from .helpers import FakeAttentionCache, synthetic_bundle


class FakeSample:
    def __init__(self, bundle):
        self.sample_id = "17"
        self.source_id = "source-17"
        self.task_type = "QA"
        self.data_source = "synthetic"
        self.metadata = {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "task_type": self.task_type,
            "data_source": self.data_source,
        }
        self._attention = FakeAttentionCache(bundle.capture)
        self.released = False

    def attention(self):
        return self._attention

    def release_attention(self):
        self.released = True


class FakeDataset:
    def __init__(self, sample):
        self.sample_ids = [sample.sample_id]
        self.sample = sample
        self.labels_requested = False

    def __getitem__(self, sample_id):
        assert str(sample_id) == self.sample.sample_id
        return self.sample

    def labels(self):
        self.labels_requested = True
        raise AssertionError("construction must never open labels")


class FakeReplay:
    def __init__(self, bundle):
        self.checkpoint = bundle.capture.checkpoint
        self.model = object()
        self.bundle = bundle

    def capture(self, _tokens, _response_start, *, attention_validator, **_kwargs):
        binding = attention_validator(
            [layer.attention for layer in self.bundle.capture.layers]
        )
        return replace(self.bundle.capture, attention_cache_binding=binding)


def test_pipeline_never_opens_labels(monkeypatch, tmp_path: Path):
    bundle = synthetic_bundle(seed=31)
    sample = FakeSample(bundle)
    dataset = FakeDataset(sample)
    replay = FakeReplay(bundle)
    split = tmp_path / "split"
    split.mkdir()
    source_json = tmp_path / "response.jsonl"
    source_json.write_text("{}\n", encoding="utf-8")
    (split / "manifest.json").write_text(
        json.dumps({"schema": "synthetic"}) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(pipeline, "_open_dataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(
        pipeline.ExactLlamaReplay,
        "from_pretrained",
        classmethod(lambda _cls, *_a, **_k: replay),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_operator_basis",
        lambda *_a, **_k: bundle.basis,
    )

    report = pipeline.construct_split(
        split_root=split,
        source_json=source_json,
        model_path="synthetic-checkpoint",
        output_root=tmp_path / "output",
        device="cpu",
        model_dtype="float32",
    )
    assert report.manifest["labels_read_during_construction"] is False
    assert report.manifest["source_dataset"]["sha256"]
    assert report.manifest["source_dataset"]["content_read_for_graph_features"] is False
    assert dataset.labels_requested is False
    assert sample.released
    assert len(report.rows) == 1
