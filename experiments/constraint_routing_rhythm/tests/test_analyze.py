from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.constraint_routing_rhythm import analyze
from experiments.constraint_routing_rhythm.artifacts import load_result
from experiments.constraint_routing_rhythm.artifacts import (
    save_result as save_result_file,
)

RUN_CONFIG = {"model": "/models/fixture", "model_id": "fixture-model"}


class FakeAttention:
    def __init__(self, marker: int):
        self.token_ids = torch.tensor([10, marker, 12, 20, 21, 22])
        self.response_idx = 3


class FakeSample:
    def __init__(self, sample_id: str, source_id: str, task: str, marker: int, events):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = task
        self.generator_model = "fixture-generator"
        self.observer_model = "fixture-model"
        self.cached = FakeAttention(marker)
        self.events = events
        self.attention_calls = 0
        self.release_calls = 0

    @property
    def labels(self):
        raise AssertionError("analysis must not read labels")

    def attention(self):
        self.attention_calls += 1
        self.events.append(f"attention:{self.sample_id}")
        return self.cached

    def release_attention(self):
        self.release_calls += 1
        self.events.append(f"release:{self.sample_id}")


class FakeDataset:
    def __init__(self, samples):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self, _sample_ids=None):
        raise AssertionError("analysis must not open the evaluation label store")


def fake_capture(marker: int, token_count: int, *, audit_relay: bool, **identity):
    event_count = token_count - 3
    query = torch.arange(2, 2 + event_count)
    routes = SimpleNamespace(
        all_map=torch.full((event_count, token_count - 1), marker / 75.0),
        local_map=torch.full((event_count, token_count - 1), marker / 100.0),
        global_map=torch.full((event_count, token_count - 1), marker / 50.0),
    )
    rhythm = SimpleNamespace(
        prediction_position=query + 1,
        functional_reach=torch.arange(event_count, dtype=torch.float32),
        relay_capacity=torch.full((event_count,), 0.25),
        carrier_mask=torch.zeros(event_count, dtype=torch.bool),
    )
    arrays = {
        "sample_id": identity["sample_id"],
        "source_id": identity["source_id"],
        "task_type": identity["task_type"],
        "model_id": identity["model_id"],
        "query_position": query,
        "prediction_position": query + 1,
        "target_token_id": 20 + torch.arange(event_count),
        "baseline_margin": torch.ones(event_count),
        "baseline_target_logprob": -torch.ones(event_count),
        "baseline_entropy": torch.ones(event_count),
        "functional_reach": torch.arange(event_count, dtype=torch.float32),
        "relay_capacity": torch.full((event_count,), 0.25),
        "constraint_deficit": torch.zeros(event_count),
        "valid": torch.ones(event_count, dtype=torch.bool),
        "evidence_tokens": 2,
        "control_audited": audit_relay,
        "matched_control_available": audit_relay,
        "relay_audited": audit_relay,
        "direct_response_cut_delta": torch.zeros(event_count),
        "matched_non_evidence_cut_delta": torch.zeros(event_count),
        "upstream_cut_delta": torch.zeros(event_count),
        "downstream_cut_delta": torch.zeros(event_count),
        "joint_cut_delta": torch.zeros(event_count),
        "relay_interaction": torch.zeros(event_count),
    }
    return SimpleNamespace(arrays=arrays, routes=routes, rhythm=rhythm)


def existing_arrays(sample_id: str, task: str):
    return {
        "sample_id": sample_id,
        "source_id": f"source-{sample_id}",
        "task_type": task,
        "model_id": "fixture-model",
        "generator_model": "fixture-generator",
        "observer_model": "fixture-model",
        "query_position": np.asarray([2, 3]),
        "prediction_position": np.asarray([3, 4]),
        "target_token_id": np.asarray([20, 21]),
        "baseline_margin": np.asarray([1.0, 1.0]),
        "baseline_target_logprob": np.asarray([-1.0, -1.0]),
        "baseline_entropy": np.asarray([1.0, 1.0]),
        "functional_reach": np.asarray([0.1, 0.2]),
        "relay_capacity": np.asarray([0.2, 0.1]),
        "constraint_deficit": np.asarray([0.0, 0.0]),
        "valid": np.asarray([True, True]),
        "evidence_tokens": 2,
        "control_audited": True,
    }


def test_audit_sources_are_seeded_source_disjoint_and_inside_limit():
    events = []
    dataset = FakeDataset(
        [
            FakeSample("q1", "shared", "QA", 1, events),
            FakeSample("q2", "shared", "QA", 2, events),
            FakeSample("q3", "inside", "QA", 3, events),
            FakeSample("q4", "outside", "QA", 4, events),
            FakeSample("s1", "summary", "Summary", 5, events),
        ]
    )

    first = analyze.select_audit_sources(dataset, 2, 17, limit=3)
    second = analyze.select_audit_sources(dataset, 2, 17, limit=3)

    assert first == second
    assert set(first["QA"]) == {"shared", "inside"}
    assert "outside" not in first["QA"]
    assert len(first["QA"]) == len(set(first["QA"]))


def test_manifest_refuses_a_changed_run_configuration(tmp_path):
    audit_sources = {task: [] for task in analyze.TASK_TYPES}
    analyze.open_manifest(tmp_path, RUN_CONFIG, audit_sources)

    with pytest.raises(ValueError, match="different run configuration"):
        analyze.open_manifest(
            tmp_path,
            {**RUN_CONFIG, "model_id": "another-model"},
            audit_sources,
        )


def test_manifest_resume_ignores_an_interrupted_atomic_temp(tmp_path):
    audit_sources = {task: [] for task in analyze.TASK_TYPES}
    analyze.open_manifest(tmp_path, RUN_CONFIG, audit_sources)
    temporary = tmp_path / "results" / "QA" / ".sample.npz.tmp.npz"
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b"interrupted")

    manifest = analyze.open_manifest(tmp_path, RUN_CONFIG, audit_sources)

    assert manifest["analysis_complete"] is False


def test_cache_observer_must_match_the_intervention_model():
    assert analyze.observer_matches_run("fixture-model", RUN_CONFIG, "fixture-model")
    assert not analyze.observer_matches_run(
        "another-model", RUN_CONFIG, "fixture-model"
    )
    assert analyze.observer_matches_run("/models/fixture", RUN_CONFIG, "fixture-model")
    assert not analyze.observer_matches_run(
        "/other/fixture", RUN_CONFIG, "fixture-model"
    )
    dataset = SimpleNamespace(spec={"model_path": "/cache/A/fixture-model"})
    assert analyze.observer_identity(dataset, "fixture-model") == (
        "/cache/A/fixture-model"
    )


def test_analyze_streams_label_free_samples_and_keeps_ordered_subsets(
    tmp_path, monkeypatch
):
    events = []
    samples = [
        FakeSample("qa-old", "source-qa-old", "QA", 31, events),
        FakeSample("summary", "source-summary", "Summary", 41, events),
        FakeSample("qa-new", "source-qa-new", "qa", 32, events),
        FakeSample("qa-beyond-limit", "source-extra", "QA", 33, events),
    ]
    dataset = FakeDataset(samples)
    output = tmp_path / "output"
    save_result_file(
        output / "results" / "QA" / "qa-old.npz",
        existing_arrays("qa-old", "QA"),
    )
    old_figure = output / "figures" / "QA" / "qa-old.png"
    old_figure.parent.mkdir(parents=True, exist_ok=True)
    old_figure.write_bytes(b"png")

    source_rows = {
        sample.source_id: {"task_type": sample.task_type} for sample in samples[:3]
    }
    audit_sources = {
        "QA": ["source-qa-old"],
        "Summary": ["source-summary"],
        "Data2txt": [],
    }
    analyze.save_manifest(
        output / analyze.MANIFEST_NAME,
        {
            "config": RUN_CONFIG,
            "analysis_complete": False,
            "selected_samples": {},
            "audit_source_ids": audit_sources,
            "audit_sample_ids": {
                "QA": ["qa-old"],
                "Summary": [],
                "Data2txt": [],
            },
            "samples": [
                {
                    "sample_id": "qa-old",
                    "source_id": "source-qa-old",
                    "task_type": "QA",
                    "generator_model": "fixture-generator",
                    "observer_model": "fixture-model",
                    "result": "results/QA/qa-old.npz",
                    "events": 2,
                    "full_response_events": 3,
                    "evidence_positions": [0, 2],
                    "audit_requested": True,
                    "plot_requested": True,
                    "complete": True,
                }
            ],
        },
    )

    def open_dataset(path, *, device, retain_embedded_labels):
        assert path == tmp_path / "test"
        assert device == "cpu"
        assert retain_embedded_labels is False
        return dataset

    monkeypatch.setattr(analyze, "open_research_dataset", open_dataset)
    monkeypatch.setattr(
        analyze,
        "load_source_info",
        lambda _path: source_rows,
    )

    def build_mask(_source, _tokenizer, token_ids, response_start):
        marker = int(token_ids[1])
        owner = next(
            sample for sample in samples if int(sample.cached.token_ids[1]) == marker
        )
        assert owner.release_calls == 2
        assert events[-1] == f"release:{owner.sample_id}"
        assert response_start == 3
        events.append(f"mask:{owner.sample_id}")
        return np.asarray([True, False, True])

    monkeypatch.setattr(analyze, "build_evidence_mask", build_mask)
    capture_calls = []

    def capture_sample(_model, token_ids, response_start, evidence_mask, **kwargs):
        marker = int(token_ids[1])
        owner = next(
            sample for sample in samples if int(sample.cached.token_ids[1]) == marker
        )
        assert events[-1] == f"mask:{owner.sample_id}"
        assert response_start == 3
        assert token_ids.tolist() == [10, marker, 12, 20, 21]
        assert evidence_mask.tolist() == [True, False, True]
        capture_calls.append((owner.sample_id, kwargs["audit_relay"]))
        events.append(f"capture:{owner.sample_id}")
        return fake_capture(marker, len(token_ids), **kwargs)

    monkeypatch.setattr(analyze, "capture_sample", capture_sample)
    real_save = analyze.save_result

    def save_result(path, arrays):
        events.append(f"save:{path.stem}")
        real_save(path, arrays)

    monkeypatch.setattr(analyze, "save_result", save_result)
    plotted = []

    def save_figure(path, **values):
        plotted.append((path, values))
        events.append(f"plot:{path.stem}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(analyze, "save_sample_figure", save_figure)
    monkeypatch.setattr(analyze.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        analyze.torch.cuda, "empty_cache", lambda: events.append("empty-cache")
    )

    counts = analyze.analyze_split(
        object(),
        SimpleNamespace(
            convert_ids_to_tokens=lambda token_ids: [
                f"token-{token_id}" for token_id in token_ids
            ]
        ),
        tmp_path / "test",
        tmp_path / "source-info.jsonl",
        output,
        model_id="fixture-model",
        limit=2,
        audit_limit=1,
        plot_limit=1,
        max_events=2,
        audit_seed=0,
        run_config=RUN_CONFIG,
    )

    assert capture_calls == [("summary", True), ("qa-new", False)]
    assert [path for path, _ in plotted] == [
        output / "figures" / "Summary" / "summary.png"
    ]
    assert plotted[0][1]["local_route"].shape == (2, 4)
    assert plotted[0][1]["global_route"].shape == (2, 4)
    assert plotted[0][1]["relay_capacity"].shape == (2,)
    assert plotted[0][1]["token_labels"] == ["token-20", "token-21"]
    assert plotted[0][1]["source_token_labels"] == [
        "token-10",
        "token-41",
        "token-12",
        "token-20",
    ]
    assert samples[0].attention_calls == 1
    assert samples[0].release_calls == 2
    assert samples[1].attention_calls == 1
    assert samples[1].release_calls == 2
    assert samples[2].attention_calls == 1
    assert samples[2].release_calls == 2
    assert samples[3].attention_calls == 0
    assert samples[3].release_calls == 2
    assert events.count("gc") == events.count("empty-cache") == 2
    assert events.index("release:summary") < events.index("mask:summary")
    assert events.index("plot:summary") < events.index("save:summary")

    assert counts["QA"] == {
        "selected": 2,
        "saved": 1,
        "skipped": 1,
        "audited": 1,
        "plotted": 1,
    }
    assert counts["Summary"] == {
        "selected": 1,
        "saved": 1,
        "skipped": 0,
        "audited": 1,
        "plotted": 1,
    }
    assert counts["Data2txt"]["selected"] == 0

    qa_result = load_result(output / "results" / "QA" / "qa-new.npz")
    assert qa_result["prediction_position"].tolist() == [3, 4]
    assert "local_map" not in qa_result
    assert "global_map" not in qa_result
    assert qa_result["generator_model"].item() == "fixture-generator"
    manifest = json.loads((output / analyze.MANIFEST_NAME).read_text())
    assert manifest["audit_source_ids"] == audit_sources
    assert manifest["audit_sample_ids"] == {
        "QA": ["qa-old"],
        "Summary": ["summary"],
        "Data2txt": [],
    }
    assert manifest["model_roles"]["intervention_model"] == "fixture-model"


def test_analyze_propagates_capture_failure_after_releasing_cache(
    tmp_path, monkeypatch
):
    events = []
    sample = FakeSample("broken", "source-broken", "Data2txt", 55, events)
    dataset = FakeDataset([sample])
    monkeypatch.setattr(
        analyze, "open_research_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        analyze,
        "load_source_info",
        lambda _path: {sample.source_id: {"task_type": sample.task_type}},
    )
    monkeypatch.setattr(
        analyze,
        "build_evidence_mask",
        lambda *_args: np.asarray([True, False, False]),
    )

    def fail_capture(*_args, **_kwargs):
        events.append("capture-failed")
        raise RuntimeError("native rerun failed")

    monkeypatch.setattr(analyze, "capture_sample", fail_capture)

    with pytest.raises(RuntimeError, match="native rerun failed"):
        analyze.analyze_split(
            object(),
            object(),
            tmp_path / "test",
            tmp_path / "source-info.jsonl",
            tmp_path / "output",
            model_id="fixture-model",
            run_config=RUN_CONFIG,
        )

    assert events == ["attention:broken", "release:broken", "capture-failed"]
    assert not list((tmp_path / "output").rglob("*.npz"))
