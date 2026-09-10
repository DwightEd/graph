import json

import pytest

from control_graph.pipeline import (
    BuildConfig,
    BuildGraphDataset,
    DetectGraphAnomalies,
    DetectionConfig,
)


def write_events(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def event(index: int, *, split: str, source_id: str, shift: float = 0.0) -> dict:
    return {
        "schema": "control-graph/factorial-event@1",
        "event_id": f"event-{index}",
        "source_id": source_id,
        "split": split,
        "relation": "temporal",
        "margins": {
            "onset_a": 1.0 + shift,
            "onset_b": -1.0,
            "world_a_after_a": 1.2 + shift,
            "world_a_after_b": 0.8,
            "world_b_after_a": -0.8,
            "world_b_after_b": -1.2,
        },
    }


def test_build_then_detect_keeps_labels_outside_the_scoring_path(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    records = [
        event(index, split="train", source_id=f"train-{index}", shift=index * 0.01)
        for index in range(6)
    ]
    records.extend(
        [
            event(10, split="test", source_id="test-10", shift=0.02),
            event(11, split="test", source_id="test-11", shift=4.0),
        ]
    )
    write_events(events_path, records)
    graph_path = tmp_path / "graphs.jsonl"

    build_summary = BuildGraphDataset(BuildConfig(events_path, graph_path)).run()
    result = DetectGraphAnomalies(
        DetectionConfig(graph_path, tmp_path / "detection", "train", "test")
    ).run()

    assert build_summary["graphs"] == 8
    assert set(build_summary) == {"schema", "graphs", "sources", "splits", "output"}
    assert result["fit_graphs"] == 6
    assert result["scored_graphs"] == 2
    score_records = [
        json.loads(line)
        for line in (tmp_path / "detection" / "scores.jsonl").read_text().splitlines()
    ]
    assert all("label" not in record for record in score_records)
    model = json.loads((tmp_path / "detection" / "model.json").read_text())
    assert set(model) == {"schema", "edge_order", "profiles", "fit_split", "fit_sources"}
    assert set(score_records[0]["edge_deviations"]) == {
        "source_onset",
        "source_followup",
        "prefix_followup",
        "source_prefix_coupling",
    }
    assert score_records[1]["anomaly_score"] > score_records[0]["anomaly_score"]


def test_factorial_input_rejects_embedded_hallucination_labels(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    record = event(0, split="train", source_id="source-0")
    record["label"] = 1
    write_events(events_path, [record])

    with pytest.raises(ValueError, match="unexpected fields"):
        BuildGraphDataset(BuildConfig(events_path, tmp_path / "graphs.jsonl")).run()


def test_detection_rejects_source_overlap_between_fit_and_score(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    records = [
        event(index, split="train", source_id=f"source-{index}") for index in range(4)
    ]
    records.append(event(10, split="test", source_id="source-0"))
    write_events(events_path, records)
    graph_path = tmp_path / "graphs.jsonl"
    BuildGraphDataset(BuildConfig(events_path, graph_path)).run()

    with pytest.raises(ValueError, match="source-disjoint"):
        DetectGraphAnomalies(
            DetectionConfig(graph_path, tmp_path / "detection", "train", "test")
        ).run()
