import json

import pytest

from control_graph.evaluation import DetectionEvaluator, EvaluationConfig


def write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def score(event_id: str, source_id: str, value: float) -> dict:
    return {
        "schema": "control-graph/anomaly-score@1",
        "event_id": event_id,
        "source_id": source_id,
        "split": "test",
        "relation": "temporal",
        "anomaly_score": value,
        "dominant_edge": "source_followup",
        "edge_deviations": {
            "source_onset": 0.0,
            "source_followup": value,
            "prefix_followup": 0.0,
            "source_prefix_coupling": 0.0,
        },
    }


def test_evaluation_opens_labels_only_after_scores_are_frozen(tmp_path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        scores_path,
        [
            score("n1", "s1", 0.1),
            score("h1", "s2", 0.9),
            score("n2", "s3", 0.2),
            score("h2", "s4", 0.8),
        ],
    )
    write_jsonl(
        labels_path,
        [
            {"event_id": "n1", "label": 0},
            {"event_id": "h1", "label": 1},
            {"event_id": "n2", "label": 0},
            {"event_id": "h2", "label": 1},
        ],
    )

    report = DetectionEvaluator(
        EvaluationConfig(scores_path, labels_path, tmp_path / "report.json", bootstrap=20)
    ).run()

    assert report["auroc"] == 1.0
    assert report["auprc"] == 1.0
    assert report["labels_used_stage"] == "evaluation_only"
    assert set(report) == {
        "schema",
        "events",
        "sources",
        "prevalence",
        "auroc",
        "auprc",
        "cluster_bootstrap",
        "bootstrap_replicates",
        "valid_bootstrap_replicates",
        "confidence_intervals",
        "labels_used_stage",
    }


def test_evaluation_requires_exact_score_label_alignment(tmp_path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(scores_path, [score("n1", "s1", 0.1), score("h1", "s2", 0.9)])
    write_jsonl(labels_path, [{"event_id": "n1", "label": 0}])

    with pytest.raises(ValueError, match="exactly match"):
        DetectionEvaluator(
            EvaluationConfig(scores_path, labels_path, tmp_path / "report.json")
        ).run()
