import json

import pytest

from route_graph.alarms import alarm_positions, evaluate, weighted_quantile
from route_graph.archive import digest
from route_graph.data import text_digest, write_jsonl


def test_alarm_decisions_use_only_past_and_strict_threshold():
    assert alarm_positions([1, 2, 3, 9, 10], 2, cooldown=2) == [2, 4]
    assert alarm_positions([1, 2, 3], 2, cooldown=2) == [2]
    assert weighted_quantile([0, 100], [0.9, 0.1], 0.95) == 100
    assert weighted_quantile([0, 100], [0.9, 0.1], 0.5) == 0


def test_post_error_alarms_cannot_improve_onset_recall(tmp_path):
    text = "abcdefghijkl"
    rows = [
        {
            "schema": "route-graph/score@1",
            "response_id": "r",
            "source_id": "s",
            "token_index": t,
            "token_count": len(text),
            "char_span": [t, t + 1],
            "response_sha256": text_digest(text),
            "scores": {"early": 2 if t == 4 else 0, "late": 2 if t == 10 else 0},
        }
        for t in range(len(text))
    ]
    scores, labels, thresholds = (
        tmp_path / "scores.jsonl",
        tmp_path / "labels.jsonl",
        tmp_path / "thresholds.json",
    )
    write_jsonl(scores, rows)
    write_jsonl(
        labels,
        [
            {
                "id": "r",
                "source_id": "s",
                "response": text,
                "labels": [{"start": 5, "end": 9}],
            }
        ],
    )
    thresholds.write_text(
        json.dumps(
            {
                "schema": "route-graph/alarm-thresholds@1",
                "labels_used": False,
                "archive_index_sha256": "fixture",
                "thresholds": {"early": 1, "late": 1},
            }
        )
    )
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "archive_index_sha256": "fixture",
                "scores_sha256": digest(scores),
                "labels_used": False,
                "neighbors": 3,
                "per_source": 8,
            }
        )
    )
    result = evaluate(scores, labels, thresholds, tmp_path / "report.json")
    assert result["methods"]["early"]["hits"] == 1
    assert result["methods"]["early"]["mean_lead"] == 1
    assert result["methods"]["late"]["hits"] == 0
    assert result["methods"]["late"]["alarms"] == 0
    assert result["methods"]["late"]["observed_tokens"] == 6

    assert alarm_positions([3, 5, 8], 1, cooldown=1, budget=1) == [0]
    frozen = json.loads(thresholds.read_text())
    frozen["archive_index_sha256"] = "another-run"
    thresholds.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="different archives"):
        evaluate(scores, labels, thresholds, tmp_path / "bad.json")
    frozen["archive_index_sha256"] = "fixture"
    thresholds.write_text(json.dumps(frozen))
    original = labels.read_text()
    labels.write_text(original + original)
    with pytest.raises(ValueError, match="duplicate annotation"):
        evaluate(scores, labels, thresholds, tmp_path / "bad.json")
    annotation = json.loads(original)
    annotation["labels"] = [{"start": -1, "end": 100}]
    labels.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="invalid annotation"):
        evaluate(scores, labels, thresholds, tmp_path / "bad.json")
    completed = json.loads((tmp_path / "complete.json").read_text())
    completed["neighbors"] = 7
    (tmp_path / "complete.json").write_text(json.dumps(completed))
    with pytest.raises(ValueError, match="neighbors=3"):
        evaluate(scores, labels, thresholds, tmp_path / "bad.json")


def test_auroc_entrypoint_rejects_changed_scores_before_labels(tmp_path, monkeypatch):
    from route_graph.archive_evaluation import main

    scores = tmp_path / "scores.jsonl"
    scores.write_text("{}\n")
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "labels_used": False,
                "scores_sha256": "different-score-file",
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "archive_evaluation",
            "evaluate",
            "--scores",
            str(scores),
            "--labels",
            str(tmp_path / "absent-labels.jsonl"),
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
    )
    with pytest.raises(ValueError, match="completed scoring run"):
        main()
