import pytest

from route_graph.data import text_digest, write_jsonl
from route_graph.evaluation import RouteEvaluator


def test_evaluation_covers_first_error_at_token_zero_and_rejects_missing_tokens(
    tmp_path,
):
    rows = []
    for response, labels in [("a", [1, 1, 0]), ("b", [0, 1, 0])]:
        for index, label in enumerate(labels):
            rows.append(
                {
                    "schema": "route-graph/score@1",
                    "response_id": response,
                    "source_id": response,
                    "response_sha256": text_digest("a b c"),
                    "token_index": index,
                    "token_count": 3,
                    "char_span": [index * 2, index * 2 + 1],
                    "token_id": index,
                    "candidate_ids": [0, 1],
                    "reference_active_features": 0 if index == 0 else 2,
                    "scores": {
                        "residual": label,
                        "null": 0,
                        "signal": 0,
                        "position": index,
                    },
                }
            )
    score_path, labels_path = tmp_path / "scores.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(score_path, rows)
    write_jsonl(
        labels_path,
        [
            {
                "id": "a",
                "source_id": "a",
                "response": "a b c",
                "labels": [{"start": 0, "end": 3}],
            },
            {
                "id": "b",
                "source_id": "b",
                "response": "a b c",
                "labels": [{"start": 2, "end": 3}],
            },
        ],
    )
    result = RouteEvaluator(
        score_path, labels_path, tmp_path / "evaluation.json", 20, 42
    ).run()
    assert result["subsets"]["all_tokens"]["tokens"] == 6
    assert result["subsets"]["first_error"]["positives"] == 2
    assert result["subsets"]["span_onsets"]["positives"] == 2
    assert result["subsets"]["continuations"]["positives"] == 1
    assert result["subsets"]["all_tokens"]["metrics"]["residual"]["auroc"] == 1
    assert result["paired_auroc_gain"]["residual_minus_null"]["estimate"] == 0.5
    assert result["reference_coverage"]["inactive_tokens"] == 2
    assert result["reference_coverage"]["active_token_fraction"] == pytest.approx(4 / 6)
    missing = tmp_path / "missing.jsonl"
    write_jsonl(missing, rows[1:])
    with pytest.raises(ValueError, match="complete token stream"):
        RouteEvaluator(missing, labels_path, tmp_path / "bad.json", 0, 42).run()
