import json

import pytest

from route_graph.data import RagtruthPreparer, read_jsonl, read_responses


def test_preparation_uses_source_splits_and_ignores_annotations(tmp_path):
    sources = [
        {
            "source_id": str(i),
            "task_type": "QA",
            "prompt": f"Q evidence {i} A:",
            "source_info": {"passages": f"evidence {i}"},
        }
        for i in range(4)
    ]
    responses = [
        {
            "id": str(i),
            "source_id": str(i // 2),
            "model": "fixture",
            "response": "Answer here",
            "labels": [{"start": 0, "end": 6}],
            "quality": "bad",
        }
        for i in range(8)
    ]
    for name, rows in [("source_info", sources), ("response", responses)]:
        (tmp_path / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    first = tmp_path / "prepared.jsonl"
    RagtruthPreparer(tmp_path, first, "QA", "fixture", 4, 42).run()
    rows = read_responses(first)
    assert {r["split"] for r in rows} == {"train", "test"}
    assert len({(r["source_id"], r["split"]) for r in rows}) == 4
    for response in responses:
        response["labels"] = []
        response["quality"] = "good"
    (tmp_path / "response.jsonl").write_text("\n".join(map(json.dumps, responses)))
    second = tmp_path / "second.jsonl"
    RagtruthPreparer(tmp_path, second, "QA", "fixture", 4, 42).run()
    assert read_jsonl(first) == read_jsonl(second)
    rows[0]["label"] = 1
    first.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="fields"):
        read_responses(first)
