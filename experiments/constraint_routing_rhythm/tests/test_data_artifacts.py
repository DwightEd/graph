import json
import zipfile

import numpy as np
import pytest

from experiments.constraint_routing_rhythm.artifacts import load_result, save_result
from experiments.constraint_routing_rhythm.data import (
    HISTORICAL_SYSTEM_PROMPT,
    build_evidence_mask,
    load_source_info,
    render_historical_prompt,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["content"] == HISTORICAL_SYSTEM_PROMPT
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user><assistant>"
        )

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def source_row(task):
    if task == "QA":
        evidence = {"question": "Where is Ada?", "passages": "Ada is in Paris.\n"}
        prompt = "Where is Ada?\npassages:\nAda is in Paris.\noutput:"
    elif task == "Summary":
        evidence = "Ada moved to Paris."
        prompt = f"Summarize this:\n{evidence}\noutput:"
    else:
        evidence = {"name": "Ada's Cafe", "city": "Paris"}
        prompt = f"Structured data:\n{evidence}\nOverview:"
    return {
        "source_id": f"source-{task}",
        "task_type": task,
        "source_info": evidence,
        "prompt": prompt,
    }


@pytest.mark.parametrize("task", ["QA", "Summary", "Data2txt"])
def test_evidence_mask_matches_each_task(task):
    source = source_row(task)
    tokenizer = CharacterTokenizer()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    token_ids = np.asarray([ord(character) for character in rendered] + [999])

    actual = build_evidence_mask(
        source,
        tokenizer,
        token_ids,
        response_start=len(rendered),
    )

    evidence = source["source_info"]
    if task == "QA":
        evidence = evidence["passages"].removesuffix("\n")
    evidence = str(evidence)
    start = rendered.index(source["prompt"]) + source["prompt"].index(evidence)
    expected = np.zeros(len(rendered), dtype=bool)
    expected[start : start + len(evidence)] = True
    np.testing.assert_array_equal(actual, expected)


def test_source_rows_are_keyed_without_identity_fields(tmp_path):
    rows = [source_row("QA"), source_row("Summary")]
    path = tmp_path / "source_info.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    sources = load_source_info(path)

    assert set(sources) == {"source-QA", "source-Summary"}


def test_result_roundtrip_is_uncompressed_and_thin(tmp_path):
    result = {
        "sample_id": np.asarray("sample-1"),
        "source_id": np.asarray("source-1"),
        "query_position": np.asarray([4, 5, 6], dtype=np.int64),
        "prediction_position": np.asarray([5, 6, 7], dtype=np.int64),
        "target_token_id": np.asarray([10, 11, 12], dtype=np.int64),
        "constraint_effect": np.asarray([0.2, -0.1, 0.4], dtype=np.float32),
    }
    path = tmp_path / "result.npz"

    save_result(path, result)
    loaded = load_result(path)

    for name, expected in result.items():
        np.testing.assert_array_equal(loaded[name], expected)
    with zipfile.ZipFile(path) as archive:
        assert all(
            member.compress_type == zipfile.ZIP_STORED for member in archive.infolist()
        )
    assert not list(tmp_path.glob(".*.tmp.npz"))


@pytest.mark.parametrize(
    "prediction,score,error",
    [
        ([5, 7], [0.1, 0.2], "query_position \\+ 1"),
        ([5, 6], [0.1], "wrong length"),
    ],
)
def test_load_rejects_damaged_event_alignment(tmp_path, prediction, score, error):
    path = tmp_path / "damaged.npz"
    np.savez(
        path,
        query_position=np.asarray([4, 5]),
        prediction_position=np.asarray(prediction),
        constraint_effect=np.asarray(score),
    )

    with pytest.raises(ValueError, match=error):
        load_result(path)
