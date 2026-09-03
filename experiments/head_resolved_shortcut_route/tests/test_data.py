import json

import numpy as np
import pytest

from experiments.head_resolved_shortcut_route.data import (
    HISTORICAL_SYSTEM_PROMPT,
    build_evidence_mask,
    load_source_info,
    render_historical_prompt,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0] == {
            "role": "system",
            "content": HISTORICAL_SYSTEM_PROMPT,
        }
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


def qa_source():
    prompt = (
        "Answer the question: Where is Ada?\n"
        "passages:\nAda is in Paris.\n"
        "In case the passages do not answer it, say unknown.\n"
        "output:"
    )
    return {
        "source_id": "source-1",
        "task_type": "QA",
        "source_info": {
            "question": "Where is Ada?",
            "passages": "Ada is in Paris.\n",
        },
        "prompt": prompt,
    }


def summary_source():
    evidence = "Ada moved to Paris. She works there."
    return {
        "source_id": "source-2",
        "task_type": "Summary",
        "source_info": evidence,
        "prompt": f"Summarize the following news within 20 words:\n{evidence}\noutput:",
    }


def data2txt_source():
    evidence = {"name": "Ada's Cafe", "city": "Paris", "WiFi": None}
    return {
        "source_id": "source-3",
        "task_type": "Data2txt",
        "source_info": evidence,
        "prompt": f"Instruction:\nUse only this data.\nStructured data:\n{evidence}\nOverview:",
    }


def test_source_info_is_keyed_by_source_id(tmp_path):
    path = tmp_path / "source_info.jsonl"
    rows = [qa_source(), {**qa_source(), "source_id": 2}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    sources = load_source_info(path)

    assert set(sources) == {"source-1", "2"}
    assert sources["source-1"]["source_info"]["question"] == "Where is Ada?"


def test_duplicate_source_id_is_rejected(tmp_path):
    path = tmp_path / "source_info.jsonl"
    row = qa_source()
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source_id"):
        load_source_info(path)


@pytest.mark.parametrize("source", [qa_source(), summary_source(), data2txt_source()])
def test_evidence_mask_exactly_matches_each_task_source(source):
    tokenizer = CharacterTokenizer()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    cached_ids = np.asarray([ord(character) for character in rendered] + [999])

    mask = build_evidence_mask(
        source,
        tokenizer,
        cached_ids,
        response_start=len(rendered),
    )

    evidence = source["source_info"]
    if source["task_type"] == "QA":
        evidence = evidence["passages"].removesuffix("\n")
    evidence = str(evidence)
    evidence_start = rendered.index(source["prompt"]) + source["prompt"].index(evidence)
    expected = np.zeros(len(rendered), dtype=bool)
    expected[evidence_start : evidence_start + len(evidence)] = True
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)


def test_rebuilt_prompt_must_equal_cached_prefix():
    tokenizer = CharacterTokenizer()
    source = qa_source()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    cached_ids = np.asarray([ord(character) for character in rendered])
    cached_ids[len(cached_ids) // 2] += 1

    with pytest.raises(ValueError, match="exactly match cached token prefix"):
        build_evidence_mask(
            source,
            tokenizer,
            cached_ids,
            response_start=len(rendered),
        )


def test_unknown_task_is_rejected():
    tokenizer = CharacterTokenizer()
    source = {**qa_source(), "task_type": "Translation"}

    with pytest.raises(ValueError, match="unsupported task type"):
        build_evidence_mask(source, tokenizer, [], response_start=1)
