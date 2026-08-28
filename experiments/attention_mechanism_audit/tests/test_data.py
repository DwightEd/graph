import json

import numpy as np
import pytest

from experiments.attention_mechanism_audit.data import (
    CONSTRAINT,
    EVIDENCE,
    HISTORICAL_SYSTEM_PROMPT,
    OTHER_PROMPT,
    QUESTION,
    build_prompt_role_ids,
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
            "passages": "Ada is in Paris.",
        },
        "prompt": prompt,
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


def test_prompt_roles_exactly_partition_cached_prompt():
    tokenizer = CharacterTokenizer()
    source = qa_source()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    cached_ids = np.asarray([ord(character) for character in rendered] + [999])

    roles = build_prompt_role_ids(
        source,
        tokenizer,
        cached_ids,
        response_start=len(rendered),
    )

    prompt_start = rendered.index(source["prompt"])
    question_start = rendered.index("Where is Ada?", prompt_start)
    evidence_start = rendered.index("Ada is in Paris.", prompt_start)
    constraint_start = rendered.index("Answer the question:", prompt_start)
    assert roles.dtype == np.int8
    assert roles.shape == (len(rendered),)
    assert roles[0] == OTHER_PROMPT
    assert roles[constraint_start] == CONSTRAINT
    assert np.all(
        roles[question_start : question_start + len("Where is Ada?")] == QUESTION
    )
    assert np.all(
        roles[evidence_start : evidence_start + len("Ada is in Paris.")] == EVIDENCE
    )
    assert roles[-1] == OTHER_PROMPT


def test_rebuilt_prompt_must_equal_cached_prefix():
    tokenizer = CharacterTokenizer()
    source = qa_source()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    cached_ids = np.asarray([ord(character) for character in rendered])
    cached_ids[len(cached_ids) // 2] += 1

    with pytest.raises(ValueError, match="exactly match cached token prefix"):
        build_prompt_role_ids(
            source,
            tokenizer,
            cached_ids,
            response_start=len(rendered),
        )


def test_only_qa_roles_are_built():
    tokenizer = CharacterTokenizer()
    source = {**qa_source(), "task_type": "Summary"}

    with pytest.raises(ValueError, match="supports QA only"):
        build_prompt_role_ids(source, tokenizer, [], response_start=1)
