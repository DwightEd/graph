from __future__ import annotations

import torch

from experiments.reanchor_flow.units import (
    build_source_units,
    field_spans,
    passage_spans,
    sentence_spans,
)


def test_passage_units_cover_the_complete_evidence_without_overlap() -> None:
    text = "first passage\n\nsecond passage\n\nthird"
    spans = passage_spans(text)
    assert [span.name for span in spans] == [
        "passage:1",
        "passage:2",
        "passage:3",
    ]
    assert spans[0].start == 0
    assert spans[-1].stop == len(text)
    assert all(left.stop == right.start for left, right in zip(spans, spans[1:]))


def test_sentence_units_do_not_split_common_abbreviations() -> None:
    text = "Dr. Ada measured it. The result held! Last claim?"
    spans = sentence_spans(text)
    assert len(spans) == 3
    assert text[spans[0].start : spans[0].stop].startswith("Dr. Ada")


def test_data2txt_units_follow_nested_leaf_fields_and_list_items() -> None:
    text = "{'person': {'name': 'Ada', 'awards': ['A', 'B']}, 'year': 1843}"
    spans = field_spans(text)
    assert [span.name for span in spans] == [
        "person.name",
        "person.awards.0",
        "person.awards.1",
        "year",
    ]
    assert spans[0].start == 0
    assert spans[-1].stop == len(text)
    assert all(left.stop == right.start for left, right in zip(spans, spans[1:]))


def test_data2txt_units_use_absolute_offsets_in_multiline_unicode_input() -> None:
    text = "{\n  '人物': {'name': '艾达'},\n  'year': 1843\n}"
    spans = field_spans(text)
    assert [span.name for span in spans] == ["人物.name", "year"]
    assert "'艾达'" in text[spans[0].start : spans[0].stop]
    assert "1843" in text[spans[1].start : spans[1].stop]
    assert spans[0].stop == spans[1].start


def test_source_units_align_prompt_characters_and_response_carriers() -> None:
    class CharacterTokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return f"SYSTEM|{messages[1]['content']}|ASSISTANT"

        @staticmethod
        def __call__(text, **_kwargs):
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

    prompt = "Question\nfirst\n\nsecond"
    source = {
        "task_type": "QA",
        "prompt": prompt,
        "source_info": {"passages": "first\n\nsecond"},
    }
    tokenizer = CharacterTokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "unused"}, {"role": "user", "content": prompt}]
    )
    response_start = len(rendered)
    token_ids = torch.tensor([*range(response_start), 100, 101])
    units = build_source_units(source, tokenizer, token_ids, response_start)
    assert units.kind[1:3] == ("passage", "passage")
    assert int((units.token_unit_id == 1).sum()) > 0
    assert int((units.token_unit_id == 2).sum()) > 0
    assert units.kind[-1] == "response"
    assert int(units.token_unit_id[-1]) == units.count - 1
