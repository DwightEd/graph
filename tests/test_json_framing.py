import pytest

from route_graph.json_framing import parse_framed_json


def test_intact_object_retains_all_values_and_records_terminal_punctuation():
    raw = '{"quote":"brace } inside string", "value": 2}]%'
    parsed = parse_framed_json(raw)
    assert parsed["prediction"] == {"quote": "brace } inside string", "value": 2}
    assert parsed["framing"]["discarded_suffix"] == "]%"
    assert parsed["framing"]["status"] == "recovered"


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":1',
        '{"answer":1} {"answer":2}',
        '{"answer":1} correct answer is 2',
        '{"answer":1,"answer":2}',
        '{"answer": NaN}',
        '{"answer": 1e1000}',
        '[{"answer":1}]',
    ],
)
def test_no_completion_content_repair_or_silent_alternative_selection(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_framed_json(raw)


def test_strict_valid_json_does_not_claim_repair():
    parsed = parse_framed_json(' {"quote":"原文"} \n')
    assert parsed["prediction"]["quote"] == "原文"
    assert parsed["framing"]["status"] == "strict"


def test_markdown_fence_is_counted_as_recovered_not_strict():
    parsed = parse_framed_json('```json\n{"answer":1}\n```')
    assert parsed["framing"]["status"] == "recovered"
    assert parsed["framing"]["removed_json_fence"]


@pytest.mark.parametrize(
    "raw",
    [
        ' \n {"answer":1}]% \t\n',
        ' \n```json\n  {"answer":1}]  \n```\t\n',
        ' \n {"answer":1} \t\n',
    ],
)
def test_lossless_raw_prefix_suffix_and_root_coordinates(raw):
    parsed = parse_framed_json(raw)
    framing = parsed["framing"]
    a, b = framing["raw_root_span"]
    assert framing["raw_prefix"] + raw[a:b] + framing["raw_suffix"] == raw
    assert raw[a:b] == '{"answer":1}'
    assert framing["raw_suffix"].endswith("\n")
