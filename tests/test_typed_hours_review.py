"""Review regressions for the typed-hours native handoff boundary."""

from next_iteration.typed_hours import DAYS, native_contrast, prepare_row


class CharTokenizer:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        result = {"input_ids": [ord(ch) + 2 for ch in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i, _ in enumerate(text)]
        return result


def _row_with_second_sentence_hours():
    source_text = repr({
        "name": "Cafe",
        "hours": dict.fromkeys(DAYS, "8:0-16:0"),
        "attributes": {"WiFi": "free"},
    })
    prompt = "Instruction: summarize this record. Source: " + source_text
    response = (
        "Cafe serves brunch. "
        "The business hours are from 8 AM to 6 PM every day, and they offer free WiFi. "
        "Outdoor seating is available."
    )
    tokenizer = CharTokenizer()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    source_start = prompt.index(source_text)
    source_end = source_start + len(source_text)
    token_ids = [tokenizer.bos_token_id] + prompt_ids + response_ids
    source_mask = [False] * len(token_ids)
    for index in range(1 + source_start, 1 + source_end):
        source_mask[index] = True
    return {
        "id": "review-fixture",
        "source_id": "review-source",
        "task": "Data2txt",
        "prompt": prompt,
        "response": response,
        "source_span": [source_start, source_end],
        "prompt_length": len(prompt_ids) + 1,
        "token_ids": token_ids,
        "source_mask": source_mask,
    }, tokenizer


def test_native_handoff_uses_complete_prefix_through_non_initial_base_only():
    row, tokenizer = _row_with_second_sentence_hours()
    prepared = prepare_row(row)
    fact_id = next(i for i, fact in enumerate(prepared["facts"]) if fact["status"] == "typed_contrast_available")

    result = native_contrast(row, prepared, fact_id, tokenizer)

    fact = prepared["facts"][fact_id]
    original_event = result["metadata"]["original"]
    alternative_event = result["metadata"]["alternative"]
    assert original_event == row["response"][: fact["base"]["span"][1]]
    assert "Outdoor seating" not in original_event
    a, b = result["metadata"]["answer_spans"][0]
    assert original_event[a:b] == "6 PM"
    assert alternative_event[a:b] == "4 PM"
    assert result["source_keys"]
    assert len(result["source_keys_per_day"]) == 7
    assert all(keys for keys in result["source_keys_per_day"])
    assert [sum(mask) for mask in result["slot_masks"]] == [4, 4]


def test_native_bridge_json_round_trip_preserves_canonical_token_proof():
    import json

    from route_graph.audit_phase_native import event_contrast
    from route_graph.frozen_reader import digest

    row, tokenizer = _row_with_second_sentence_hours()
    prepared = prepare_row(row)
    fact_id = next(i for i, fact in enumerate(prepared["facts"]) if fact["status"] == "typed_contrast_available")
    bridge = native_contrast(row, prepared, fact_id, tokenizer)
    serialized = json.loads(json.dumps(bridge))

    # JSON turns the tuple-valued ContinuationContrast fields into lists, while
    # preserving the sealed canonical payload and the exact reconstructed event.
    assert serialized != bridge
    assert serialized["sha256"] == bridge["sha256"]
    assert serialized["sha256"] == digest({key: value for key, value in serialized.items() if key != "sha256"})
    restored = event_contrast(serialized)
    assert list(restored.prefix) == serialized["contrast"]["prefix"]
    assert [list(item) for item in restored.continuations] == serialized["contrast"]["continuations"]
