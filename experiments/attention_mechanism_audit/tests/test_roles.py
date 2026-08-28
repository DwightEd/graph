import json

import numpy as np
import pytest

from experiments.attention_mechanism_audit import roles as roles_module
from experiments.attention_mechanism_audit.roles import (
    CONSTRAINT,
    EVIDENCE,
    OTHER_PROMPT,
    QUESTION,
    PromptRoleMap,
    build_prompt_role_map,
    expand_role_spans,
    load_role_jsonl,
    position_stratified_role_permutation,
    prompt_token_sha256,
    render_historical_prompt,
    sample_role_permutation_seed,
    validate_cached_prompt,
    write_role_jsonl,
)
from experiments.attention_mechanism_audit.tests.helpers import (
    CharacterChatTokenizer,
    data2txt_source,
    qa_source,
    summary_source,
)


def _build(source):
    tokenizer = CharacterChatTokenizer()
    rendered = render_historical_prompt(tokenizer, source["prompt"])
    cached = np.asarray([ord(character) + 1 for character in rendered] + [900, 901])
    return build_prompt_role_map(source, tokenizer, cached, len(rendered)), rendered


def test_qa_roles_rebuild_exact_cache_prefix_and_cover_prompt():
    role_map, rendered = _build(qa_source())
    source = qa_source()
    user_start = rendered.index(source["prompt"])
    question_start = user_start + source["prompt"].index("Which color?")
    evidence_start = user_start + source["prompt"].index("passage 1:")

    assert role_map.role_ids.shape == (len(rendered),)
    assert set(role_map.role_ids.tolist()) == {
        EVIDENCE,
        QUESTION,
        CONSTRAINT,
        OTHER_PROMPT,
    }
    assert role_map.role_ids[0] == OTHER_PROMPT
    assert role_map.role_ids[question_start] == QUESTION
    assert role_map.role_ids[evidence_start] == EVIDENCE
    assert role_map.role_ids[user_start] == CONSTRAINT
    assert sum(end - start for _, start, end in role_map.role_spans) == len(rendered)


@pytest.mark.parametrize("source_factory", [summary_source, data2txt_source])
def test_task_specific_summary_and_data2txt_evidence_anchors(source_factory):
    role_map, rendered = _build(source_factory())

    assert role_map.role_mask("evidence").any()
    assert not role_map.role_mask("question").any()
    assert role_map.role_ids[0] == OTHER_PROMPT
    assert role_map.prompt_token_sha256 == prompt_token_sha256(
        [ord(character) + 1 for character in rendered]
    )


def test_cached_prefix_mismatch_and_hash_mismatch_are_rejected():
    role_map, rendered = _build(qa_source())
    cached = np.asarray([ord(character) + 1 for character in rendered] + [900])
    cached[4] += 1
    with pytest.raises(ValueError, match="differs at position 4"):
        build_prompt_role_map(qa_source(), CharacterChatTokenizer(), cached, len(rendered))
    with pytest.raises(ValueError, match="does not match prompt_token_sha256"):
        validate_cached_prompt(
            None,
            np.arange(6),
            4,
            expected_sha256=role_map.prompt_token_sha256,
        )


def test_role_spans_must_be_an_exact_partition():
    valid = [
        {"role": "other_prompt", "start": 0, "end": 1},
        {"role": "constraint", "start": 1, "end": 2},
        {"role": "question", "start": 2, "end": 4},
        {"role": "evidence", "start": 4, "end": 6},
    ]
    np.testing.assert_array_equal(
        expand_role_spans(valid, 6),
        [OTHER_PROMPT, CONSTRAINT, QUESTION, QUESTION, EVIDENCE, EVIDENCE],
    )
    gap = [valid[0], {"role": "evidence", "start": 2, "end": 6}]
    overlap = [valid[0], {"role": "evidence", "start": 0, "end": 6}]
    with pytest.raises(ValueError, match="exactly partition"):
        expand_role_spans(gap, 6)
    with pytest.raises(ValueError, match="exactly partition"):
        expand_role_spans(overlap, 6)


def test_role_jsonl_round_trip_is_label_free_and_keeps_replayable_prompt(tmp_path):
    role_map, _ = _build(qa_source())
    path = write_role_jsonl({role_map.source_id: role_map}, tmp_path / "roles.jsonl")
    raw = path.read_text(encoding="utf-8")

    assert "label" not in raw
    assert "prompt_token_ids" in raw
    loaded = load_role_jsonl(path)[role_map.source_id]
    np.testing.assert_array_equal(loaded.role_ids, role_map.role_ids)
    np.testing.assert_array_equal(loaded.prompt_token_ids, role_map.prompt_token_ids)
    assert loaded.prompt_token_sha256 == role_map.prompt_token_sha256


def test_role_jsonl_failed_atomic_replace_preserves_previous_index(
    tmp_path, monkeypatch
):
    role_map, _ = _build(qa_source())
    path = tmp_path / "roles.jsonl"
    previous = b"previous-complete-index\n"
    path.write_bytes(previous)

    def fail_replace(_source, _destination):
        raise OSError("sentinel replace failure")

    monkeypatch.setattr(roles_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="sentinel replace failure"):
        write_role_jsonl({role_map.source_id: role_map}, path)

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(".roles.jsonl.*.tmp")) == []


def test_role_jsonl_rejects_duplicate_source_rows(tmp_path):
    role_map, _ = _build(qa_source())
    row = json.dumps(role_map.to_json())
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate prompt-role row"):
        load_role_jsonl(path)


def test_loaded_role_map_validates_cached_hash_and_persisted_token_ids():
    role_map, rendered = _build(qa_source())
    loaded = PromptRoleMap.from_json(role_map.to_json())
    cached = np.asarray([ord(character) + 1 for character in rendered] + [999])

    loaded.validate(cached)
    cached[0] += 1
    with pytest.raises(ValueError, match="differs at position 0"):
        loaded.validate(cached)

    corrupted = role_map.to_json()
    corrupted["prompt_token_ids"][0] += 1
    with pytest.raises(ValueError, match="do not match prompt_token_sha256"):
        PromptRoleMap.from_json(corrupted)


def _distance_bins(length: int, bin_width: int) -> np.ndarray:
    distance = length - 1 - np.arange(length)
    return distance // bin_width


def test_position_stratified_role_null_preserves_each_bin_and_global_counts():
    roles = np.asarray(
        [
            EVIDENCE,
            QUESTION,
            CONSTRAINT,
            OTHER_PROMPT,
            QUESTION,
            EVIDENCE,
            OTHER_PROMPT,
            CONSTRAINT,
            EVIDENCE,
            QUESTION,
            OTHER_PROMPT,
            CONSTRAINT,
        ],
        dtype=np.int8,
    )
    original = roles.copy()
    permuted = position_stratified_role_permutation(roles, bin_width=4, seed=91)
    bins = _distance_bins(len(roles), 4)

    np.testing.assert_array_equal(roles, original)
    assert not np.shares_memory(permuted, roles)
    np.testing.assert_array_equal(
        np.bincount(permuted, minlength=4),
        np.bincount(roles, minlength=4),
    )
    for current_bin in np.unique(bins):
        selected = bins == current_bin
        np.testing.assert_array_equal(
            np.bincount(permuted[selected], minlength=4),
            np.bincount(roles[selected], minlength=4),
        )


def test_position_stratified_role_null_is_seed_reproducible():
    roles = np.tile(
        np.asarray([EVIDENCE, QUESTION, CONSTRAINT, OTHER_PROMPT]),
        8,
    )

    left = position_stratified_role_permutation(roles, bin_width=8, seed=1234)
    right = position_stratified_role_permutation(roles, bin_width=8, seed=1234)
    different = position_stratified_role_permutation(roles, bin_width=8, seed=5678)

    np.testing.assert_array_equal(left, right)
    assert not np.array_equal(left, different)


def test_position_stratified_role_null_leaves_constant_bins_unchanged():
    roles = np.asarray(
        [
            EVIDENCE,
            EVIDENCE,
            QUESTION,
            QUESTION,
            CONSTRAINT,
            CONSTRAINT,
            OTHER_PROMPT,
            OTHER_PROMPT,
        ]
    )

    permuted = position_stratified_role_permutation(roles, bin_width=2, seed=7)

    np.testing.assert_array_equal(permuted, roles)
    assert permuted is not roles


@pytest.mark.parametrize("bin_width", [-2, -1, 0, 1, True, 2.5])
def test_position_stratified_role_null_rejects_invalid_bin_width(bin_width):
    with pytest.raises(ValueError, match="bin_width"):
        position_stratified_role_permutation(
            np.asarray([EVIDENCE, QUESTION]),
            bin_width=bin_width,
            seed=1,
        )


def test_sample_role_permutation_seed_uses_complete_multicharacter_identity():
    tokens = np.asarray([128000, 451, 72, 9001], dtype=np.int64)

    first = sample_role_permutation_seed(20260828, "回答-样本-10", tokens)
    repeated = sample_role_permutation_seed(20260828, "回答-样本-10", tokens.copy())
    other_sample = sample_role_permutation_seed(20260828, "回答-样本-01", tokens)
    other_tokens = sample_role_permutation_seed(
        20260828,
        "回答-样本-10",
        np.asarray([128000, 451, 9001, 72]),
    )

    assert first == repeated
    assert first != other_sample
    assert first != other_tokens
    assert 0 <= first < 2**64
