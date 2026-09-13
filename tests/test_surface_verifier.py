"""Protocol fixtures below are explicitly not natural experiment outcomes."""

import copy

import pytest
from test_source_pointer_contrast import IDENTITY, FiniteReader, fixture

from next_iteration.surface_graph import (
    edit_candidate,
    source_occurrences,
    surface_graph,
)
from next_iteration.surface_verifier import (
    assess_target,
    finalize,
    validate_target_assessment,
    verification_status,
    verify_candidate,
)
from route_graph.causal_contrast import ContinuationContrast
from route_graph.frozen_reader import digest


@pytest.fixture(autouse=True)
def reader_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(FiniteReader, "cache_root", tmp_path, raising=False)


def setup(extra=""):
    row, _, _, tokenizer = fixture(extra=extra)
    response = surface_graph(row["response"], sample_id=row["id"], side="response")
    source = source_occurrences(row["prompt"][slice(*row["source_span"])], sample_id=row["source_id"], task=row["task"])
    slot = next(s for s in response["slots"] if s["quote"] == "9")
    occurrence = next(o for o in source["occurrences"] if o["field_path"] == [0, "crates"])
    draft = edit_candidate(response, source, slot["slot_id"], occurrence["occurrence_id"])
    return row, response, source, draft, tokenizer


@pytest.mark.parametrize(("decisions", "status"), [
    ("USSPV", "target_error_not_validated"),
    ("CCSPV", "edited_full_base_not_supported_or_multislot_error"),
    ("CSIPV", "selected_owner_or_constraint_not_validated"),
    ("CSSFV", "non_target_meaning_or_grammar_not_preserved"),
    ("CSSPK", "value_only_edit_not_validated"),
    ("CSSPU", "value_only_edit_not_validated"),
    ("CSSPV", "conditional_contrast_available"),
])
def test_each_required_gate_controls_native_handoff(decisions, status):
    *_, draft, _ = setup()
    result = verify_candidate(FiniteReader(decisions), draft, reader_identity=IDENTITY)
    assert verification_status(draft, result, reader_identity=IDENTITY) == status


def test_full_base_and_scope_preserved_into_exact_native_event():
    row, response, source, draft, tokenizer = setup(extra=" per hour")
    reader = FiniteReader("CSSPV")
    verification = verify_candidate(reader, draft, reader_identity=IDENTITY)
    result = finalize(row, response, source, draft, verification, tokenizer, reader_identity=IDENTITY)
    assert result["status"] == "conditional_contrast_available"
    assert result["metadata"]["original"] == "Alpha shipped 9 crates per hour."
    assert result["metadata"]["alternative"] == "Alpha shipped 7 crates per hour."
    assert result["source_occurrence"]["field_path"] == [0, "crates"]
    assert result["source_keys"] and result["target_keys"]
    contrast = ContinuationContrast(**result["contrast"])
    assert len(contrast.continuations[0]) == len("9 crates per hour.")
    assert sum(result["slot_masks"][0]) == 1 and result["certificate_count"] == 0
    assert reader.payloads[0]["slot"]["span"] == [14, 15]
    assert reader.payloads[2]["occurrence"]["field_path"] == [0, "crates"]


def test_changed_prediction_or_reader_identity_rejected():
    *_, draft, _ = setup()
    verification = verify_candidate(FiniteReader("CSSPV"), draft, reader_identity=IDENTITY)
    with pytest.raises(ValueError, match="model/protocol"):
        verification_status(draft, verification, reader_identity={"wrong": "reader"})
    bad = copy.deepcopy(verification)
    bad["checks"]["edit_kind"]["scores"] = {"V": 0., "K": 1., "U": 0.}
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="immutable prediction"):
        verification_status(draft, bad, reader_identity=IDENTITY)


def test_wrong_source_parent_cannot_be_substituted_after_verification():
    row, response, source, draft, tokenizer = setup()
    verification = verify_candidate(FiniteReader("CSSPV"), draft, reader_identity=IDENTITY)
    bad = copy.deepcopy(draft)
    other = next(o for o in source["occurrences"] if o["field_path"] == [1, "crates"])
    bad["occurrence"] = other
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="another candidate"):
        finalize(row, response, source, bad, verification, tokenizer, reader_identity=IDENTITY)


def test_assessment_consumption_revalidates_model_cache_and_scores():
    from pathlib import Path

    _, response, source, draft, _ = setup()
    assessment = assess_target(FiniteReader("C"), response, source, draft["slot"]["slot_id"], reader_identity=IDENTITY)
    assert validate_target_assessment(response, source, assessment, reader_identity=IDENTITY)["C"] == 1
    with pytest.raises(ValueError, match="graph/slot/model"):
        validate_target_assessment(response, source, assessment, reader_identity={"wrong": "reader"})
    changed = copy.deepcopy(assessment)
    changed["scores"] = {"S": 1., "C": 0., "N": 0., "U": 0.}
    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="immutable prediction"):
        validate_target_assessment(response, source, changed, reader_identity=IDENTITY)
    Path(assessment["reader_record"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="cache bytes changed"):
        validate_target_assessment(response, source, assessment, reader_identity=IDENTITY)


def test_local_llama_tokenizer_keeps_complete_base_event_exact():
    from pathlib import Path

    from transformers import AutoTokenizer

    row, response, source, draft, _ = setup(extra=" per hour")
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).parents[3] / "models/Meta-Llama-3.1-8B-Instruct",
        local_files_only=True,
    )
    row = copy.deepcopy(row)
    prompt = tokenizer(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    completion = tokenizer(row["response"], add_special_tokens=False, return_offsets_mapping=True)
    row["prompt_length"] = 1 + len(prompt["input_ids"])
    row["token_ids"] = [tokenizer.bos_token_id, *prompt["input_ids"], *completion["input_ids"]]
    source_start, source_end = row["source_span"]
    row["source_mask"] = [False, *[
        right > left and left < source_end and right > source_start
        for left, right in prompt["offset_mapping"]
    ], *([False] * len(completion["input_ids"]))]

    verification = verify_candidate(FiniteReader("CSSPV"), draft, reader_identity=IDENTITY)
    result = finalize(row, response, source, draft, verification, tokenizer, reader_identity=IDENTITY)
    contrast = ContinuationContrast(**result["contrast"])
    assert list(contrast.prefix + contrast.continuations[0]) == row["token_ids"]
    assert all(any(mask) for mask in result["slot_masks"])
