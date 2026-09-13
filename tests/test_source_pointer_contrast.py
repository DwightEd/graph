"""Scientific interface checks; fixtures are not mechanism experiment results."""

import copy
import json
from typing import ClassVar

import pytest

from next_iteration.source_pointer_contrast import (
    finalize_contrast,
    prepare_contrast,
    verification_status,
    verify_contrast,
)
from route_graph.audit_alignment import align_row, span_keys
from route_graph.audit_artifacts import file_sha256
from route_graph.causal_contrast import ContinuationContrast
from route_graph.event_matcher import match_catalog
from route_graph.frozen_reader import digest
from route_graph.source_event_graph import (
    compile_literal_fields,
    compile_pointer_events,
    raw_inventory,
)


class CharacterTokenizer:
    bos_token_id = 1

    def __call__(self, text, **_):
        return {"input_ids": [ord(c) + 3 for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FiniteReader:
    identity: ClassVar[dict] = {"model": "cpu_fixture", "tokenizer": "characters", "code_sha256": "fixture_only"}

    def __init__(self, decisions="CSSP"):
        self.decisions = iter(decisions)
        self.payloads = []

    def ask(self, instruction, payload, labels, max_new_tokens):
        assert max_new_tokens == 1
        self.payloads.append(copy.deepcopy(payload))
        chosen = next(self.decisions)
        prediction = {"labels": labels, "probabilities": [float(k == chosen) for k in labels]}
        request = {"model": self.identity, "instruction": instruction, "payload": payload,
                   "labels": labels, "max_new_tokens": max_new_tokens,
                   "generation_config": "cpu_mock_not_model_result"}
        path = self.cache_root / (digest(request) + ".json")
        path.write_text(json.dumps({"request": request, "prediction": prediction}))
        self.last_record = {"path": str(path), "sha256": file_sha256(path)}
        return prediction


IDENTITY = FiniteReader.identity


@pytest.fixture(autouse=True)
def reader_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(FiniteReader, "cache_root", tmp_path, raising=False)


def fixture(value=7, role="quantity", target="9", extra="", source=None):
    source = source or repr([{"name": "Alpha", "crates": value}, {"name": "Beta", "crates": value}])
    response = f"Alpha shipped {target} crates{extra}."
    si = raw_inventory(source, side="source", sample_id="s")
    ri = raw_inventory(response, side="response", sample_id="r")
    tokens = ri["units"][0]["tokens"]

    def pointer(start, end):
        indices = [i for i, t in enumerate(tokens) if start <= t["span"][0] < t["span"][1] <= end]
        return {"unit_id": "u0", "start_token": min(indices), "end_token": max(indices) + 1}

    roles = [{"role": "subject", "pointer": pointer(0, 5)},
             {"role": "predicate", "pointer": pointer(6, 13)}]
    if role == "subject":
        # Subject safety is exercised with a real subject pointer and record value.
        target_span = [0, 5]
    else:
        target_span = [14, 14 + len(target)]
        roles.append({"role": role, "pointer": pointer(*target_span)})
    rg = compile_pointer_events(ri, {"inventory_sha256": ri["sha256"], "side": "response",
                                    "sample_id": "r", "prediction": {"events": [
                                        {"anchor": pointer(0, len(response)), "roles": roles}]}})
    assert rg["events"] and not rg["failures"]
    sg = compile_literal_fields(si)
    sc, rc = match_catalog(si, sg), match_catalog(ri, rg)
    source_node = next(n for n in sc["nodes"] if n["path"] == [0, "crates"])
    target_node = next(n for n in rc["nodes"] if n["role"] == role)
    tokenizer = CharacterTokenizer()
    prompt = "Source: " + source + "\nAnswer: "
    prompt_ids = [1] + tokenizer(prompt)["input_ids"]
    source_span = [8, 8 + len(source)]
    row = {"id": "r", "source_id": "s", "task": "Data2txt", "prompt": prompt,
           "response": response, "source_span": source_span, "prompt_length": len(prompt_ids),
           "token_ids": prompt_ids + tokenizer(response)["input_ids"]}
    row["source_mask"] = [False] + [source_span[0] <= i < source_span[1] for i in range(len(prompt))] + [False] * len(response)
    alignment = align_row(row, tokenizer)
    member = source_node["memberships"][0]
    candidate = {"source_node_id": source_node["id"], "span": source_node["span"],
                 "membership": member, "field_path": source_node["path"]}
    term = {"key": [target_node["id"], source_node["id"], member["id"]],
            "target_role": target_node, "candidate": candidate,
            "keys": span_keys(alignment, "source", source_node["span"])}
    structure = {"source_inventory": si, "response_inventory": ri, "source_graph": sg, "response_graph": rg}
    features = {"source_catalog": sc, "response_catalog": rc, "claims": [
        {"claim_id": "claim", "span": [0, len(response)], "text": response,
         "event_ids": [rg["events"][0]["id"]], "source_terms": [term]}]}
    return row, structure, features, tokenizer


def draft_for(parts):
    return prepare_contrast(*parts[:3], "claim", 0)


def reseal(record):
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})


@pytest.mark.parametrize(("value", "role", "target", "status"), [
    (None, "quantity", "9", "literal_None_is_unknown"),
    (True, "attribute", "false", "boolean_requires_explicit_lexical_policy"),
    ("Beta", "subject", "9", "subject_coreference_not_supported_in_v1"),
    ("new", "condition", "9", "role_requires_whole_event_rewrite"),
    ("Beta", "object", '"Gamma"', "quoted_slot_requires_separate_surface_policy"),
    (9, "quantity", "9", "identical_surface_no_contrast"),
    ("two\nlines", "object", "9", "nonprintable_or_empty_surface"),
])
def test_unsafe_surfaces_never_create_semantic_or_native_requests(value, role, target, status):
    draft = draft_for(fixture(value, role, target))
    assert draft["status"] == status
    with pytest.raises(ValueError, match="rejected"):
        verify_contrast(FiniteReader(), draft, reader_identity=IDENTITY)


def test_repeated_value_is_bound_to_selected_record_and_not_support_elsewhere():
    draft = draft_for(fixture())
    assert draft["same_surface_occurrence_count"] == 2
    reader = FiniteReader("CSIP")
    verification = verify_contrast(reader, draft, reader_identity=IDENTITY)
    assert verification_status(draft, verification, reader_identity=IDENTITY) == "ambiguous_same_value_origin"
    node_payload = reader.payloads[2]
    assert node_payload["highlighted_source_node"]["path"] == [0, "crates"]
    assert node_payload["selected_source_membership"] == draft["source_membership"]
    assert reader.payloads[0]["current_span"] == "Alpha shipped 9 crates."
    assert reader.payloads[1]["current_span"] == "Alpha shipped 7 crates."


def test_numeric_surface_preserves_explicit_unit_and_requires_semantic_unit_check():
    parts = fixture(value=7.5, extra=" per hour")
    draft = draft_for(parts)
    assert draft["edited_prefix_text"] == "Alpha shipped 7.5 crates per hour."
    reader = FiniteReader("CSSF")
    verification = verify_contrast(reader, draft, reader_identity=IDENTITY)
    assert verification_status(draft, verification, reader_identity=IDENTITY) == "non_target_scope_or_grammar_not_validated"
    assert finalize_contrast(*parts[:3], draft, verification, parts[3], reader_identity=IDENTITY)["certificate_count"] == 0


def test_second_unsupported_role_cannot_be_repaired_or_omitted():
    parts = fixture(extra=" to Mars")
    draft = draft_for(parts)
    verification = verify_contrast(FiniteReader("CCSP"), draft, reader_identity=IDENTITY)
    assert "to Mars" in draft["payload"]["edited_event"]
    assert verification_status(draft, verification, reader_identity=IDENTITY) == "single_slot_edited_support_not_validated"


def test_full_event_sequences_keep_non_target_characters_and_all_branch_logprobs():
    parts = fixture()
    draft = draft_for(parts)
    verification = verify_contrast(FiniteReader(), draft, reader_identity=IDENTITY)
    result = finalize_contrast(*parts[:3], draft, verification, parts[3], reader_identity=IDENTITY)
    assert result["status"] == "conditional_contrast_available"
    contrast = ContinuationContrast(**result["contrast"])
    assert len(contrast.continuations[0]) == len("9 crates.")
    assert sum(result["slot_masks"][0]) == 1
    assert result["metadata"]["answer_spans"] == [[14, 15], [14, 15]]
    assert result["source_keys"] == parts[2]["claims"][0]["source_terms"][0]["keys"]
    assert contrast.score([[-1.] * 9, [-2.] * 9]) == 9.
    assert result["not_ground_truth"] and result["certificate_count"] == 0


@pytest.mark.parametrize("tamper", ["scope", "request", "scores", "membership", "keys"])
def test_edited_scope_selected_occurrence_and_finite_prediction_cannot_drift(tamper):
    parts = fixture()
    draft = draft_for(parts)
    verification = verify_contrast(FiniteReader(), draft, reader_identity=IDENTITY)
    if tamper == "scope":
        draft["edited_prefix_text"] += " Another claim."
        reseal(draft)
    elif tamper == "request":
        verification["checks"]["node"]["request_sha256"] = "another-node"
        reseal(verification)
    elif tamper == "scores":
        verification["checks"]["original"]["scores"] = {"S": 0., "C": .5, "N": .5, "U": 0.}
        reseal(verification)
    elif tamper == "membership":
        parts[2]["claims"][0]["source_terms"][0]["candidate"]["membership"] = {"id": "elsewhere", "kind": "literal_record"}
    else:
        parts[2]["claims"][0]["source_terms"][0]["keys"] = [2]
    with pytest.raises(ValueError):
        finalize_contrast(*parts[:3], draft, verification, parts[3], reader_identity=IDENTITY)


@pytest.mark.parametrize("tamper", ["reader_identity", "cache_bytes", "cache_hash"])
def test_finite_result_requires_same_frozen_reader_and_exact_cache(tamper):
    from pathlib import Path

    parts = fixture()
    draft = draft_for(parts)
    verification = verify_contrast(FiniteReader(), draft, reader_identity=IDENTITY)
    record = verification["checks"]["node"]["reader_record"]
    if tamper == "reader_identity":
        verification["reader_identity"] = {**IDENTITY, "model": "different_model"}
        reseal(verification)
    elif tamper == "cache_hash":
        record["sha256"] = "wrong_cache"
        reseal(verification)
    else:
        path = Path(record["path"])
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        finalize_contrast(*parts[:3], draft, verification, parts[3], reader_identity=IDENTITY)


@pytest.mark.parametrize("change_during_execution", [False, True])
def test_preparation_freezes_executing_code_before_constructing_drafts(tmp_path, monkeypatch, change_during_execution):
    from next_iteration import source_pointer_prepare as entry

    row, a, b, _ = fixture()
    run = tmp_path / "parent"
    run.mkdir()
    inputs = run / "inputs.jsonl"
    inputs.write_text(json.dumps(row) + "\n")
    frozen = {"input_path": str(inputs), "input_sha256": file_sha256(inputs)}
    (run / "settings.json").write_text(json.dumps(frozen))
    for stage in ("A", "B"):
        (run / stage).mkdir()
        (run / stage / "r.json").write_text(stage)
    source_dir = tmp_path / "code"
    source_dir.mkdir()
    for name in ("__init__.py", "source_pointer_prepare.py", "source_pointer_contrast.py", "reader_receipt.py"):
        (source_dir / name).write_text("# fixture " + name)
    monkeypatch.setattr(entry, "__file__", str(source_dir / "source_pointer_prepare.py"))
    monkeypatch.setattr(entry, "verify_executed_code", lambda *_: None)
    monkeypatch.setattr(entry, "rows", lambda _: [row])
    monkeypatch.setattr(entry, "read_artifact", lambda _, stage, *__: a if stage == "A" else b)
    original = entry.prepare_contrast

    def construct(*args):
        result = original(*args)
        if change_during_execution:
            (source_dir / "source_pointer_contrast.py").write_text("# altered after first draft")
        return result

    monkeypatch.setattr(entry, "prepare_contrast", construct)
    output = tmp_path / "out"
    if change_during_execution:
        with pytest.raises(ValueError, match="changed during preparation"):
            entry.prepare_run(run, output)
        assert not (output / "manifest.json").exists()
        assert (output / "executed_code" / "source_pointer_contrast.py").read_text().startswith("# fixture")
    else:
        result = entry.prepare_run(run, output)
        assert result["counts"]["attempts"] == 1
        assert result["native_forward_calls"] == result["reader_calls"] == 0
        manifest = json.loads((output / "manifest.json").read_text())
        assert len(manifest["code_sha256"]) == 4
        assert manifest["drafts_sha256"] == file_sha256(output / "drafts.jsonl")


def test_local_llama_tokenizer_finalizes_exact_complete_events():
    from pathlib import Path

    from transformers import AutoTokenizer

    row, structure, features, _ = fixture()
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).parents[3] / "models/Meta-Llama-3.1-8B-Instruct",
        local_files_only=True,
    )
    row = copy.deepcopy(row)
    prompt = tokenizer(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    response = tokenizer(row["response"], add_special_tokens=False, return_offsets_mapping=True)
    row["prompt_length"] = 1 + len(prompt["input_ids"])
    row["token_ids"] = [tokenizer.bos_token_id, *prompt["input_ids"], *response["input_ids"]]
    source_start, source_end = row["source_span"]
    row["source_mask"] = [False, *[
        right > left and left < source_end and right > source_start
        for left, right in prompt["offset_mapping"]
    ], *([False] * len(response["input_ids"]))]
    alignment = align_row(row, tokenizer)
    features = copy.deepcopy(features)
    term = features["claims"][0]["source_terms"][0]
    term["keys"] = span_keys(alignment, "source", term["candidate"]["span"])

    draft = prepare_contrast(row, structure, features, "claim", 0)
    verification = verify_contrast(FiniteReader(), draft, reader_identity=IDENTITY)
    result = finalize_contrast(
        row, structure, features, draft, verification, tokenizer, reader_identity=IDENTITY
    )
    contrast = ContinuationContrast(**result["contrast"])
    assert list(contrast.prefix + contrast.continuations[0]) == row["token_ids"]
    assert all(any(mask) for mask in result["slot_masks"])
