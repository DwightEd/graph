"""Frozen natural candidate/owner validation before native mechanism work.

This run produces auxiliary owner features, fixed dense candidates and finite
checks. It does not run native interventions or report hallucination accuracy.
"""

import argparse
import fcntl
import gc
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.reader_receipt import ReceiptReader
from next_iteration.surface_graph import _sealed, edit_candidate
from next_iteration.surface_owner import (
    OWNER_PROTOCOL,
    capture_masked_contexts,
    match_owners,
    verify_capture,
)
from next_iteration.surface_verifier import (
    THRESHOLD,
    assess_target,
    finalize,
    validate_target_assessment,
    verify_candidate,
)
from route_graph.audit_alignment import align_row
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import lock_stream, model_manifest, rows
from route_graph.frozen_reader import digest, write_json_once
from route_graph.soft_graph_runner import verify_executed_code

GRAPH = Path(__file__).resolve().parents[1]
MODEL_ROOT = GRAPH.parent.parent / "models"
PROTOCOL = {"schema": "surface-owner-validation@1", "phase_order": ["features", "finite"],
    "owner": OWNER_PROTOCOL, "finite_threshold": THRESHOLD, "chunk_documents": 32, "batch_size": 4,
    "masked_context_limit": 4096, "reader_context_limit": 8192, "seed": 20260913,
    "selection": "all_roster_slots; frozen_dense_top4; strict_checks_only_for_target_CN_ge_.8",
    "native_forward_calls": 0, "labels_used": False,
    "purpose": "natural_single_slot_contrast_and_constraint_ownership_supply_not_accuracy"}


def verify_preflight(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["labels_read"] or manifest["matcher_mode"] != "lexical_only_preflight":
        raise ValueError("invalid structural parent")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("structural preflight artifact changed")
    for name, sha in manifest["code_sha256"].items():
        original = Path(name)
        if file_sha256(original) != sha or file_sha256(path / "executed_code" / original.parent.name / original.name) != sha:
            raise ValueError("preflight live/snapshot code changed; prepare fresh revision")
    if file_sha256(manifest["input_path"]) != manifest["input_sha256"]:
        raise ValueError("preflight input changed")
    return manifest


def code_files():
    paths = list((GRAPH / "route_graph").glob("*.py"))
    paths.extend(GRAPH / "next_iteration" / n for n in (
        "__init__.py", "reader_receipt.py", "graph_boundaries.py", "surface_graph.py", "surface_owner.py",
        "surface_verifier.py", "surface_runner.py", "surface_preflight.py"))
    paths.extend(GRAPH / "experiments" / n for n in ("interleave_surface_owner.py", "interleave_soft_graph.py"))
    return {str(p.relative_to(GRAPH)): file_sha256(p) for p in sorted(paths)}


def prepare(args):
    parent = verify_preflight(args.preflight)
    observer_tokenizer = AutoTokenizer.from_pretrained(args.observer_model, local_files_only=True)
    documents = json.loads((args.preflight / "documents.json").read_text())["documents"]
    sizes = [len(observer_tokenizer.encode(d["text"], add_special_tokens=True)) for d in documents]
    if not sizes or max(sizes) > PROTOCOL["masked_context_limit"]:
        raise ValueError("masked document inventory empty or over frozen context limit")
    roster = rows(argparse.Namespace(inputs=Path(parent["input_path"])))
    for row in roster:
        align_row(row, observer_tokenizer)
    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True)
    if any(len(reader_tokenizer.encode(k, add_special_tokens=False)) != 1 for k in "SCNIUPFVK"):
        raise ValueError("finite labels must each be one reader token")
    lengths = {"masked_documents": len(sizes), "masked_max_tokens": max(sizes),
        "masked_total_tokens": sum(sizes), "document_lengths_sha256": digest(sizes),
        "original_observer_rows_aligned": len(roster), "finite_single_token_labels_validated": True}
    expected = {"protocol": PROTOCOL, "preflight_path": str(args.preflight),
        "preflight_manifest_sha256": file_sha256(args.preflight / "manifest.json"),
        "input_path": parent["input_path"], "input_sha256": parent["input_sha256"],
        "observer_model": str(args.observer_model), "reader_model": str(args.reader_model),
        "code_sha256": code_files(), "torch": str(torch.__version__), "transformers": transformers.__version__,
        "dtype": "bfloat16", "attention": "sdpa", "labels_used": False, "length_preflight": lengths}
    destination = args.output / "settings.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("settings/runtime/code changed; preserve output and choose new revision")
        for name, path in (("observer_files", args.observer_model), ("reader_files", args.reader_model)):
            if saved[name] != model_manifest(path):
                raise ValueError("frozen model files changed")
        verify_executed_code(args.output, saved)
        return saved
    expected.update(observer_files=model_manifest(args.observer_model), reader_files=model_manifest(args.reader_model))
    for relative, sha in expected["code_sha256"].items():
        target = args.output / "executed_code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("partial snapshot exists; preserve and use a new output")
        shutil.copyfile(GRAPH / relative, target)
        if file_sha256(target) != sha:
            raise ValueError("code changed during snapshot")
    write_json_once(args.output / "executed_code/manifest.json", expected["code_sha256"])
    verify_executed_code(args.output, expected)
    write_json_once(destination, expected)
    return expected


def progress(output, **record):
    value = {"pid": os.getpid(), "unix_time": time.time(), **record}
    temporary = output / "progress.partial.json"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output / "progress.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def _load_model(path, output, phase):
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive free CUDA GPU required; no population overlap")
    progress(output, status="loading_model", stage=phase)
    torch.manual_seed(PROTOCOL["seed"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    return model, tokenizer


def _load_samples(args, frozen):
    parent = verify_preflight(args.preflight)
    if file_sha256(args.preflight / "manifest.json") != frozen["preflight_manifest_sha256"]:
        raise ValueError("structural parent changed")
    roster = rows(argparse.Namespace(inputs=Path(parent["input_path"])))
    samples = [json.loads(line) for line in (args.preflight / "samples.jsonl").open()]
    if [str(r["id"]) for r in roster] != [s["response_id"] for s in samples]:
        raise ValueError("structural parent response roster changed")
    for row, sample in zip(roster, samples, strict=True):
        if (sample["response_graph"]["text"] != row["response"] or sample["source_graph"]["text"] != row["prompt"][slice(*row["source_span"])]):
            raise ValueError("graph differs from original input")
    return roster, samples


def _write_artifact(path, data, frozen, upstream):
    write_json_once(path, {"settings_sha256": digest(frozen), "upstream": upstream,
                           "data_sha256": digest(data), "data": data})


def _read_artifact(path, frozen, upstream):
    saved = json.loads(path.read_text())
    if (saved["settings_sha256"] != digest(frozen) or saved["upstream"] != upstream
            or saved["data_sha256"] != digest(saved["data"])):
        raise ValueError("surface stage artifact differs from frozen inputs")
    return saved["data"]


def _combined_capture(args, frozen, docs):
    identity = {"observer_files": frozen["observer_files"], "purpose": OWNER_PROTOCOL["features"]}
    vectors, receipts, files = {}, [], {}
    chunk_size = PROTOCOL["chunk_documents"]
    for i, start in enumerate(range(0, len(docs), chunk_size)):
        chunk = docs[start:start + chunk_size]
        base = args.output / "features" / f"{i:04d}"
        upstream = {"documents_sha256": digest(chunk)}
        data = _read_artifact(base.with_suffix(".json"), frozen, upstream)
        array = np.load(base.with_suffix(".npy"), allow_pickle=False)
        if data["array_file_sha256"] != file_sha256(base.with_suffix(".npy")) or len(array) != len(chunk):
            raise ValueError("masked feature array file changed")
        batch = {doc["document_id"]: vector for doc, vector in zip(chunk, array, strict=True)}
        verify_capture(batch, data["capture"], chunk, identity)
        vectors.update(batch)
        receipts.append(data["capture"])
        files[str(base.with_suffix(".json").relative_to(args.output))] = file_sha256(base.with_suffix(".json"))
        files[str(base.with_suffix(".npy").relative_to(args.output))] = data["array_file_sha256"]
    first = {k: v for k, v in receipts[0].items() if k not in {"sha256", "records", "feature_forward_calls"}}
    combined = _sealed({**first, "records": [r for receipt in receipts for r in receipt["records"]],
        "feature_forward_calls": sum(r["feature_forward_calls"] for r in receipts)})
    verify_capture(vectors, combined, docs, identity)
    return vectors, combined, identity, files


def feature_stage(args, frozen, samples):
    docs = json.loads((args.preflight / "documents.json").read_text())["documents"]
    chunk_size = PROTOCOL["chunk_documents"]
    pending = [i for i, _ in enumerate(range(0, len(docs), chunk_size))
               if not (args.output / "features" / f"{i:04d}.json").exists()]
    if pending:
        model, tokenizer = _load_model(args.observer_model, args.output, "features")
        identity = {"observer_files": frozen["observer_files"], "purpose": OWNER_PROTOCOL["features"]}
        try:
            (args.output / "features").mkdir(exist_ok=True)
            for i in pending:
                chunk = docs[i * chunk_size:(i + 1) * chunk_size]
                progress(args.output, status="running", stage="features", chunk=i, documents_total=len(docs))
                verify_executed_code(args.output, frozen)
                vectors, receipt = capture_masked_contexts(model, tokenizer, chunk, model_identity=identity,
                    batch_size=PROTOCOL["batch_size"], max_tokens=PROTOCOL["masked_context_limit"])
                base = args.output / "features" / f"{i:04d}"
                with base.with_suffix(".npy").open("xb") as stream:
                    np.save(stream, np.stack([vectors[d["document_id"]] for d in chunk]), allow_pickle=False)
                verify_executed_code(args.output, frozen)
                _write_artifact(base.with_suffix(".json"), {"capture": receipt,
                    "array_file_sha256": file_sha256(base.with_suffix(".npy"))}, frozen, {"documents_sha256": digest(chunk)})
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    vectors, capture, identity, files = _combined_capture(args, frozen, docs)
    (args.output / "B").mkdir(exist_ok=True)
    for sample in samples:
        path = args.output / "B" / (sample["response_id"] + ".json")
        upstream = {"preflight_sample_sha256": digest(sample), "feature_files": files}
        if path.exists():
            _read_artifact(path, frozen, upstream)
            continue
        matches = match_owners(sample["response_graph"], sample["source_graph"], vectors,
                               capture=capture, expected_model_identity=identity)
        _write_artifact(path, matches, frozen, upstream)
    summary = {"feature_forward_calls": capture["feature_forward_calls"],
        "documents": len(docs), "capture_sha256": capture["sha256"], "native_forward_calls": 0}
    summary_path = args.output / "feature_summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text()) != summary:
            raise ValueError("existing feature summary differs from validated capture")
    else:
        write_json_once(summary_path, summary)


def finite_stage(args, frozen, roster, samples):
    docs = json.loads((args.preflight / "documents.json").read_text())["documents"]
    _, _, _, files = _combined_capture(args, frozen, docs)
    pending = []
    for row, sample in zip(roster, samples, strict=True):
        bpath = args.output / "B" / (sample["response_id"] + ".json")
        b = _read_artifact(bpath, frozen, {"preflight_sample_sha256": digest(sample), "feature_files": files})
        cpath = args.output / "C" / (sample["response_id"] + ".json")
        upstream = {"B_sha256": file_sha256(bpath), "row_sha256": digest(row)}
        if cpath.exists():
            _read_artifact(cpath, frozen, upstream)
        else:
            pending.append((row, sample, b, cpath, upstream))
    if not pending:
        return
    model, tokenizer = _load_model(args.reader_model, args.output, "finite")
    observer_tokenizer = AutoTokenizer.from_pretrained(args.observer_model, local_files_only=True)
    identity = {"model": frozen["reader_files"], "tokenizer": frozen["reader_files"], "code_sha256": frozen["code_sha256"]}
    reader = ReceiptReader(model, tokenizer, args.output / "reader_cache", identity, PROTOCOL["reader_context_limit"])
    (args.output / "C").mkdir(exist_ok=True)
    try:
        for row, sample, proposals, path, upstream in pending:
            response, source = sample["response_graph"], sample["source_graph"]
            target_checks, attempts, counts = [], [], Counter()
            initial_calls, initial_outcomes = reader.calls, reader.outcomes.copy()
            for index, match in enumerate(proposals["matches"]):
                progress(args.output, status="running", stage="finite", response_id=row["id"],
                    slot=index, slots_total=len(proposals["matches"]), reader_calls=reader.calls)
                target = assess_target(reader, response, source, match["slot_id"], reader_identity=identity)
                target_checks.append(target)
                q = validate_target_assessment(response, source, target, reader_identity=identity)
                gated = q["C"] + q["N"] >= THRESHOLD
                counts["targets"] += 1
                counts["target_CN_gate_passed"] += gated
                for selected in match["selected"]:
                    draft = edit_candidate(response, source, match["slot_id"], selected["occurrence_id"])
                    status = draft["status"]
                    result = {"draft": draft, "selection": selected, "target_check_sha256": target["sha256"]}
                    if status == "candidate_not_semantically_verified" and gated:
                        verification = verify_candidate(reader, draft, reader_identity=identity)
                        final = finalize(row, response, source, draft, verification, observer_tokenizer, reader_identity=identity)
                        result.update(verification=verification, final=final)
                        status = final["status"]
                        counts["candidate_pairs_finite_checked"] += 1
                    elif status == "candidate_not_semantically_verified":
                        status = "strict_checks_skipped_target_error_not_validated"
                    result["status"] = status
                    counts["pair_status:" + status] += 1
                    attempts.append(result)
            verify_executed_code(args.output, frozen)
            _write_artifact(path, {"response_id": str(row["id"]), "target_checks": target_checks, "attempts": attempts,
                "counts": dict(counts), "reader_calls": reader.calls - initial_calls,
                "reader_outcomes": dict(reader.outcomes - initial_outcomes), "labels_used": False,
                "native_forward_calls": 0, "not_ground_truth": True}, frozen, upstream)
    finally:
        del reader, model, tokenizer, observer_tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-model", type=Path, default=MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--reader-model", type=Path, default=MODEL_ROOT / "Qwen3-8B")
    parser.add_argument("--stage", choices=["prepare", "all", *PROTOCOL["phase_order"]], default="all")
    parser.add_argument("--inherited-lock-fd", type=int)
    args = parser.parse_args()
    for key in ("preflight", "output", "observer_model", "reader_model"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with lock_stream(args) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = prepare(args)
        if args.stage == "prepare":
            print(json.dumps({"status": "prepared", "settings_sha256": digest(frozen)}), flush=True)
            return
        roster, samples = _load_samples(args, frozen)
        for stage in PROTOCOL["phase_order"] if args.stage == "all" else [args.stage]:
            verify_executed_code(args.output, frozen)
            if stage == "features":
                feature_stage(args, frozen, samples)
            else:
                finite_stage(args, frozen, roster, samples)
            progress(args.output, status="phase_complete", stage=stage)
        verify_executed_code(args.output, frozen)
        progress(args.output, status="complete", stage=args.stage, native_forward_calls=0, labels_evaluated=False)


if __name__ == "__main__":
    main()
