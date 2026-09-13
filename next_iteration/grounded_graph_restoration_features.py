"""Actual source-erased pre-token queries paired with original source graph states.

The original packet retains its original-input meaning. A separate intervention
receipt binds the executed input and H_empty; no original receipt is relabelled.
"""

import argparse
import fcntl
import hashlib
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_feature_runner import load_example as load_original
from next_iteration.grounded_graph_feature_runner import read
from next_iteration.grounded_graph_feature_runner import verify as verify_original
from next_iteration.grounded_graph_synthesize import progress
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest
from route_graph.frozen_reader import digest, write_json_once

REPRESENTATION = "original prompt-only source X; actual all-source-erased final-norm pretoken H_empty query and residual base"
PROTOCOL = {"schema": "grounded-graph-restoration-features@2", "representation": REPRESENTATION,
    "erase": "all prompt token positions intersecting source_span, including field keys and separators",
    "replacement": "one ASCII space token at each position; same length and response history",
    "array_dtype": "float32", "dtype": "bfloat16", "attention": "sdpa", "labels_read": False,
    "native_route_claim": False, "truncation": False,
    "boundary": "a token intersecting the source boundary is replaced whole; any outside-source character overlap is receipted"}


def intervention(packet, tokenizer):
    replacement = tokenizer.encode(" ", add_special_tokens=False)
    if len(replacement) != 1:
        raise ValueError("source erasure needs exactly one space token")
    plen = packet["prompt_record"]["prompt_length"]
    positions = [i for i, (a, b) in enumerate(packet["source_offsets"]) if b > a]
    if not positions or any(not 0 < i < plen for i in positions):
        raise ValueError("source erasure must remain inside prompt source positions")
    tokenized = tokenizer(packet["prompt_record"]["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    if [tokenizer.bos_token_id] + tokenized["input_ids"] != packet["input_ids"][:plen]:
        raise ValueError("erasure tokenizer differs from original input")
    offsets = [[0, 0], *tokenized["offset_mapping"]]
    left, right = packet["prompt_record"]["source_span"]
    expected = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
    if positions != expected or packet["query_positions"] != [t - 1 for t in packet["target_positions"]]:
        raise ValueError("source interval coverage or pretoken timing differs")
    ids = list(packet["input_ids"])
    for i in positions:
        ids[i] = replacement[0]
    return {"packet_sha256": packet["sha256"], "replacement_token_id": replacement[0],
        "erased_prompt_token_indices": positions, "executed_input_ids": ids, "executed_input_sha256": digest(ids),
        "original_input_sha256": digest(packet["input_ids"]), "query_positions": packet["query_positions"],
        "boundary_crossing_tokens": [{"position": i, "prompt_char_span": list(offsets[i])}
            for i in positions if offsets[i][0] < left or offsets[i][1] > right]}


@torch.no_grad()
def capture_empty(model, packet, tokenizer):
    if model.training or any(p.requires_grad for p in model.parameters()) or model.config.model_type != "llama":
        raise ValueError("empty-source capture requires frozen eval Llama")
    receipt = intervention(packet, tokenizer)
    seen = []
    hook = model.model.norm.register_forward_hook(lambda module, inputs, output: seen.append(output))
    try:
        states = model.model(input_ids=torch.tensor([receipt["executed_input_ids"]], device=model.device),
            use_cache=False, return_dict=True).last_hidden_state
        if len(seen) != 1 or states.data_ptr() != seen[0].data_ptr():
            raise ValueError("empty query was not actual final-norm output")
        query = states[0, packet["query_positions"]].float().cpu().numpy()
    finally:
        hook.remove()
    if not np.isfinite(query).all():
        raise ValueError("empty-source query contains nonfinite states")
    receipt.update(actual_observer_forwards=1, actual_final_norm_hook_count=len(seen),
        observer_dtype=str(model.dtype), attention_implementation=model.config._attn_implementation,
        representation=REPRESENTATION, query_shape=list(query.shape), query_dtype=str(query.dtype),
        query_sha256=hashlib.sha256(query.tobytes()).hexdigest())
    return query, receipt


def verify(output, *, complete=True):
    settings, manifest = read(output / "settings.json"), read(output / "manifest.json")
    if not complete or settings["protocol"] != PROTOCOL or manifest["status"] != "complete":
        raise ValueError("completed restoration features with exact protocol required")
    if file_sha256(output / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("restoration settings changed")
    original = Path(settings["original_feature_path"])
    original_settings, entries = verify_original(original, complete=True)
    if file_sha256(original / "manifest.json") != settings["original_feature_manifest_sha256"]:
        raise ValueError("original feature parent changed")
    for key in ("parent", "model_path", "model_files", "inventory_path"):
        if settings[key] != original_settings[key]:
            raise ValueError("restoration original metadata differs")
    if read(output / "entries.json") != entries:
        raise ValueError("restoration omitted or altered original entry roster")
    expected = {"entries.json", "summary.json"} | {e["packet_file"] for e in entries} | {
        f"features/{e['id']}{suffix}" for e in entries if e["status"] == "available" for suffix in (".npz", ".json")}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("restoration artifact census differs")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("restoration artifact changed")
    for entry in entries:
        if file_sha256(output / entry["packet_file"]) != file_sha256(original / entry["packet_file"]):
            raise ValueError("original packet altered by restoration")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(output / "executed_code" / name) != sha:
            raise ValueError("restoration live/snapshot code changed")
    summary = read(output / "summary.json")
    if (summary["examples"] != len(entries) or summary["tokens"] != sum(e["tokens"] for e in entries)
            or summary["counts"] != dict(Counter(e["status"] for e in entries))
            or summary["actual_forwards"] != sum(e["status"] == "available" for e in entries)):
        raise ValueError("restoration complete census differs")
    return settings, entries


def load_example(output, entry, settings, tokenizer):
    original = Path(settings["original_feature_path"])
    packet, arrays, original_receipt = load_original(original, entry, read(original / "settings.json"), tokenizer)
    receipt = read(output / "features" / f"{entry['id']}.json")
    if (receipt["sha256"] != digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt["original_receipt_sha256"] != original_receipt["sha256"]
            or receipt["model_files"] != settings["model_files"]
            or receipt["capture_code_sha256"] != settings["code_sha256"]["next_iteration/grounded_graph_restoration_features.py"]
            or receipt["representation"] != REPRESENTATION or receipt["actual_observer_forwards"] != 1
            or receipt["observer_dtype"] != "torch.bfloat16" or receipt["attention_implementation"] != "sdpa"
            or receipt["actual_final_norm_hook_count"] != 1):
        raise ValueError("restoration receipt identity mismatch")
    if any(receipt[k] != v for k, v in intervention(packet, tokenizer).items()):
        raise ValueError("restoration executed input differs from declared erasure")
    with np.load(output / "features" / f"{entry['id']}.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"query_empty"}:
            raise ValueError("restoration array schema differs")
        query = saved["query_empty"]
    if (query.shape != arrays["query"].shape or list(query.shape) != receipt["query_shape"]
            or query.dtype != np.float32 or receipt["query_dtype"] != "float32"
            or hashlib.sha256(query.tobytes()).hexdigest() != receipt["query_sha256"] or not np.isfinite(query).all()):
        raise ValueError("restoration query bytes/shape/dtype differ")
    return packet, {"source": arrays["source"], "query": query, "full_query": arrays["query"]}, receipt


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh restoration features required; preserve partial outputs")
    original_settings, entries = verify_original(args.original, complete=True)
    code = {**original_settings["code_sha256"], "next_iteration/grounded_graph_restoration_features.py": file_sha256(Path(__file__))}
    for name, sha in code.items():
        destination = args.output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if file_sha256(destination) != sha:
            raise ValueError("restoration code changed during snapshot")
    settings = {k: original_settings[k] for k in ("parent", "model_path", "model_files", "inventory_path")}
    settings.update(protocol=PROTOCOL, original_feature_path=str(args.original.resolve()),
        original_feature_manifest_sha256=file_sha256(args.original / "manifest.json"), code_sha256=code, labels_read=False)
    write_json_once(args.output / "settings.json", settings)
    write_json_once(args.output / "entries.json", entries)
    artifacts = {"entries.json": file_sha256(args.output / "entries.json")}
    for entry in entries:
        destination = args.output / entry["packet_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.original / entry["packet_file"], destination)
        artifacts[entry["packet_file"]] = file_sha256(destination)
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("restoration observer files changed")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive GPU with20GiB free required")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa").eval().requires_grad_(False)
    (args.output / "features").mkdir()
    started, calls = time.monotonic(), 0
    for index, entry in enumerate(entries):
        if entry["status"] == "available":
            packet, _, original_receipt = load_original(args.original, entry, original_settings, tokenizer)
            query, receipt = capture_empty(model, packet, tokenizer)
            receipt.update(original_receipt_sha256=original_receipt["sha256"], model_files=settings["model_files"],
                capture_code_sha256=code["next_iteration/grounded_graph_restoration_features.py"])
            receipt["sha256"] = digest(receipt)
            name = f"features/{entry['id']}"
            with (args.output / (name + ".npz")).open("xb") as stream:
                np.savez(stream, query_empty=query)
            write_json_once(args.output / (name + ".json"), receipt)
            for suffix in (".npz", ".json"):
                artifacts[name + suffix] = file_sha256(args.output / (name + suffix))
            load_example(args.output, entry, settings, tokenizer)
            calls += 1
        if (index + 1) % 8 == 0 or index + 1 == len(entries):
            progress(args.output, status="running", done=index + 1, total=len(entries), actual_forwards=calls,
                seconds=time.monotonic() - started)
    verify_original(args.original, complete=True)
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("restoration observer files changed during capture")
    summary = {"status": "complete", "examples": len(entries), "actual_forwards": calls,
        "tokens": sum(e["tokens"] for e in entries), "counts": dict(Counter(e["status"] for e in entries)),
        "seconds": time.monotonic() - started, "cuda_peak_bytes": torch.cuda.max_memory_allocated(), "labels_read": False}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json")})
    verify(args.output)
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
