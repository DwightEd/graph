"""Freeze and capture real pre-token features for weak training or natural response scoring."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.grounded_graph_data import ROOT, inventory_parent
from next_iteration.grounded_graph_features import (
    EDGE_KINDS,
    NODE_KINDS,
    REPRESENTATION,
    capture,
    prepare_example,
    validate_packet,
)
from next_iteration.grounded_graph_reconstruction import verify as verify_reconstruction
from next_iteration.grounded_graph_synthesize import progress
from next_iteration.grounded_graph_synthesize import verify as verify_synthesis
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest, rows
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "grounded-graph-features@1", "max_tokens": 16384,
    "representation": REPRESENTATION, "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS,
    "dtype": "bfloat16", "attention": "sdpa", "array_dtype": "float32", "labels_read": False,
    "truncation": False, "availability": "mechanical coordinate/length only; all words remain in denominator"}


def read(path):
    return json.loads(Path(path).read_text())


def freeze_code(output):
    paths = list((ROOT / "route_graph").glob("*.py")) + [ROOT / "next_iteration" / n for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "grounded_graph_data.py",
         "grounded_graph_synthesize.py", "grounded_graph_features.py", "grounded_graph_feature_runner.py", "grounded_graph_reconstruction.py")]
    hashes = {}
    for path in paths:
        name = str(path.relative_to(ROOT))
        hashes[name] = file_sha256(path)
        destination = output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if file_sha256(destination) != hashes[name]:
            raise ValueError("feature code changed during snapshot")
    return hashes


def parent_data(path):
    settings, prepared = verify_synthesis(path)
    manifest = read(path / "manifest.json")
    if (manifest["status"] != "complete" or manifest["settings_sha256"] != prepared["settings_sha256"]
            or file_sha256(path / "prepare_manifest.json") != manifest["prepare_manifest_sha256"]):
        raise ValueError("weak training data incomplete or unbound")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("weak data artifact changed")
    return settings, prepared


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh feature output required")
    code = freeze_code(args.output)
    observer_files = model_manifest(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.reconstruction is not None:
        source_settings, examples = verify_reconstruction(args.reconstruction)
        inventory_path = Path(source_settings["inventory_path"])
        input_sha = source_settings["input_sha256"]
        parent = {"kind": "source_reconstruction", "path": str(args.reconstruction.resolve()),
            "manifest_sha256": file_sha256(args.reconstruction / "manifest.json")}
    elif args.data is not None:
        data_settings, prepared = parent_data(args.data)
        inventory_path = Path(data_settings["inventory_path"])
        parent = {"kind": "weak_training", "path": str(args.data.resolve()),
            "manifest_sha256": file_sha256(args.data / "manifest.json")}
        input_sha = data_settings["input_sha256"]
        prompts = {x["source_id"]: x for x in read(args.data / "source_prompts.json")}
        examples = []
        for name in prepared["request_order"]:
            sid = read(args.data / name)["source_id"]
            result = read(args.data / "results" / f"{sid}.json")["compiled"]
            for example in result["examples"]:
                examples.append({"id": sid + "_" + example["target_id"], "prompt_record": prompts[sid],
                    "text": example["text"], "targets": [example], "split": result["split"], "task": result["task"],
                    "source_id": sid, "source_text_sha256": prompts[sid]["source_text_sha256"]})
        parent["selected_target_denominator"] = sum(read(args.data / name)["target_census"]["selected"] for name in prepared["request_order"])
    else:
        if args.inventory is None:
            raise ValueError("natural inputs require full source inventory")
        inventory_path = args.inventory
        inv_settings = read(inventory_path / "settings.json")
        input_sha = inv_settings["input_sha256"]
        parent = {"kind": "natural_response", "path": str(args.inputs.resolve()), "input_sha256": file_sha256(args.inputs)}
        examples = []
        for row in rows(args):
            sid = str(row["source_id"])
            prompt = {"source_id": sid, "task": row["task"], "prompt": row["prompt"],
                "prompt_token_ids": row["token_ids"][:row["prompt_length"]], "prompt_length": row["prompt_length"],
                "source_span": row["source_span"], "original_row_id": str(row["id"])}
            examples.append({"id": str(row["id"]), "prompt_record": prompt, "text": row["response"], "targets": [],
                "source_id": sid, "split": row["official_split"], "task": row["task"], "generator": row["generator"],
                "expected_token_ids": row["token_ids"], "expected_offsets": row["offsets"], "row_sha256": digest(row),
                "source_text_sha256": hashlib.sha256(row["prompt"][slice(*row["source_span"])].encode()).hexdigest()})
    inv_manifest, inv_settings = inventory_parent(inventory_path, input_sha)
    settings = {"protocol": PROTOCOL, "parent": parent, "inventory_path": str(inventory_path.resolve()),
        "inventory_manifest_sha256": file_sha256(inventory_path / "manifest.json"),
        "inventory_settings_sha256": inv_manifest["settings_file_sha256"], "inventory_input_sha256": input_sha,
        "model_path": str(args.model.resolve()), "model_files": observer_files, "source_artifacts": {}, "labels_read": False}
    entries, artifacts, cache = [], {}, {}
    if not examples or len({x["id"] for x in examples}) != len(examples):
        raise ValueError("feature examples must be nonempty and unique")
    for example in examples:
        sid = example["source_id"]
        name = f"sources/{sid}.json"
        if sid not in cache:
            sha = file_sha256(inventory_path / name)
            wrapped = read(inventory_path / name)
            if sha != inv_manifest["artifacts"][name] or wrapped["settings_object_digest"] != digest(inv_settings):
                raise ValueError("source inventory/settings changed")
            settings["source_artifacts"][name] = sha
            cache[sid] = wrapped["data"]
        inventory = cache[sid]
        if inventory["task"] != example["task"]:
            raise ValueError("source task differs from feature example")
        packet = prepare_example(example["prompt_record"], inventory, example["text"], example["targets"], tokenizer)
        if "expected_token_ids" in example and (packet["input_ids"] + packet["target_token_ids"][-1:] != example["expected_token_ids"]
                or packet["response_offsets"] != example["expected_offsets"]):
            raise ValueError("natural original replay tokens/offsets differ")
        status = "available" if len(packet["input_ids"]) <= PROTOCOL["max_tokens"] and any(n["available"] for n in packet["nodes"]) else "unavailable_length_or_source"
        name = f"packets/{example['id']}.json"
        write_json_once(args.output / name, packet)
        artifacts[name] = file_sha256(args.output / name)
        entries.append({k: v for k, v in example.items() if k not in
            ("prompt_record", "text", "targets", "expected_token_ids", "expected_offsets")} | {
                "packet_file": name, "packet_sha256": packet["sha256"], "status": status,
                "tokens": len(packet["target_token_ids"]), "input_tokens": len(packet["input_ids"]), "nodes": len(packet["nodes"])})
    settings["code_sha256"] = code
    if model_manifest(args.model) != observer_files:
        raise ValueError("observer/tokenizer files changed during packet preparation")
    write_json_once(args.output / "settings.json", settings)
    write_json_once(args.output / "entries.json", entries)
    artifacts["entries.json"] = file_sha256(args.output / "entries.json")
    write_json_once(args.output / "prepare_manifest.json", {"status": "prepared", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json")})
    verify(args.output)
    print(json.dumps({"examples": len(entries), "counts": dict(Counter(e["status"] for e in entries)),
        "source_count": len(cache), "tokens": sum(e["tokens"] for e in entries), "max_input_tokens": max(e["input_tokens"] for e in entries),
        "model_forwards": 0}), flush=True)


def verify(output, *, complete=False):
    settings, prepared = read(output / "settings.json"), read(output / "prepare_manifest.json")
    if settings["protocol"] != PROTOCOL or file_sha256(output / "settings.json") != prepared["settings_sha256"]:
        raise ValueError("feature settings/protocol changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(output / "executed_code" / name) != sha:
            raise ValueError("feature live/snapshot code changed")
    for name, sha in prepared["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("prepared feature artifact changed")
    inventory_path = Path(settings["inventory_path"])
    inv, _ = inventory_parent(inventory_path, settings["inventory_input_sha256"])
    if (file_sha256(inventory_path / "manifest.json") != settings["inventory_manifest_sha256"]
            or inv["settings_file_sha256"] != settings["inventory_settings_sha256"]):
        raise ValueError("feature inventory parent changed")
    for name, sha in settings["source_artifacts"].items():
        if file_sha256(inventory_path / name) != sha:
            raise ValueError("feature source inventory changed")
    parent = settings["parent"]
    if parent["kind"] in ("weak_training", "source_reconstruction"):
        if file_sha256(Path(parent["path"]) / "manifest.json") != parent["manifest_sha256"]:
            raise ValueError("weak parent manifest changed")
        if parent["kind"] == "weak_training":
            parent_data(Path(parent["path"]))
        else:
            verify_reconstruction(Path(parent["path"]))
    elif file_sha256(parent["path"]) != parent["input_sha256"]:
        raise ValueError("natural response input changed")
    entries = read(output / "entries.json")
    if complete:
        manifest = read(output / "manifest.json")
        if (manifest["status"] != "complete" or manifest["prepare_manifest_sha256"] != file_sha256(output / "prepare_manifest.json")
                or manifest["settings_sha256"] != prepared["settings_sha256"]):
            raise ValueError("feature capture incomplete or unbound")
        expected = {"summary.json"} | {f"features/{e['id']}{suffix}" for e in entries
            if e["status"] == "available" for suffix in (".npz", ".json")}
        if set(manifest["artifacts"]) != expected:
            raise ValueError("complete feature artifacts omit or add entry coordinates")
        for name, sha in manifest["artifacts"].items():
            if file_sha256(output / name) != sha:
                raise ValueError("captured feature artifact changed")
        summary = read(output / "summary.json")
        if (summary["examples"] != len(entries) or summary["counts"] != dict(Counter(e["status"] for e in entries))
                or summary["tokens"] != sum(e["tokens"] for e in entries)
                or summary["actual_forwards"] != sum(e["status"] == "available" for e in entries)):
            raise ValueError("feature completion census differs from all prepared entries")
    return settings, entries


def model_identity(settings):
    return {"model_files": settings["model_files"],
        "tokenizer_files": [f for f in settings["model_files"] if "token" in f["name"] or f["name"] == "special_tokens_map.json"],
        "capture_code_sha256": settings["code_sha256"]["next_iteration/grounded_graph_features.py"]}


def load_example(output, entry, settings, tokenizer):
    """Caller first verifies the complete manifest once, then every row/array receipt."""
    packet = read(output / entry["packet_file"])
    inventory = read(Path(settings["inventory_path"]) / "sources" / f"{entry['source_id']}.json")["data"]
    validate_packet(packet, inventory, tokenizer)
    if packet["sha256"] != entry["packet_sha256"] or packet["source_id"] != entry["source_id"]:
        raise ValueError("feature packet row identity mismatch")
    receipt = read(output / "features" / f"{entry['id']}.json")
    if (receipt["sha256"] != digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt["packet_sha256"] != packet["sha256"] or receipt["model_identity"] != model_identity(settings)
            or receipt["representation"] != REPRESENTATION):
        raise ValueError("captured feature receipt identity mismatch")
    with np.load(output / "features" / f"{entry['id']}.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"source", "query"}:
            raise ValueError("feature array schema mismatch")
        arrays = {k: stored[k] for k in stored.files}
    for name, array in arrays.items():
        if (array.dtype != np.float32 or list(array.shape) != receipt[name + "_array_shape"]
                or hashlib.sha256(array.tobytes()).hexdigest() != receipt[name + "_array_sha256"]
                or not np.isfinite(array).all()):
            raise ValueError("captured feature array bytes/shape/dtype differ")
    if arrays["source"].shape[0] != len(packet["nodes"]) or arrays["query"].shape[0] != len(packet["target_token_ids"]):
        raise ValueError("source/query rows do not match complete node/token coordinates")
    return packet, arrays, receipt


def encode(args):
    settings, entries = verify(args.output)
    if (args.output / "features").exists() or (args.output / "manifest.json").exists():
        raise FileExistsError("fresh feature execution required; preserve partial arrays")
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("observer/tokenizer identity changed")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive GPU with20GiB free required")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval().requires_grad_(False)
    (args.output / "features").mkdir()
    artifacts, calls, started = {}, 0, time.monotonic()
    for index, entry in enumerate(entries):
        if entry["status"] == "available":
            packet = read(args.output / entry["packet_file"])
            inventory = read(Path(settings["inventory_path"]) / "sources" / f"{entry['source_id']}.json")["data"]
            arrays, receipt = capture(model, tokenizer, inventory, packet, model_identity(settings))
            name = f"features/{entry['id']}"
            with (args.output / (name + ".npz")).open("xb") as stream:
                np.savez(stream, **arrays)
            write_json_once(args.output / (name + ".json"), receipt)
            for suffix in (".npz", ".json"):
                artifacts[name + suffix] = file_sha256(args.output / (name + suffix))
            load_example(args.output, entry, settings, tokenizer)
            calls += receipt["actual_observer_forwards"]
        if (index + 1) % 8 == 0 or index + 1 == len(entries):
            progress(args.output, status="running", done=index + 1, total=len(entries), actual_forwards=calls,
                seconds=time.monotonic() - started)
    verify(args.output)
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("observer files changed during capture")
    summary = {"status": "complete", "examples": len(entries), "actual_forwards": calls,
        "counts": dict(Counter(e["status"] for e in entries)), "tokens": sum(e["tokens"] for e in entries),
        "seconds": time.monotonic() - started, "cuda_peak_bytes": torch.cuda.max_memory_allocated(), "labels_read": False}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "prepare_manifest_sha256": file_sha256(args.output / "prepare_manifest.json"),
        "settings_sha256": file_sha256(args.output / "settings.json")})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_cli = sub.add_parser("prepare")
    parent_cli = prepare_cli.add_mutually_exclusive_group(required=True)
    parent_cli.add_argument("--data", type=Path)
    parent_cli.add_argument("--reconstruction", type=Path)
    parent_cli.add_argument("--inputs", type=Path)
    prepare_cli.add_argument("--inventory", type=Path)
    prepare_cli.add_argument("--model", type=Path, required=True)
    prepare_cli.add_argument("--output", type=Path, required=True)
    sub.add_parser("encode").add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            encode(args)
