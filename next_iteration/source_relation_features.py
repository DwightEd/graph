"""Frozen auxiliary text vectors for source pointer reconstruction.

Prepare tokenizes every exact view on CPU. Encode never truncates; documents
outside the limit stay in the inventory with an explicit unavailable status.
No original generation-state or semantic-ground-truth claim is made here.
"""

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.surface_graph import _check
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest
from route_graph.frozen_reader import digest, write_json_once

GRAPH = Path(__file__).resolve().parents[1]
PROTOCOL = {"schema": "source-rel-mini-features@1", "max_tokens": 4096,
    "representation": "auxiliary canonical-view causal final-layer last-nonpad-token",
    "dtype": "bfloat16", "array_dtype": "float32", "attention": "sdpa", "batch_size": 4,
    "seed": 20260913, "truncation": False, "labels_read": False,
    "over_limit": "retain inventory; mark unavailable; exclude whole affected query pool"}


def read_json(path):
    return json.loads(path.read_text())


def verify_data(path):
    manifest = read_json(path / "manifest.json")
    if manifest["status"] != "complete" or file_sha256(path / "settings.json") != manifest["settings_file_sha256"]:
        raise ValueError("source data is incomplete or settings changed")
    settings = read_json(path / "settings.json")
    if settings["labels_read"] or file_sha256(settings["input_path"]) != settings["input_sha256"]:
        raise ValueError("source data input/label boundary changed")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("source data artifact changed")
    verify_code(path, settings)
    records = []
    for sid in settings["sources"]:
        value = read_json(path / "sources" / f"{sid}.json")
        if value["settings_object_digest"] != digest(settings):
            raise ValueError("source data settings binding mismatch")
        _check(value["data"])
        records.append(value["data"])
    return records


def freeze_code(output, paths):
    hashes = {}
    for path in sorted(set(paths)):
        name = str(path.relative_to(GRAPH))
        hashes[name] = file_sha256(path)
        destination = output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("code snapshot already exists")
        shutil.copyfile(path, destination)
        if file_sha256(destination) != hashes[name]:
            raise ValueError("code changed during snapshot")
    return hashes


def verify_code(output, settings):
    for name, sha in settings["code_sha256"].items():
        if file_sha256(GRAPH / name) != sha or file_sha256(output / "executed_code" / name) != sha:
            raise ValueError("live/snapshotted code changed; preserve run and use a fresh revision")


def progress(output, **values):
    record = {"pid": os.getpid(), "unix_time": time.time(), **values}
    temporary = output / "progress.partial.json"
    temporary.write_text(json.dumps(record, allow_nan=False) + "\n")
    os.replace(temporary, output / "progress.json")
    print(json.dumps(record, allow_nan=False), flush=True)


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh feature output required")
    records = [r for r in verify_data(args.data) if r["status"] == "compiled"]
    if args.sanity:
        # One source from each internal split, fixed before features or scores.
        records = [next(r for r in records if r["split"] == split) for split in ("source_train", "source_validation")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    docs = {}
    bindings = []
    for record in records:
        b = {"source_id": record["source_id"], "split": record["split"], "queries": [], "candidates": []}
        for kind in ("queries", "candidates"):
            for item in record[kind]:
                did = digest(item["text"])
                if did not in docs:
                    ids = tokenizer.encode(item["text"], add_special_tokens=True)
                    docs[did] = {"id": did, "text": item["text"], "input_ids": ids,
                        "token_count": len(ids), "status": "encodable" if 0 < len(ids) <= PROTOCOL["max_tokens"] else "unavailable_length"}
                b[kind].append({**item, "document_id": did})
        bindings.append(b)
    documents = sorted(docs.values(), key=lambda d: d["id"])
    return prepare_inventory(args.output, args.model, documents, bindings,
        {"kind": "source_reconstruction", "data_path": str(args.data.resolve()),
         "data_manifest_sha256": file_sha256(args.data / "manifest.json"), "sanity_only": args.sanity},
        extra_code=[Path(__file__).with_name("source_relation_data.py")])


def prepare_inventory(output, model_path, documents, bindings, parent, extra_code=()):
    """Freeze a complete, already tokenized inventory for the shared encoder."""
    if output.exists() or not documents or len({d["id"] for d in documents}) != len(documents):
        raise ValueError("fresh output and nonempty unique document inventory required")
    sizes = [d["token_count"] for d in documents]
    for doc in documents:
        expected_status = "encodable" if 0 < len(doc["input_ids"]) <= PROTOCOL["max_tokens"] else "unavailable_length"
        if (doc["id"] != digest(doc["text"]) or doc["token_count"] != len(doc["input_ids"])
                or doc["status"] != expected_status):
            raise ValueError("view text/token binding mismatch")
    bindings = json.loads(json.dumps(bindings))
    availability = {d["id"]: d["status"] for d in documents}
    for binding in bindings:
        candidates = {c["id"]: c for c in binding["candidates"]}
        for query in binding["queries"]:
            references = [query["document_id"]] + [candidates[cid]["document_id"] for cid in query["candidate_ids"]]
            query["feature_status"] = "encodable" if all(availability[did] == "encodable" for did in references) else "unavailable_length"
    paths = list((GRAPH / "route_graph").glob("*.py")) + [Path(__file__).with_name(n) for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "source_relation_features.py")]
    paths.extend(extra_code)
    code = freeze_code(output, paths)
    settings = {"protocol": PROTOCOL, "parent": parent, "model_path": str(model_path.resolve()),
        "model_files": model_manifest(model_path), "code_sha256": code,
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "inventory_digest": digest(documents), "bindings_digest": digest(bindings),
        "length_preflight": {"documents": len(sizes), "total_tokens": sum(sizes), "max_tokens": max(sizes),
            "unavailable_documents": sum(d["status"] != "encodable" for d in documents)},
        "labels_read": False}
    write_json_once(output / "settings.json", settings)
    write_json_once(output / "documents.json", documents)
    write_json_once(output / "bindings.json", bindings)
    verify_code(output, settings)
    write_json_once(output / "prepare_manifest.json", {"status": "prepared", "settings_file_sha256": file_sha256(output / "settings.json"),
        "artifacts": {n: file_sha256(output / n) for n in ("documents.json", "bindings.json")}, "model_forwards": 0})
    print(json.dumps(settings["length_preflight"]), flush=True)
    return settings


def verify_prepared(output, *, verify_model=False):
    manifest = read_json(output / "prepare_manifest.json")
    settings = read_json(output / "settings.json")
    if (manifest["status"] != "prepared" or file_sha256(output / "settings.json") != manifest["settings_file_sha256"]
            or settings["protocol"] != PROTOCOL or settings["torch"] != str(torch.__version__)
            or settings["transformers"] != transformers.__version__):
        raise ValueError("prepared feature settings/runtime changed")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("prepared feature inventory changed")
    verify_code(output, settings)
    if verify_model and model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("model/tokenizer files changed")
    docs, bindings = read_json(output / "documents.json"), read_json(output / "bindings.json")
    if digest(docs) != settings["inventory_digest"] or digest(bindings) != settings["bindings_digest"]:
        raise ValueError("inventory/settings digest mismatch")
    parent = settings["parent"]
    if parent["kind"] == "source_reconstruction":
        if file_sha256(Path(parent["data_path"]) / "manifest.json") != parent["data_manifest_sha256"]:
            raise ValueError("parent data manifest changed")
        verify_data(Path(parent["data_path"]))
    return settings, docs, bindings


def encode(args):
    settings, docs, _ = verify_prepared(args.output, verify_model=True)
    if any((args.output / name).exists() for name in ("features.partial.npy", "features.npy", "manifest.json")):
        raise FileExistsError("feature execution already started; no overwrite/retry")
    locks = [GRAPH.parent / "reanchor/runs/relation_interleave_20260913.lock", args.output / ".encode.lock"]
    streams = []
    try:
        for path in locks:
            stream = path.open("a")
            streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise RuntimeError("exclusive free GPU required")
        tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
        for doc in docs:
            if tokenizer.encode(doc["text"], add_special_tokens=True) != doc["input_ids"]:
                raise ValueError("actual tokenizer IDs differ from full CPU preflight")
        torch.manual_seed(PROTOCOL["seed"])
        progress(args.output, status="loading_model", documents=len(docs))
        model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        model.requires_grad_(False)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        hidden = model.config.hidden_size
        matrix = np.lib.format.open_memmap(args.output / "features.partial.npy", mode="w+", dtype=np.float32,
            shape=(len(docs), hidden))
        matrix[:] = np.nan
        eligible = sorted((i for i, d in enumerate(docs) if d["status"] == "encodable"),
            key=lambda i: (docs[i]["token_count"], docs[i]["id"]))
        start, forwards, total_tokens = time.monotonic(), 0, 0
        for begin in range(0, len(eligible), PROTOCOL["batch_size"]):
            indices = eligible[begin:begin + PROTOCOL["batch_size"]]
            ids = [docs[i]["input_ids"] for i in indices]
            width = max(map(len, ids))
            x = torch.tensor([v + [pad_id] * (width - len(v)) for v in ids], device="cuda", dtype=torch.long)
            mask = torch.tensor([[1] * len(v) + [0] * (width - len(v)) for v in ids], device="cuda", dtype=torch.long)
            with torch.inference_mode():
                # Base model avoids unnecessary vocabulary-sized logits.
                states = model.model(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
                vectors = states[torch.arange(len(ids), device="cuda"), mask.sum(-1) - 1].float().cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError("nonfinite encoded features")
            matrix[indices] = vectors
            del states, vectors, x, mask
            forwards += 1
            total_tokens += sum(map(len, ids))
            if forwards % 50 == 0 or begin + len(indices) == len(eligible):
                matrix.flush()
                progress(args.output, status="encoding", completed=begin + len(indices), total=len(eligible),
                    forwards=forwards, seconds=time.monotonic() - start, encoded_tokens=total_tokens)
        matrix.flush()
        del matrix, model
        torch.cuda.empty_cache()
        verify_prepared(args.output, verify_model=True)
        os.replace(args.output / "features.partial.npy", args.output / "features.npy")
        result = {"status": "complete", "feature_forwards": forwards, "encoded_documents": len(eligible),
            "unavailable_documents": len(docs) - len(eligible), "encoded_tokens": total_tokens,
            "seconds_excluding_model_load": time.monotonic() - start, "hidden_size": hidden,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "labels_read": False,
            "representation": PROTOCOL["representation"]}
        write_json_once(args.output / "summary.json", result)
        write_json_once(args.output / "manifest.json", {"status": "complete",
            "prepare_manifest_sha256": file_sha256(args.output / "prepare_manifest.json"),
            "artifacts": {n: file_sha256(args.output / n) for n in ("features.npy", "summary.json")}})
        progress(args.output, **result)
    finally:
        for stream in reversed(streams):
            stream.close()


def load_features(output):
    settings, docs, bindings = verify_prepared(output)
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "complete" or file_sha256(output / "prepare_manifest.json") != manifest["prepare_manifest_sha256"]:
        raise ValueError("features incomplete or prepare manifest changed")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("encoded artifact changed")
    matrix = np.load(output / "features.npy", mmap_mode="r", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (len(docs), read_json(output / "summary.json")["hidden_size"]):
        raise ValueError("unexpected feature shape/dtype")
    for i, doc in enumerate(docs):
        if (doc["status"] == "encodable" and not np.isfinite(matrix[i]).all()) or (doc["status"] != "encodable" and not np.isnan(matrix[i]).all()):
            raise ValueError("feature availability mismatch")
    return settings, docs, bindings, matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "encode"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.data is None or args.model is None:
            parser.error("prepare requires --data and --model")
        prepare(args)
    else:
        encode(args)
