"""Held-out anchor restoration and fixed-query source-content dependence.

Reuse the actual v1 source-payload erasure encodings, with exact packet, observer,
node-order and original receipt checks. Both arms keep the new H_empty fixed.
"""

import argparse
import fcntl
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_erasure import PROTOCOL as ERASURE_PROTOCOL
from next_iteration.grounded_graph_erasure import measure
from next_iteration.grounded_graph_feature_runner import read
from next_iteration.grounded_graph_restoration_features import load_example, verify
from next_iteration.grounded_graph_restoration_predict import load_adapter, trained
from next_iteration.grounded_graph_synthesize import progress
from next_iteration.grounded_graph_train import frozen_head, token_statistics
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import write_json_once

PROTOCOL = {"schema": "grounded-graph-restoration-dependence@2", "sources": "all48source-validation",
    "branches": ["original_source_empty_query", "payload_erased_source_empty_query"],
    "query": "same actual all-source-erased H_empty in both branches",
    "erasure_cache": "v1 actual selected-owner-payload erasure, not all-source-erasure and not zero-vector masking",
    "anchors": "all predefined source-coordinate pointer anchor first tokens",
    "gate": "positive source-mean anchor adapter gain over empty LM AND positive source-mean fixed-query source-payload erasure gain drop",
    "labels_read": False, "native_route_claim": False, "semantic_necessity_claim": False}


def erasure_parent(path, features, entries):
    settings, manifest = read(path / "settings.json"), read(path / "manifest.json")
    selected = [e for e in entries if e["split"] == "validation"]
    expected_ids = [e["source_id"] for e in selected]
    if (settings["protocol"] != ERASURE_PROTOCOL or manifest["status"] != "complete"
            or file_sha256(path / "settings.json") != manifest["settings_sha256"]
            or settings["source_ids"] != expected_ids or len(expected_ids) != 48 or len(set(expected_ids)) != 48
            or settings["model_files"] != features["model_files"]
            or settings["feature_path"] != features["original_feature_path"]
            or settings["feature_manifest_sha256"] != features["original_feature_manifest_sha256"]):
        raise ValueError("cached erasure differs from original source features/observer/heldout roster")
    expected = {"summary.json"} | {f"sources/{sid}{suffix}" for sid in expected_ids for suffix in (".json", ".npz")}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("cached erasure incomplete artifact census")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("cached erasure artifact changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(path / "executed_code" / name) != sha:
            raise ValueError("cached erasure code provenance changed")
    summary = read(path / "summary.json")
    if summary["sources"] != 48 or summary["actual_observer_forwards"] != 48:
        raise ValueError("cached erasure was not48actual observer forwards")
    return selected


def erased_arrays(path, entry, packet, arrays, restoration_receipt):
    record = read(path / "sources" / f"{entry['source_id']}.json")
    receipt = record["erasure_receipt"]
    owners = sorted({t["owner_node_index"] for t in packet["weak_targets"]})
    positions = sorted({i for n in owners for i in packet["nodes"][n]["prompt_token_indices"]})
    executed = list(packet["input_ids"])
    for i in positions:
        executed[i] = receipt["replacement_token_id"]
    if (record["status"] != "complete" or record["source_id"] != entry["source_id"] or record["task"] != entry["task"]
            or receipt["packet_sha256"] != packet["sha256"]
            or record["original_receipt_sha256"] != restoration_receipt["original_receipt_sha256"]
            or receipt["source_owner_indices"] != owners or receipt["erased_prompt_token_indices"] != positions
            or receipt["executed_input_ids"] != executed or receipt["actual_observer_forwards"] != 1
            or receipt["replacement_token_id"] != restoration_receipt["replacement_token_id"]):
        raise ValueError("erasure array packet/node/anchor order or executed input mismatch")
    with np.load(path / "sources" / f"{entry['source_id']}.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"source", "query"}:
            raise ValueError("erasure array schema differs")
        actual = {k: saved[k] for k in saved.files}
    for name, array in actual.items():
        expected = receipt["arrays"][name]
        if (array.shape != arrays[name].shape or list(array.shape) != expected["shape"] or array.dtype != np.float32
                or expected["dtype"] != "float32" or hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]
                or not np.isfinite(array).all()):
            raise ValueError("erasure actual array bytes/shape/dtype mismatch")
    return {"source": actual["source"], "query": arrays["query"]}


def aggregate(records):
    result = {}
    for arm in ("graph", "no_edges"):
        gain, erased, pointer = [], [], []
        for record in records:
            original = record["branches"]["original_source_empty_query"][arm]
            changed = record["branches"]["payload_erased_source_empty_query"][arm]
            gain.append(float(np.mean(original["adapter_logp_gain"])))
            erased.append(float(np.mean(changed["adapter_logp_gain"])))
            pointer.append(float(np.mean(original["coordinate_pointer_probability"]) - np.mean(changed["coordinate_pointer_probability"])))
        drop = np.asarray(gain) - np.asarray(erased)
        result[arm] = {"sources": len(records), "source_mean_anchor_gain": float(np.mean(gain)),
            "source_mean_erased_anchor_gain": float(np.mean(erased)), "source_mean_anchor_gain_drop": float(drop.mean()),
            "source_mean_coordinate_probability_drop": float(np.mean(pointer)),
            "positive_gain_sources": sum(x > 0 for x in gain), "positive_drop_sources": int((drop > 0).sum()),
            "mechanism_gate_passed": bool(np.mean(gain) > 0 and drop.mean() > 0)}
    return result


def validate_branch(branch):
    count = len(branch["query_indices"])
    values = [branch["base_logp"]]
    for arm in ("graph", "no_edges"):
        row = branch[arm]
        values.extend(row[key] for key in ("adapter_logp", "adapter_logp_gain", "coordinate_pointer_probability"))
        probability = np.asarray(row["coordinate_pointer_probability"])
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("coordinate pointer probability escapes zero to one")
    if count == 0 or any(np.asarray(v).shape != (count,) or not np.isfinite(v).all() for v in values):
        raise ValueError("restoration anchor measurements nonfinite or incomplete")


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh restoration dependence output required")
    train_settings, _, _ = trained(args.training)
    features = Path(train_settings["feature_path"])
    settings, entries = verify(features)
    selected = erasure_parent(args.erasure, settings, entries)
    code = {**train_settings["code_sha256"], **{f"next_iteration/{n}": file_sha256(ROOT / "next_iteration" / n)
        for n in ("grounded_graph_restoration_dependence.py", "grounded_graph_restoration_predict.py", "grounded_graph_erasure.py", "grounded_graph_predict.py")}}
    for name, sha in code.items():
        target = args.output / "executed_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_sha256(target) != sha:
            raise ValueError("dependence code changed during snapshot")
    frozen = {"protocol": PROTOCOL, "code_sha256": code, "training_path": str(args.training.resolve()),
        "training_manifest_sha256": file_sha256(args.training / "manifest.json"), "erasure_path": str(args.erasure.resolve()),
        "erasure_manifest_sha256": file_sha256(args.erasure / "manifest.json"), "labels_read": False}
    write_json_once(args.output / "settings.json", frozen)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("free GPU required for frozen-head dependence scoring")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    head = frozen_head(settings, "cuda:0")
    adapters = {arm: load_adapter(args.training, arm, head.device) for arm in ("graph", "no_edges")}
    records, artifacts, started = [], {}, time.monotonic()
    for entry in selected:
        packet, arrays, receipt = load_example(features, entry, settings, tokenizer)
        erased = erased_arrays(args.erasure, entry, packet, arrays, receipt)
        branches = {key: measure(adapters, head, packet, value) for key, value in (
            ("original_source_empty_query", arrays), ("payload_erased_source_empty_query", erased))}
        for branch in branches.values():
            validate_branch(branch)
        original, changed = branches.values()
        if original["query_indices"] != changed["query_indices"] or original["base_logp"] != changed["base_logp"]:
            raise ValueError("query or anchor baseline changed during graph-only intervention")
        indices = original["query_indices"]
        full_logp, _ = token_statistics(torch.from_numpy(arrays["full_query"][indices]).to(head.device),
            torch.tensor(packet["target_token_ids"], device=head.device)[indices], head)
        if not torch.isfinite(full_logp).all():
            raise ValueError("full-source anchor baseline nonfinite")
        record = {"source_id": entry["source_id"], "task": entry["task"], "anchors": len(indices),
            "packet_sha256": packet["sha256"], "restoration_receipt_sha256": receipt["sha256"],
            "full_source_base_logp": full_logp.cpu().tolist(), "branches": branches}
        name = f"sources/{entry['source_id']}.json"
        write_json_once(args.output / name, record)
        artifacts[name] = file_sha256(args.output / name)
        records.append(record)
        progress(args.output, status="running", sources=len(records), total_sources=48, adapter_forwards=4 * len(records), observer_forwards=0)
    trained(args.training)
    erasure_parent(args.erasure, settings, entries)
    for name, sha in code.items():
        if file_sha256(ROOT / name) != sha or file_sha256(args.output / "executed_code" / name) != sha:
            raise ValueError("dependence live/snapshot code changed")
    summary = {"status": "complete", "sources": len(records), "anchors": sum(r["anchors"] for r in records),
        "aggregate": aggregate(records), "by_task": {task: aggregate([r for r in records if r["task"] == task])
            for task in sorted({r["task"] for r in records})}, "observer_forwards": 0, "adapter_forwards": 4 * len(records),
        "seconds": time.monotonic() - started, "labels_read": False, "native_route_claim": False}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json")})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--erasure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
