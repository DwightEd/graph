"""Freeze label-free full-token/word predictions from jointly trained adapters."""

import argparse
import fcntl
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_feature_runner import load_example, read, verify
from next_iteration.grounded_graph_features import EDGE_KINDS, NODE_KINDS
from next_iteration.grounded_graph_synthesize import progress
from next_iteration.grounded_graph_train import PROTOCOL as TRAIN_PROTOCOL
from next_iteration.grounded_graph_train import (
    batch,
    frozen_head,
    new_adapter,
    token_statistics,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "grounded-graph-natural-prediction@1", "labels_read": False,
    "score": "logp_base - logp_adapter; larger is risk; no direction fitting",
    "scores": ["base_nll", "base_entropy", "graph_nll", "graph_difference", "no_edges_nll", "no_edges_difference", "permuted_graph_difference"],
    "word_units": "all non-whitespace spans", "word_aggregation": "token score weighted by character intersection",
    "missing": "explicit unavailable status; fixed zero neutral difference, no dropping from denominator",
    "topology_control": "fixed permutation of edge destinations within one graph, node features unchanged",
    "native_route_claim": False}


def trained(path):
    settings, manifest, summary = read(path / "settings.json"), read(path / "manifest.json"), read(path / "summary.json")
    if (settings["protocol"] != TRAIN_PROTOCOL or manifest["status"] != "complete"
            or file_sha256(path / "settings.json") != manifest["settings_sha256"]):
        raise ValueError("trained adapter settings/protocol changed")
    expected = {"summary.json"} | {f"{arm}/initial{suffix}" for arm in TRAIN_PROTOCOL["arms"] for suffix in (".json", ".pt")} | {
        f"{arm}/epoch_{epoch:02d}{suffix}" for arm in TRAIN_PROTOCOL["arms"]
        for epoch in range(1, TRAIN_PROTOCOL["epochs"] + 1) for suffix in (".pt", ".json")}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("trained adapter epoch/arm artifact census differs")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("trained adapter artifact changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(path / "executed_code" / name) != sha:
            raise ValueError("trained adapter live/snapshot code changed")
    features = Path(settings["feature_path"])
    if file_sha256(features / "manifest.json") != settings["feature_manifest_sha256"]:
        raise ValueError("training feature parent changed")
    feature_settings, _ = verify(features, complete=True)
    return settings, summary, feature_settings


def load_adapter(path, arm, device):
    _, summary, _ = trained(path)
    selected = summary["arms"][arm]["selected"]
    if file_sha256(path / selected["checkpoint"]) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    ckpt = torch.load(path / selected["checkpoint"], map_location="cpu", weights_only=True)
    if (ckpt["arm"] != arm or ckpt["epoch"] != selected["epoch"] or ckpt["node_kinds"] != NODE_KINDS
            or ckpt["edge_kinds"] != EDGE_KINDS or ckpt["settings_sha256"] != file_sha256(path / "settings.json")):
        raise ValueError("checkpoint arm/epoch/vocabulary/training settings mismatch")
    adapter = new_adapter(device).eval().requires_grad_(False)
    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    return adapter


def words(packet, scores, status):
    offsets = np.asarray(packet["response_offsets"], dtype=np.int64)
    result = []
    for word in re.finditer(r"\S+", packet["response_text"]):
        overlap = np.maximum(0, np.minimum(offsets[:, 1], word.end()) - np.maximum(offsets[:, 0], word.start()))
        usable = status == "available" and overlap.sum() > 0
        result.append({"span": list(word.span()), "text": word.group(), "status": "available" if usable else "unavailable",
            "scores": {k: float(np.dot(v, overlap) / overlap.sum()) if usable else 0. for k, v in scores.items()}})
    return result


def permuted_arguments(arguments, seed):
    copy = dict(arguments)
    edges = arguments["edge_index"].clone()
    generator = torch.Generator(device=edges.device).manual_seed(seed)
    perm = torch.randperm(len(arguments["node_types"]), generator=generator, device=edges.device)
    edges[1] = perm[edges[1]]
    copy["edge_index"] = edges
    return copy


@torch.no_grad()
def score_one(adapters, head, packet, arrays, seed):
    arguments, targets, _, _ = batch([(packet, arrays)], head.device)
    base_logp, base_entropy = token_statistics(arguments["query_states"], targets, head)
    scores = {"base_nll": -base_logp.cpu().numpy(), "base_entropy": base_entropy.cpu().numpy()}
    traces = {}
    for arm in TRAIN_PROTOCOL["arms"]:
        result = adapters[arm](**arguments, use_edges=arm == "graph")
        logp, _ = token_statistics(result["hidden"], targets, head)
        scores[arm + "_nll"] = -logp.cpu().numpy()
        scores[arm + "_difference"] = (base_logp - logp).cpu().numpy()
        traces[arm + "_pointer"] = result["pointer"].cpu().numpy()
        traces[arm + "_gate"] = result["gate"].cpu().numpy()
        traces[arm + "_residual_norm"] = result["residual"].norm(dim=-1).cpu().numpy()
    result = adapters["graph"](**permuted_arguments(arguments, seed), use_edges=True)
    logp, _ = token_statistics(result["hidden"], targets, head)
    scores["permuted_graph_difference"] = (base_logp - logp).cpu().numpy()
    if set(scores) != set(PROTOCOL["scores"]) or any(v.shape != (len(targets),) or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("natural full-token score shape/finiteness mismatch")
    return scores, traces


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh natural prediction output required")
    train_settings, train_summary, trained_features = trained(args.training)
    settings, entries = verify(args.features, complete=True)
    if settings["parent"]["kind"] != "natural_response" or settings["model_files"] != trained_features["model_files"]:
        raise ValueError("natural inference observer differs from trained feature observer")
    # Freeze before loading inference models or producing any score.
    code = {**train_settings["code_sha256"], "next_iteration/grounded_graph_predict.py": file_sha256(Path(__file__))}
    for name, sha in code.items():
        destination = args.output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if file_sha256(destination) != sha:
            raise ValueError("prediction code changed during snapshot")
    prediction_settings = {"protocol": PROTOCOL, "code_sha256": code, "features_path": str(args.features.resolve()),
        "features_manifest_sha256": file_sha256(args.features / "manifest.json"), "training_path": str(args.training.resolve()),
        "training_manifest_sha256": file_sha256(args.training / "manifest.json"),
        "selected_checkpoints": {k: v["selected"] for k, v in train_summary["arms"].items()}, "labels_read": False}
    write_json_once(args.output / "settings.json", prediction_settings)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("free GPU required for frozen-head inference")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    head = frozen_head(settings, "cuda:0")
    adapters = {arm: load_adapter(args.training, arm, head.device) for arm in TRAIN_PROTOCOL["arms"]}
    artifacts, started, word_count, unavailable, calls = {}, time.monotonic(), 0, 0, 0
    train_source_sha = set(train_settings["source_text_sha256"]["train"])
    val_source_sha = set(train_settings["source_text_sha256"]["validation"])
    for entry in entries:
        packet = read(args.features / entry["packet_file"])
        if entry["status"] == "available":
            packet, arrays, _ = load_example(args.features, entry, settings, tokenizer)
            scores, traces = score_one(adapters, head, packet, arrays, int(digest([20260913, entry["id"]])[:8], 16))
            name = f"predictions/{entry['id']}.npz"
            (args.output / "predictions").mkdir(exist_ok=True)
            with (args.output / name).open("xb") as stream:
                np.savez(stream, **scores, **traces)
            artifacts[name] = file_sha256(args.output / name)
            calls += 3
        else:
            scores = {name: np.zeros(entry["tokens"], dtype=np.float32) for name in PROTOCOL["scores"]}
        word_rows = words(packet, scores, entry["status"])
        raw_sha = entry["source_text_sha256"]
        record = {"id": entry["id"], "source_id": entry["source_id"], "task": entry["task"], "generator": entry["generator"],
            "official_split": entry["split"], "response_text": packet["response_text"], "row_sha256": entry["row_sha256"],
            "packet_sha256": packet["sha256"], "entry_status": entry["status"], "words": word_rows,
            "source_training_overlap": "train" if raw_sha in train_source_sha else "validation" if raw_sha in val_source_sha else "unseen",
            "source_node_order": [n["id"] for n in packet["nodes"]], "response_offsets": packet["response_offsets"],
            "target_token_ids": packet["target_token_ids"], "labels_read": False, "native_route_claim": False}
        name = f"predictions/{entry['id']}.json"
        write_json_once(args.output / name, record)
        artifacts[name] = file_sha256(args.output / name)
        word_count += len(word_rows)
        unavailable += sum(w["status"] != "available" for w in word_rows)
        progress(args.output, status="running", responses=sum(n.endswith('.json') for n in artifacts), total=len(entries),
            words=word_count, unavailable_words=unavailable, adapter_forwards=calls, observer_forwards=0,
            seconds=time.monotonic() - started)
    verify(args.features, complete=True)
    trained(args.training)
    for name, sha in code.items():
        if file_sha256(ROOT / name) != sha or file_sha256(args.output / "executed_code" / name) != sha:
            raise ValueError("prediction live/snapshot code changed during inference")
    summary = {"status": "complete", "responses": len(entries), "words": word_count, "unavailable_words": unavailable,
        "adapter_forwards": calls, "observer_forwards": 0, "labels_read": False, "seconds": time.monotonic() - started}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json")})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
