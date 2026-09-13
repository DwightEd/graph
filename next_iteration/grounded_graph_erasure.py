"""Re-encode erased source payloads to test adapter dependence on copied evidence.

Graph-only erasure holds original pre-token queries fixed. Full erasure also
replaces the query states. Neither is a claim about the original model's native
routing or semantic necessity when other source occurrences remain available.
"""

import argparse
import fcntl
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_feature_runner import load_example, verify
from next_iteration.grounded_graph_predict import load_adapter, trained
from next_iteration.grounded_graph_synthesize import progress
from next_iteration.grounded_graph_train import batch, token_statistics
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "grounded-graph-source-payload-erasure@1", "split": "source-validation only; all48sources",
    "erase": "all prompt token positions belonging to observed coordinate target owners",
    "replacement": "single token encoding of one ASCII space, same number of positions",
    "branches": ["original", "graph_source_erased", "full_source_erased"],
    "query_policy": "graph_source_erased keeps original query; full_source_erased uses actual perturbed pretoken query",
    "score_positions": "all source-coordinate anchor first tokens, fixed before outputs",
    "labels_read": False, "semantic_necessity_claim": False, "native_route_claim": False,
    "redundant_evidence": "unselected occurrences can retain same value; surviving gain does not alone prove formatting shortcut"}


@torch.no_grad()
def erase_capture(model, tokenizer, packet):
    replacement = tokenizer.encode(" ", add_special_tokens=False)
    if len(replacement) != 1:
        raise ValueError("space replacement must be exactly one observer token")
    owners = sorted({t["owner_node_index"] for t in packet["weak_targets"]})
    positions = sorted({i for n in owners for i in packet["nodes"][n]["prompt_token_indices"]})
    plen = packet["prompt_record"]["prompt_length"]
    if not positions or any(not 0 < i < plen or packet["source_offsets"][i][0] >= packet["source_offsets"][i][1] for i in positions):
        raise ValueError("erasure escapes mapped prompt source payload positions")
    ids = list(packet["input_ids"])
    for position in positions:
        ids[position] = replacement[0]
    if ids[plen:] != packet["input_ids"][plen:]:
        raise ValueError("erasure modified observed response prefix")
    seen = []
    hook = model.model.norm.register_forward_hook(lambda module, inputs, output: seen.append(output))
    try:
        states = model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False,
            return_dict=True).last_hidden_state
        if len(seen) != 1 or states.data_ptr() != seen[0].data_ptr():
            raise ValueError("erasure states are not actual final-norm output")
        h = states[0].float()
        weights = torch.zeros((len(packet["nodes"]), plen), device=h.device)
        for index, node in enumerate(packet["nodes"]):
            keys = node["prompt_token_indices"]
            if keys:
                weights[index, keys] = 1. / len(keys)
        arrays = {"source": (weights @ h[:plen]).cpu().numpy(), "query": h[packet["query_positions"]].cpu().numpy()}
    finally:
        hook.remove()
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("erased native states nonfinite")
    receipt = {"packet_sha256": packet["sha256"], "replacement_token_id": replacement[0], "source_owner_indices": owners,
        "erased_prompt_token_indices": positions, "executed_input_ids": ids, "executed_input_sha256": digest(ids),
        "original_input_sha256": digest(packet["input_ids"]), "actual_observer_forwards": 1,
        "coordinates": "original source/token graph coordinates held fixed under explicit input-token intervention",
        "actual_final_norm_hook_count": len(seen), "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
            "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
    return arrays, receipt


@torch.no_grad()
def measure(adapters, head, packet, arrays):
    arguments, targets, positives, mask = batch([(packet, arrays)], head.device)
    base, _ = token_statistics(arguments["query_states"][mask], targets[mask], head)
    rows = {"query_indices": mask.nonzero().flatten().cpu().tolist(), "base_logp": base.cpu().tolist()}
    for arm, adapter in adapters.items():
        result = adapter(**arguments, use_edges=arm == "graph")
        logp, _ = token_statistics(result["hidden"][mask], targets[mask], head)
        probability = (result["pointer"][mask] * positives[mask]).sum(-1)
        rows[arm] = {"adapter_logp": logp.cpu().tolist(), "adapter_logp_gain": (logp - base).cpu().tolist(),
            "coordinate_pointer_probability": probability.cpu().tolist()}
    return rows


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh source erasure output required")
    train_settings, _, trained_features = trained(args.training)
    settings, entries = verify(args.features, complete=True)
    if (settings["parent"]["kind"] != "source_reconstruction" or settings["model_files"] != trained_features["model_files"]
            or file_sha256(args.features / "manifest.json") != train_settings["feature_manifest_sha256"]):
        raise ValueError("source erasure requires actual training source feature parent")
    selected = [e for e in entries if e["split"] == "validation"]
    if len(selected) != 48:
        raise ValueError("complete48source-validation erasure denominator required")
    code = {**train_settings["code_sha256"], "next_iteration/grounded_graph_predict.py": file_sha256(Path(__file__).with_name("grounded_graph_predict.py")),
        "next_iteration/grounded_graph_erasure.py": file_sha256(Path(__file__))}
    for name, sha in code.items():
        path = args.output / "executed_code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        if file_sha256(path) != sha:
            raise ValueError("erasure code changed while snapshotting")
    frozen = {"protocol": PROTOCOL, "code_sha256": code, "feature_path": str(args.features.resolve()),
        "feature_manifest_sha256": file_sha256(args.features / "manifest.json"), "training_path": str(args.training.resolve()),
        "training_manifest_sha256": file_sha256(args.training / "manifest.json"), "source_ids": [e["source_id"] for e in selected],
        "model_files": settings["model_files"], "labels_read": False}
    write_json_once(args.output / "settings.json", frozen)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive GPU required for actual source re-encoding")
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("erasure observer/tokenizer identity changed")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa").eval().requires_grad_(False)
    adapters = {arm: load_adapter(args.training, arm, model.device) for arm in ("graph", "no_edges")}
    artifacts, records, started, calls = {}, [], time.monotonic(), 0
    for entry in selected:
        if entry["status"] != "available":
            record = {"source_id": entry["source_id"], "task": entry["task"], "status": entry["status"]}
        else:
            packet, arrays, original_receipt = load_example(args.features, entry, settings, tokenizer)
            erased, receipt = erase_capture(model, tokenizer, packet)
            graph_erased = {"source": erased["source"], "query": arrays["query"]}
            values = {name: measure(adapters, model.lm_head.weight, packet, branch) for name, branch in
                (("original", arrays), ("graph_source_erased", graph_erased), ("full_source_erased", erased))}
            record = {"source_id": entry["source_id"], "task": entry["task"], "status": "complete",
                "original_receipt_sha256": original_receipt["sha256"], "erasure_receipt": receipt, "branches": values}
            name = f"sources/{entry['source_id']}.npz"
            (args.output / "sources").mkdir(exist_ok=True)
            with (args.output / name).open("xb") as stream:
                np.savez(stream, **erased)
            artifacts[name] = file_sha256(args.output / name)
            calls += receipt["actual_observer_forwards"]
        name = f"sources/{entry['source_id']}.json"
        write_json_once(args.output / name, record)
        artifacts[name] = file_sha256(args.output / name)
        records.append(record)
        progress(args.output, status="running", sources=len(records), total_sources=len(selected), actual_observer_forwards=calls,
            seconds=time.monotonic() - started)
    aggregate = {}
    for branch in ("graph_source_erased", "full_source_erased"):
        aggregate[branch] = {}
        for arm in ("graph", "no_edges"):
            gain_drops, probability_drops = [], []
            for record in records:
                if record["status"] != "complete":
                    continue
                original, erased = record["branches"]["original"][arm], record["branches"][branch][arm]
                gain_drops.append(np.mean(original["adapter_logp_gain"]) - np.mean(erased["adapter_logp_gain"]))
                probability_drops.append(np.mean(original["coordinate_pointer_probability"]) - np.mean(erased["coordinate_pointer_probability"]))
            aggregate[branch][arm] = {"sources": len(gain_drops),
                "source_mean_adapter_gain_drop": float(np.mean(gain_drops)) if gain_drops else None,
                "source_mean_coordinate_probability_drop": float(np.mean(probability_drops)) if probability_drops else None}
    verify(args.features, complete=True)
    trained(args.training)
    for name, sha in code.items():
        if file_sha256(ROOT / name) != sha or file_sha256(args.output / "executed_code" / name) != sha:
            raise ValueError("erasure code changed during execution")
    summary = {"status": "complete", "sources": len(selected), "actual_observer_forwards": calls,
        "aggregate": aggregate, "seconds": time.monotonic() - started, "labels_read": False,
        "semantic_necessity_claim": False, "native_route_claim": False}
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
