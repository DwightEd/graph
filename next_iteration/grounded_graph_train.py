"""Fit graph/no-edge residual adapters from sealed source-only weak examples."""

import argparse
import fcntl
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoTokenizer

from next_iteration.grounded_graph_adapter import (
    GroundedGraphAdapter,
    pointer_loss,
    token_loss,
)
from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_feature_runner import load_example, read, verify
from next_iteration.grounded_graph_features import (
    EDGE_KINDS,
    NODE_KINDS,
    REPRESENTATION,
)
from next_iteration.grounded_graph_synthesize import progress
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest
from route_graph.frozen_reader import write_json_once

PROTOCOL = {"schema": "grounded-graph-training@1", "seed": 20260913, "epochs": 10,
    "batch_examples": 8, "learning_rate": 5e-4, "weight_decay": .01, "gradient_clip": 1.,
    "input_dim": 4096, "adapter_dim": 128, "node_types": len(NODE_KINDS), "edge_types": len(EDGE_KINDS),
    "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS, "message_steps": 2, "pointer_weight": 1.,
    "arms": ["graph", "no_edges"], "loss": "all verbatim source tokens CE + source-coordinate anchor first-token pointer NLL",
    "selection": "minimum source-validation token-mean CE + pointer-mean NLL; earliest tie",
    "labels_read": False, "representation": REPRESENTATION,
    "fixed_detection_score": "logp_base(observed token) - logp_adapter(observed token); larger is risk"}


def frozen_head(settings, device):
    path = Path(settings["model_path"])
    if model_manifest(path) != settings["model_files"]:
        raise ValueError("frozen observer/tokenizer bytes differ")
    config = read(path / "config.json")
    if config["model_type"] != "llama" or config["hidden_size"] != PROTOCOL["input_dim"]:
        raise ValueError("training requires bound 4096-dimensional Llama observer")
    index = read(path / "model.safetensors.index.json")["weight_map"]
    with safe_open(path / index["lm_head.weight"], framework="pt", device="cpu") as file:
        weight = file.get_tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    weight.requires_grad_(False)
    if weight.shape != (config["vocab_size"], config["hidden_size"]):
        raise ValueError("unembedding shape differs from observer")
    return weight


def new_adapter(device):
    torch.manual_seed(PROTOCOL["seed"])
    return GroundedGraphAdapter(**{k: PROTOCOL[k] for k in
        ("input_dim", "adapter_dim", "node_types", "edge_types", "message_steps")}).to(device)


def batch(examples, device):
    sources, queries, types, graph_ids, query_ids, available, edges, targets = [], [], [], [], [], [], [], []
    supervision, n, t = [], 0, 0
    for i, (packet, arrays) in enumerate(examples):
        nodes, count = packet["nodes"], len(packet["target_token_ids"])
        sources.append(arrays["source"])
        queries.append(arrays["query"])
        types.extend(x["type"] for x in nodes)
        graph_ids.extend([i] * len(nodes))
        query_ids.extend([i] * count)
        available.extend(x["available"] for x in nodes)
        edges.extend([a + n, b + n, kind] for a, b, kind in packet["edges"])
        targets.extend(packet["target_token_ids"])
        supervision.extend((x["query_index"] + t, [p + n for p in x["owner_node_indices"]]) for x in packet["pointer_supervision"])
        n += len(nodes)
        t += count
    edge = torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 3).T
    positives = torch.zeros((t, n), dtype=torch.bool, device=device)
    query_mask = torch.zeros(t, dtype=torch.bool, device=device)
    for q, owners in supervision:
        positives[q, owners] = True
        query_mask[q] = True
    arguments = {"source_states": torch.from_numpy(np.concatenate(sources)).to(device),
        "query_states": torch.from_numpy(np.concatenate(queries)).to(device),
        "node_types": torch.tensor(types, dtype=torch.long, device=device), "edge_index": edge[:2], "edge_types": edge[2],
        "graph_ids": torch.tensor(graph_ids, dtype=torch.long, device=device),
        "query_graph_ids": torch.tensor(query_ids, dtype=torch.long, device=device),
        "source_available": torch.tensor(available, dtype=torch.bool, device=device)}
    return arguments, torch.tensor(targets, dtype=torch.long, device=device), positives, query_mask


@torch.no_grad()
def token_statistics(hidden, targets, head, chunk_size=64):
    logp, entropy = [], []
    for start in range(0, len(hidden), chunk_size):
        logits = F.linear(hidden[start:start + chunk_size].to(head.dtype), head).float()
        logs = logits.log_softmax(-1)
        logp.append(logs.gather(-1, targets[start:start + chunk_size, None]).squeeze(-1))
        entropy.append(-(logs.exp() * logs).sum(-1))
    return torch.cat(logp), torch.cat(entropy)


def measure(adapter, examples, head, *, use_edges, optimizer=None, order=None):
    training = optimizer is not None
    adapter.train(training)
    order = list(range(len(examples))) if order is None else order
    ce_sum, pointer_sum, tokens, pointers, correct, actual = 0., 0., 0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), PROTOCOL["batch_examples"]):
            chosen = [examples[i] for i in order[start:start + PROTOCOL["batch_examples"]]]
            arguments, targets, positives, mask = batch(chosen, head.device)
            result = adapter(**arguments, use_edges=use_edges)
            ce = token_loss(result["hidden"], targets, head)
            pointer = pointer_loss(result, positives, mask)
            objective = ce + PROTOCOL["pointer_weight"] * pointer
            if not torch.isfinite(objective):
                raise ValueError("nonfinite grounded training objective")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), PROTOCOL["gradient_clip"])
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite grounded adapter gradients")
                optimizer.step()
            count = int(mask.sum())
            ce_sum += float(ce.detach()) * len(targets)
            pointer_sum += float(pointer.detach()) * count
            tokens += len(targets)
            pointers += count
            correct += int(positives[mask].gather(1, result["pointer_logits"][mask].argmax(-1)[:, None]).sum())
            actual += 1
    if not tokens or not pointers:
        raise ValueError("weak split contains no usable token/pointer examples")
    return {"token_ce": ce_sum / tokens, "pointer_nll": pointer_sum / pointers,
        "objective": ce_sum / tokens + PROTOCOL["pointer_weight"] * pointer_sum / pointers,
        "pointer_top1": correct / pointers, "tokens": tokens, "pointers": pointers, "adapter_forwards": actual}


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh adapter training output required")
    feature_settings, entries = verify(args.features, complete=True)
    if feature_settings["parent"]["kind"] != "source_reconstruction":
        raise ValueError("natural RAGTruth responses cannot train this adapter")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("free GPU required for adapter training")
    tokenizer = AutoTokenizer.from_pretrained(feature_settings["model_path"], local_files_only=True)
    examples = {"train": [], "validation": []}
    all_sha = {split: set() for split in examples}
    for entry in entries:
        if entry["split"] not in examples:
            raise ValueError("unexpected weak source split")
        all_sha[entry["split"]].add(entry["source_text_sha256"])
        if entry["status"] == "available":
            packet, arrays, _ = load_example(args.features, entry, feature_settings, tokenizer)
            examples[entry["split"]].append((packet, arrays))
    if all_sha["train"] & all_sha["validation"] or not all(examples.values()):
        raise ValueError("weak train/validation source leakage or empty split")
    paths = list(feature_settings["code_sha256"]) + ["next_iteration/grounded_graph_adapter.py", "next_iteration/grounded_graph_train.py"]
    code = {}
    for name in paths:
        code[name] = file_sha256(ROOT / name)
        dest = args.output / "executed_code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        if file_sha256(dest) != code[name]:
            raise ValueError("training code changed during snapshot")
    settings = {"protocol": PROTOCOL, "feature_path": str(args.features.resolve()),
        "feature_manifest_sha256": file_sha256(args.features / "manifest.json"), "code_sha256": code,
        "source_text_sha256": {k: sorted(v) for k, v in all_sha.items()}, "labels_read": False,
        "usable_examples": {k: len(v) for k, v in examples.items()}, "prepared_examples": len(entries)}
    write_json_once(args.output / "settings.json", settings)
    head = frozen_head(feature_settings, "cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    artifacts, summaries, started = {}, {}, time.monotonic()
    for arm in PROTOCOL["arms"]:
        adapter = new_adapter(head.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
        initial = measure(adapter, examples["validation"], head, use_edges=arm == "graph")
        name = f"{arm}/initial.json"
        write_json_once(args.output / name, initial)
        artifacts[name] = file_sha256(args.output / name)
        initial_checkpoint = f"{arm}/initial.pt"
        with (args.output / initial_checkpoint).open("xb") as stream:
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                "epoch": 0, "arm": arm, "settings_sha256": file_sha256(args.output / "settings.json"),
                "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS}, stream)
        artifacts[initial_checkpoint] = file_sha256(args.output / initial_checkpoint)
        best, selected = initial["objective"], {"epoch": 0, "checkpoint": initial_checkpoint,
            "checkpoint_sha256": artifacts[initial_checkpoint], "validation": initial}
        for epoch in range(1, PROTOCOL["epochs"] + 1):
            order = list(range(len(examples["train"])))
            random.Random(PROTOCOL["seed"] + epoch).shuffle(order)
            train = measure(adapter, examples["train"], head, use_edges=arm == "graph", optimizer=optimizer, order=order)
            validation = measure(adapter, examples["validation"], head, use_edges=arm == "graph")
            name = f"{arm}/epoch_{epoch:02d}"
            with (args.output / (name + ".pt")).open("xb") as stream:
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    "epoch": epoch, "arm": arm, "settings_sha256": file_sha256(args.output / "settings.json"),
                    "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS}, stream)
            record = {"epoch": epoch, "arm": arm, "train": train, "validation": validation, "labels_read": False}
            write_json_once(args.output / (name + ".json"), record)
            for suffix in (".pt", ".json"):
                artifacts[name + suffix] = file_sha256(args.output / (name + suffix))
            if validation["objective"] < best:
                best, selected = validation["objective"], {"epoch": epoch, "checkpoint": name + ".pt",
                    "checkpoint_sha256": artifacts[name + ".pt"], "validation": validation}
            progress(args.output, status="training", arm=arm, epoch=epoch, epochs=PROTOCOL["epochs"],
                train=train, validation=validation, seconds=time.monotonic() - started)
        summaries[arm] = {"initial_validation": initial, "selected": selected,
            "parameters": sum(p.numel() for p in adapter.parameters())}
        del adapter, optimizer
    verify(args.features, complete=True)
    for name, sha in code.items():
        if file_sha256(ROOT / name) != sha or file_sha256(args.output / "executed_code" / name) != sha:
            raise ValueError("training code changed during execution")
    summary = {"status": "complete", "arms": summaries, "seconds": time.monotonic() - started,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(), "observer_forwards": 0,
        "labels_read": False, "natural_detection_validated": False}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json")})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
