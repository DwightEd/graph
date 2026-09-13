"""Train source-restoration adapters with actual source-erased pre-token states.

Numerical kernels/configuration are reused unchanged from frozen v1. This runner
binds a different feature protocol and writes fresh v2 checkpoints and receipts.
"""
import argparse
import fcntl
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from next_iteration.grounded_graph_data import ROOT
from next_iteration.grounded_graph_features import EDGE_KINDS, NODE_KINDS
from next_iteration.grounded_graph_restoration_features import (
    REPRESENTATION,
    load_example,
    verify,
)
from next_iteration.grounded_graph_synthesize import progress
from next_iteration.grounded_graph_train import PROTOCOL as ORIGINAL_PROTOCOL
from next_iteration.grounded_graph_train import frozen_head, measure, new_adapter
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import write_json_once

PROTOCOL = {**ORIGINAL_PROTOCOL, "schema": "grounded-graph-restoration-training@2",
    "representation": REPRESENTATION,
    "fixed_detection_score": "logp_full - logp_restore; secondary logp_empty - logp_restore; both larger is risk",
    "mechanism_gate": "selected model must yield positive source-heldout anchor gain over empty LM and positive fixed-query graph-payload erasure drop; separate from checkpoint selection"}


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
    paths = list(feature_settings["code_sha256"]) + ["next_iteration/grounded_graph_adapter.py", "next_iteration/grounded_graph_train.py", "next_iteration/grounded_graph_restoration_train.py"]
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
