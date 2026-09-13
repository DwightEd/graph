"""Frozen five-phase, full-word graph detector with separately measured dependence."""

import argparse
import fcntl
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from route_graph.audit_alignment import align_row
from route_graph.audit_artifacts import file_sha256, verify_donor_artifact
from route_graph.audit_runner import lock_stream, model_manifest, rows
from route_graph.frozen_reader import FrozenReader, digest, write_json_once
from route_graph.soft_graph_phases import (
    PROTOCOL,
    feature_phase,
    merge_phase,
    native_phase,
    semantic_phase,
    structure_phase,
)

GRAPH = Path(__file__).resolve().parents[1]
MODEL_ROOT = GRAPH.parent.parent / "models"
STAGES = PROTOCOL["phase_order"]


def code_files():
    paths = sorted((GRAPH / "route_graph").glob("*.py"))
    paths += [GRAPH / "experiments" / name for name in ("interleave_soft_graph.py", "evaluate_soft_graph.py")]
    return {str(p.relative_to(GRAPH)): file_sha256(p) for p in paths}


def verify_executed_code(output, frozen, live=True):
    snapshot = output / "executed_code"
    recorded = json.loads((snapshot / "manifest.json").read_text())
    if recorded != frozen["code_sha256"]:
        raise ValueError("executed-code manifest differs from frozen settings")
    for relative, checksum in recorded.items():
        if file_sha256(snapshot / relative) != checksum:
            raise ValueError("executed-code snapshot incomplete or changed")
        if live and file_sha256(GRAPH / relative) != checksum:
            raise ValueError("live code differs from this frozen run; execute the saved code snapshot")


def settings(args):
    expected = {"protocol": PROTOCOL, "input_path": str(args.inputs.resolve()),
                "input_sha256": file_sha256(args.inputs), "observer_model": str(args.observer_model.resolve()),
                "reader_model": str(args.reader_model.resolve()), "code_sha256": code_files(),
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "dtype": "bfloat16", "native_attention": "eager", "reader_attention": "sdpa", "seed": 20260913,
                "evaluation_manifest": {"path": str(args.evaluation_manifest.resolve()), "sha256": file_sha256(args.evaluation_manifest)}
                if args.evaluation_manifest is not None else None}
    destination = args.output / "settings.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("frozen code/input/runtime/protocol changed; preserve run and choose a fresh output")
        for name, path in (("observer_files", args.observer_model), ("reader_files", args.reader_model)):
            if saved[name] != model_manifest(path):
                raise ValueError("frozen model/tokenizer bytes or metadata changed")
        verify_executed_code(args.output, saved)
        return saved
    expected.update(observer_files=model_manifest(args.observer_model), reader_files=model_manifest(args.reader_model))
    snapshot = args.output / "executed_code"
    for relative, checksum in expected["code_sha256"].items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(GRAPH / relative, target)
        if file_sha256(target) != checksum:
            raise ValueError("source snapshot copy changed during preparation")
    if not (snapshot / "manifest.json").exists():
        write_json_once(snapshot / "manifest.json", expected["code_sha256"])
    verify_executed_code(args.output, expected)
    # Settings publication is the final preparation commit point. An earlier
    # interrupted partial snapshot is never silently trusted or overwritten.
    write_json_once(destination, expected)
    return expected


def read_artifact(output, stage, row, frozen, verified=None):
    verified = {} if verified is None else verified
    if stage in verified:
        return verified[stage]
    path = output / stage / f"{row['id']}.json"
    saved = json.loads(path.read_text())
    if (saved["settings_sha256"] != digest(frozen) or saved["row_sha256"] != digest(row)
            or saved["data_sha256"] != digest(saved["data"])):
        raise ValueError("graph stage artifact identity differs")
    expected = {f"{s}/{row['id']}.json" for s in STAGES[:STAGES.index(stage)]}
    if set(saved["upstream"]) != expected:
        raise ValueError("graph upstream dependency roster differs")
    for relative, checksum in saved["upstream"].items():
        if file_sha256(output / relative) != checksum:
            raise ValueError("upstream graph artifact changed")
    for earlier in STAGES[:STAGES.index(stage)]:
        read_artifact(output, earlier, row, frozen, verified)
    data = saved["data"]
    if stage == "B":
        for name, checksum in data["feature_files"].items():
            if file_sha256(output / "features" / str(row["id"]) / name) != checksum:
                raise ValueError("frozen high-dimensional feature file changed")
    if stage == "D":
        for claim in data["claims"]:
            for ref in claim.get("donor_artifacts", []):
                verify_donor_artifact(ref, output / "donors")
    verified[stage] = data
    return data


def publish(args, row, frozen, data):
    upstream = {f"{s}/{row['id']}.json": file_sha256(args.output / s / f"{row['id']}.json")
                for s in STAGES[:STAGES.index(args.stage)]}
    write_json_once(args.output / args.stage / f"{row['id']}.json",
                    {"schema": "soft-graph-stage@1", "stage": args.stage, "row_sha256": digest(row),
                     "settings_sha256": digest(frozen), "upstream": upstream, "data_sha256": digest(data), "data": data})


def progress(args, **record):
    path = args.output / "progress.json"
    temporary = args.output / f"progress.{os.getpid()}.partial"
    temporary.write_text(json.dumps({"pid": os.getpid(), "updated_unix": time.time(), **record}, indent=2) + "\n")
    os.replace(temporary, path)


def execute_stage(args, frozen, inputs):
    pending = []
    for row in inputs:
        if (args.output / args.stage / f"{row['id']}.json").exists():
            read_artifact(args.output, args.stage, row, frozen)
        else:
            pending.append(row)
    if not pending:
        return
    model = tokenizer = reader = None
    if args.stage != "merge":
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise RuntimeError("exclusive free CUDA GPU required; no population overlap")
        is_reader = args.stage in {"A", "C"}
        path = args.reader_model if is_reader else args.observer_model
        progress(args, status="loading_model", stage=args.stage, total=len(inputs), completed_stage=len(inputs)-len(pending))
        torch.manual_seed(20260913)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                    device_map={"": "cuda:0"}, attn_implementation="sdpa" if is_reader else "eager").eval()
        model.requires_grad_(False)
        if is_reader:
            reader = FrozenReader(model, tokenizer, args.output / "reader_cache",
                                  {"model": frozen["reader_files"], "tokenizer": frozen["reader_files"],
                                   "code_sha256": frozen["code_sha256"]}, PROTOCOL["reader_context_limit"])
    for index, row in enumerate(pending):
        state = {"status": "running", "stage": args.stage, "current_response": row["id"],
                 "completed_stage": len(inputs) - len(pending) + index, "total": len(inputs)}
        progress(args, **state)
        started = time.monotonic()
        if model is not None:
            torch.cuda.reset_peak_memory_stats()
        prior = {}
        for earlier in STAGES[:STAGES.index(args.stage)]:
            read_artifact(args.output, earlier, row, frozen, prior)
        counts = reader.outcomes.copy() if reader is not None else None
        if args.stage == "A":
            data = structure_phase(reader, row)
        elif args.stage == "B":
            data = feature_phase(model, tokenizer, row, prior["A"], frozen, args.output)
        elif args.stage == "C":
            data = semantic_phase(reader, row, prior["B"])
        elif args.stage == "D":
            data = native_phase(model, row, prior["B"], prior["C"], args.output,
                                heartbeat=lambda cid, state=state: progress(args, **state, current_claim=cid))
        else:
            data = merge_phase(row, prior["C"], prior["D"])
        if reader is not None:
            data["reader_outcomes"] = dict(reader.outcomes - counts)
        data["execution"] = {"seconds": time.monotonic() - started,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated() if model is not None else 0}
        publish(args, row, frozen, data)
    progress(args, status="phase_complete", stage=args.stage, completed_stage=len(inputs), total=len(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--observer-model", type=Path, default=MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--reader-model", type=Path, default=MODEL_ROOT / "Qwen3-8B")
    parser.add_argument("--inherited-lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=["prepare", "all", *STAGES], default="all")
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with lock_stream(args) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = rows(args)
        tokenizer = AutoTokenizer.from_pretrained(args.observer_model, local_files_only=True)
        for row in inputs:
            alignment = align_row(row, tokenizer)
            if len(row["token_ids"]) > PROTOCOL["native_input_limit"] or [list(x) for x in alignment["response_offsets"]] != row["offsets"]:
                raise ValueError("observer length or original response offset preflight failed")
        frozen = settings(args)
        if args.stage == "prepare":
            print(json.dumps({"prepared": str(args.output), "responses": len(inputs), "settings_sha256": digest(frozen)}), flush=True)
            return
        for stage in STAGES if args.stage == "all" else [args.stage]:
            args.stage = stage
            settings(args)
            execute_stage(args, frozen, inputs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.stage == "merge":
            progress(args, status="complete", responses=len(inputs), claim_supported="not_evaluated")


if __name__ == "__main__":
    main()
