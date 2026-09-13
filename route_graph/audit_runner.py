"""Run the frozen natural audit in four exclusive GPU phases and a CPU merge."""

import argparse
import fcntl
import gc
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from route_graph.audit_artifacts import file_sha256, verify_donor_artifact
from route_graph.audit_output import merge_response
from route_graph.audit_phase_native import proposal_phase, validation_phase
from route_graph.audit_prepare import prepare_anchors
from route_graph.audit_protocol import PROTOCOL
from route_graph.audit_semantics import label_fixed_pool
from route_graph.frozen_reader import FrozenReader, digest, write_json_once

GRAPH = Path(__file__).resolve().parents[1]
MODEL_ROOT = GRAPH.parent.parent / "models"
DEPENDENCIES = {
    "A": [],
    "B": ["A"],
    "C": ["A", "B"],
    "D": ["A", "B", "C"],
    "merge": ["A", "B", "C", "D"],
}


def model_manifest(path):
    return [
        {
            "name": p.name,
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": file_sha256(p),
        }
        for p in sorted(path.iterdir())
        if p.is_file()
    ]


def code_manifest():
    names = {
        "frozen_reader.py",
        "evidence_anchor.py",
        "atomic_anchor.py",
        "cloze_anchor.py",
        "cloze_prompts.py",
        "json_framing.py",
        "metrics.py",
        "native_audit.py",
        "causal_contrast.py",
        "causal_groups.py",
    }
    names.update(p.name for p in (GRAPH / "route_graph").glob("audit_*.py"))
    return {name: file_sha256(GRAPH / "route_graph" / name) for name in sorted(names)}


def settings(args):
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "settings.json"
    evaluation_manifest = getattr(args, "evaluation_manifest", None)
    expected = {
        "protocol": PROTOCOL,
        "input_path": str(args.inputs.resolve()),
        "input_sha256": file_sha256(args.inputs),
        "observer_model": str(args.observer_model.resolve()),
        "reader_model": str(args.reader_model.resolve()),
        "code_sha256": code_manifest(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "dtype": "bfloat16",
        "native_attention": "eager",
        "reader_attention": "sdpa",
        "seed": 20260913,
        "evaluation_manifest": {
            "path": str(evaluation_manifest.resolve()),
            "sha256": file_sha256(evaluation_manifest),
        }
        if evaluation_manifest is not None
        else None,
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if any(saved.get(k) != value for k, value in expected.items()):
            raise ValueError(
                "frozen input/code/protocol/runtime differs; use a new output directory"
            )
        for key, path in (
            ("observer_files", args.observer_model),
            ("reader_files", args.reader_model),
        ):
            if model_manifest(path) != saved[key]:
                raise ValueError("frozen model/tokenizer bytes or metadata changed")
        return saved
    expected.update(
        observer_files=model_manifest(args.observer_model),
        reader_files=model_manifest(args.reader_model),
    )
    write_json_once(destination, expected)
    return expected


def rows(args):
    result = [
        json.loads(line)
        for line in args.inputs.read_text().splitlines()
        if line.strip()
    ]
    if len({str(row["id"]) for row in result}) != len(result):
        raise ValueError("duplicate response IDs")
    allowed = {
        "generator",
        "id",
        "official_split",
        "offsets",
        "prompt",
        "prompt_length",
        "response",
        "response_sha256",
        "source_id",
        "source_mask",
        "source_span",
        "task",
        "token_ids",
    }
    for row in result:
        if set(row) != allowed:
            raise ValueError(
                "method inputs must match the exact annotation-free schema"
            )
        import hashlib

        if (
            hashlib.sha256(row["response"].encode()).hexdigest()
            != row["response_sha256"]
        ):
            raise ValueError("response text/hash mismatch")
        if not str(row["id"]).isdigit():
            raise ValueError("response IDs must be digits for immutable artifact names")
    return result


def read_artifact(output, stage, row, setting_hash, verified=None):
    verified = {} if verified is None else verified
    if stage in verified:
        return verified[stage]
    path = output / stage / f"{row['id']}.json"
    saved = json.loads(path.read_text())
    if saved["settings_sha256"] != setting_hash or saved["row_sha256"] != digest(row):
        raise ValueError("stage artifact input/settings mismatch")
    if saved["content_sha256"] != digest(saved["data"]):
        raise ValueError("stage artifact checksum mismatch")
    expected = {f"{s}/{row['id']}.json" for s in DEPENDENCIES[stage]}
    if set(saved["upstream"]) != expected:
        raise ValueError(
            "stage artifact does not name the exact required upstream chain"
        )
    for relative, checksum in saved["upstream"].items():
        if file_sha256(output / relative) != checksum:
            raise ValueError("upstream artifact changed after stage publication")
    for dependency in DEPENDENCIES[stage]:
        read_artifact(output, dependency, row, setting_hash, verified)
    if stage == "D":
        for result in saved["data"].values():
            refs = result.get("donor_artifacts", []) + result.get("template", {}).get(
                "donor_artifacts", []
            )
            for ref in refs:
                verify_donor_artifact(ref, output / "donors")
    verified[stage] = saved["data"]
    return saved["data"]


def publish(args, stage, row, data, setting_hash):
    dependencies = DEPENDENCIES[stage]
    upstream = {
        f"{s}/{row['id']}.json": file_sha256(args.output / s / f"{row['id']}.json")
        for s in dependencies
    }
    write_json_once(
        args.output / stage / f"{row['id']}.json",
        {
            "settings_sha256": setting_hash,
            "row_sha256": digest(row),
            "content_sha256": digest(data),
            "upstream": upstream,
            "data": data,
        },
    )


def progress(args, data):
    path = args.output / "progress.json"
    temporary = path.with_name(f"progress.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps({**data, "pid": os.getpid(), "updated_unix": time.time()}, indent=2)
        + "\n"
    )
    os.replace(temporary, path)
    print(json.dumps(data), flush=True)


def selected_questions(anchors):
    selected = set(anchors["native_selected_question_ids"])
    return [q for q in anchors["questions"] if q.get("id") in selected]


def process_row(args, row, frozen, model, tokenizer, observer_tokenizer, setting_hash):
    stage = args.stage
    if stage == "A":
        reader_identity = {
            "model": frozen["reader_files"],
            "tokenizer": frozen["reader_files"],
            "code_sha256": frozen["code_sha256"],
        }
        reader = FrozenReader(
            model,
            tokenizer,
            args.output / "reader_cache",
            reader_identity,
            context_limit=PROTOCOL["reader_context_limit"],
        )
        result = prepare_anchors(reader, observer_tokenizer, row)
        result["reader_requests_executed"] = reader.calls
        result["reader_outcomes"] = dict(reader.outcomes)
        return result
    anchors = read_artifact(args.output, "A", row, setting_hash)
    proposals = (
        read_artifact(args.output, "B", row, setting_hash)
        if stage in {"C", "D", "merge"}
        else None
    )
    semantics = (
        read_artifact(args.output, "C", row, setting_hash)
        if stage in {"D", "merge"}
        else None
    )
    if stage == "merge":
        validations = read_artifact(args.output, "D", row, setting_hash)
        return merge_response(row, anchors, proposals, semantics, validations)
    results = {}
    reader = None
    if stage == "C":
        reader = FrozenReader(
            model,
            tokenizer,
            args.output / "reader_cache",
            {
                "model": frozen["reader_files"],
                "tokenizer": frozen["reader_files"],
                "code_sha256": frozen["code_sha256"],
            },
        )
    for question in selected_questions(anchors):
        qid = question["id"]
        if not question.get("events"):
            results[qid] = {"status": "contrast_unavailable", "forward_calls": 0}
        elif stage == "B":
            try:
                results[qid] = proposal_phase(
                    model, row, question, anchors["alignment"]
                )
            except ValueError as error:
                # Only pre-forward input-limit rejection belongs here; failures
                # after starting measurement are retained by proposal_phase.
                if "within input limit" not in str(error):
                    raise
                results[qid] = {
                    "status": "native_input_limit",
                    "reason": str(error),
                    "forward_calls": 0,
                }
        elif proposals[qid].get("status") != "proposed":
            results[qid] = {
                "status": "proposal_unavailable",
                "forward_calls": proposals[qid].get("forward_calls", 0),
                "tokens_processed": proposals[qid].get("tokens_processed", 0),
            }
        elif stage == "C":
            outcomes_before = reader.outcomes.copy()
            results[qid] = label_fixed_pool(
                reader,
                row,
                question,
                proposals[qid]["frozen"],
                previous_questions=[
                    q
                    for q in anchors["questions"]
                    if q.get("claim_span") == question.get("previous_claim_span")
                    and q.get("id") is not None
                    and "answer_span" in q
                ],
            )
            results[qid]["reader_outcomes"] = dict(reader.outcomes - outcomes_before)
        else:
            results[qid] = validation_phase(
                model,
                row,
                question,
                proposals[qid],
                semantics[qid],
                anchors["alignment"],
                args.output / "donors" / str(row["id"]) / qid,
            )
    return results


def execute_stage(args, frozen, inputs):
    setting_hash = digest(frozen)
    pending = []
    for row in inputs:
        if (args.output / args.stage / f"{row['id']}.json").exists():
            read_artifact(args.output, args.stage, row, setting_hash)
        else:
            pending.append(row)
    if not pending:
        return
    model = tokenizer = observer_tokenizer = None
    if args.stage != "merge":
        if not torch.cuda.is_available():
            raise RuntimeError("a free CUDA GPU is required")
        free, _ = torch.cuda.mem_get_info()
        if free < 20 * 1024**3:
            raise RuntimeError(
                "GPU is not free; do not overlap population and native audit"
            )
        torch.manual_seed(20260913)
        reader_stage = args.stage in {"A", "C"}
        path = args.reader_model if reader_stage else args.observer_model
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa" if reader_stage else "eager",
        ).eval()
        model.requires_grad_(False)
        if args.stage == "A":
            observer_tokenizer = AutoTokenizer.from_pretrained(
                args.observer_model, local_files_only=True
            )
    for index, row in enumerate(pending):
        started = time.monotonic()
        if model is not None:
            torch.cuda.reset_peak_memory_stats()
        progress(
            args,
            {
                "status": "running",
                "stage": args.stage,
                "current_response": row["id"],
                "completed_stage": len(inputs) - len(pending) + index,
                "total": len(inputs),
            },
        )
        data = process_row(
            args, row, frozen, model, tokenizer, observer_tokenizer, setting_hash
        )
        publish(args, args.stage, row, data, setting_hash)
        write_json_once(
            args.output / "timing" / args.stage / f"{row['id']}.json",
            {
                "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if model is not None
                else None,
            },
        )
    progress(
        args,
        {
            "status": "phase_complete",
            "stage": args.stage,
            "completed_stage": len(inputs),
            "total": len(inputs),
        },
    )


def lock_stream(args):
    path = args.output / ".phase.lock"
    if args.inherited_lock_fd is None:
        return path.open("a")
    descriptor = os.dup(args.inherited_lock_fd)
    actual, expected = os.fstat(descriptor), path.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        os.close(descriptor)
        raise ValueError("inherited descriptor is not this audit's run lock")
    return os.fdopen(descriptor, "a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument(
        "--observer-model", type=Path, default=MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--reader-model", type=Path, default=MODEL_ROOT / "Qwen3-8B")
    parser.add_argument(
        "--inherited-lock-fd", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--stage", choices=["prepare", "all", *PROTOCOL["phase_order"]], default="all"
    )
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with lock_stream(args) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen, inputs = settings(args), rows(args)
        if args.stage == "prepare":
            print(
                json.dumps(
                    {
                        "prepared": str(args.output),
                        "responses": len(inputs),
                        "settings_sha256": digest(frozen),
                    }
                ),
                flush=True,
            )
        elif args.stage == "all":
            for stage in PROTOCOL["phase_order"]:
                args.stage = stage
                settings(args)  # Recheck immutable code/weights before each load.
                execute_stage(args, frozen, inputs)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            progress(
                args,
                {
                    "status": "complete",
                    "responses": len(inputs),
                    "claim_supported": "not_evaluated",
                },
            )
        else:
            execute_stage(args, frozen, inputs)


if __name__ == "__main__":
    main()
