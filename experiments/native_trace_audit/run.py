"""Replay only the reviewed prefixes; no model edits, generation, training or full-data test."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
from state_audit.storage import (
    read_json,
    start_stage,
    write_arrays,
    write_csv,
    write_json,
)
from tqdm import tqdm

from .inputs import compile_panels, inventory, query_plans

EXAMPLES = Path(__file__).parent / "examples" / "reviewed_prefixes.json"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLES)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/native_trace_audit_v1")
    )
    parser.add_argument(
        "--stage", choices=("inventory", "run", "report"), default="run"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--recent", type=int, default=10)
    parser.add_argument("--receiver-budget", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.window < 0
        or args.recent < 1
        or args.receiver_budget < 0
        or args.prefill_chunk_size < 1
    ):
        parser.error(
            "window/receiver-budget must be nonnegative; recent/prefill-chunk-size must be positive"
        )
    return args


def capture_panel(model, tokenizer, context, panel, args):
    from state_audit.native_trace import iter_native_traces

    directory = (
        args.output / "cases" / context["case_id"] / context["side"] / panel["name"]
    )
    plans = query_plans(
        context,
        panel,
        tokenizer.all_special_ids,
        args.window,
        args.recent,
        args.receiver_budget,
    )
    manifest = {
        "context": context,
        "panel": panel,
        "plans": plans,
        "special_ids": tokenizer.all_special_ids,
        "readout": "first_divergent_token_logit_difference_not_sequence_support",
    }
    start_stage(directory / "plan.json", manifest, args.resume)
    pending = [
        p for p in plans if not (directory / f"query_{p['query']:06d}.npz").exists()
    ]
    iterator = iter_native_traces(
        model, panel["token_ids"], pending, prefill_chunk_size=args.prefill_chunk_size
    )
    description = f"{context['case_id']}/{context['side']}/{panel['name']}"
    with closing(iterator):
        for plan in tqdm(pending, desc=description, leave=False):
            arrays = measured_query(model, iterator)
            arrays["decision_offset"] = np.asarray(plan["decision_offset"])
            validate_arrays(arrays, directory, plan["query"])
            write_arrays(directory / f"query_{plan['query']:06d}.npz", **arrays)
    write_json(directory / "complete.json", {"queries": len(plans), "complete": True})


def measured_query(model, iterator):
    import torch

    device = model.native.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    arrays = next(iterator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        arrays["cuda_peak_allocated_bytes"] = np.asarray(
            torch.cuda.max_memory_allocated(device)
        )
        arrays["cuda_peak_reserved_bytes"] = np.asarray(
            torch.cuda.max_memory_reserved(device)
        )
    arrays["capture_seconds"] = np.asarray(perf_counter() - started)
    return arrays


def validate_arrays(arrays, directory, query):
    for name, value in arrays.items():
        missing_observation = name == "observed_logp" and int(arrays["observed_id"]) < 0
        if (
            not missing_observation
            and value.dtype.kind == "f"
            and not np.isfinite(value).all()
        ):
            raise FloatingPointError(f"{directory}, query {query}: nonfinite {name}")


def main():
    args = arguments()
    from .report import write_report

    if args.stage == "report":
        print(json.dumps(write_report(args.output), ensure_ascii=False))
        return
    manifest = read_json(args.input)
    rows = inventory(manifest)
    write_csv(args.output / "inventory.csv", rows, list(rows[0]))
    if args.stage == "inventory":
        write_json(
            args.output / "inventory.json",
            {
                "purpose": "reviewed_input_validation_only",
                "contexts": len(rows),
                "rows": rows,
                "model_run": False,
            },
        )
        print(
            json.dumps(
                {
                    "contexts": len(rows),
                    "source_roles_verified": True,
                    "model_run": False,
                }
            )
        )
        return
    settings = {
        "model": args.model or manifest["model"],
        "device": args.device,
        "dtype": args.dtype,
        "window": args.window,
        "recent": args.recent,
        "receiver_budget": args.receiver_budget,
        "inputs": manifest,
        "purpose": "native_trajectory_mechanism_audit_not_detector_evaluation",
        "version": 1,
    }
    start_stage(args.output / "settings.json", settings, args.resume)
    run_replays(args, manifest, settings)
    print(json.dumps(write_report(args.output), ensure_ascii=False))


def run_replays(args, manifest, settings):
    from state_audit.model import load_model

    model, tokenizer = load_model(
        settings["model"], device=args.device, dtype=args.dtype
    )
    for context in tqdm(manifest["contexts"], desc="reviewed prefixes"):
        for panel in compile_panels(context, tokenizer):
            capture_panel(model, tokenizer, context, panel, args)


if __name__ == "__main__":
    main()
