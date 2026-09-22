"""Replay immutable cached tokens at labelled short spans and matched controls."""

import argparse
import hashlib
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from state_audit.attribution import aggregate_sources, iter_target_attributions
from state_audit.model.adapter import load_model
from state_audit.storage import read_json, start_stage, write_arrays, write_json
from state_audit.tokenization import special_token_ids
from tqdm import tqdm

from ..unsupervised_token_graph.evaluation_data import verified_offsets
from ..unsupervised_token_graph.span_audit.inputs import AuditInputs
from .capture_reporting import answer_directory, write_report

SOURCE_NAMES = ("ordinary_prompt", "recent_history", "earlier_history")


def source_groups(prompt_length, target, local_window):
    """Prefix keys only: target t is predicted at query P + t - 1."""
    groups = np.zeros(prompt_length + target, dtype=np.int64)
    groups[prompt_length:] = 2
    recent_start = max(prompt_length, len(groups) - local_window)
    groups[recent_start:] = 1
    return groups


def target_plan(entry):
    """Capture an overlapping target once while retaining every span membership."""
    plan = {}
    for membership in entry["targets"]:
        plan.setdefault(int(membership["target"]), []).append(membership)
    return dict(sorted(plan.items()))


def verify_cohort_answer(answer, entry):
    """A different tokenization cannot silently reuse gold target positions."""
    record = entry["record"]
    identity_matches = (
        str(answer.response_id) == str(record["id"])
        and str(answer.source_id) == str(record["source_id"])
        and answer.task == record["task"]
        and answer.generator == record["generator"]
        and answer.split == record["split"]
        and answer.prompt_length == record["prompt_length"]
        and len(answer.response_ids) == entry["tokens"]
        and answer.text == entry["text"]
        and np.array_equal(answer.offsets, entry["offsets"])
    )
    if not identity_matches:
        raise ValueError(f"{record['id']}: cached answer differs from frozen short-span cohort")


def verify_model_alignment(answer, tokenizer):
    offsets = verified_offsets(tokenizer, answer.token_ids, answer.prompt_length, answer.text)
    if not np.array_equal(offsets, answer.offsets):
        raise ValueError(f"{answer.response_id}: observer tokenizer differs from cached response")


def add_source_arrays(result, answer, local_window):
    """Summarize a measured target without changing its native per-edge arrays."""
    target = int(result["target"])
    groups = source_groups(answer.prompt_length, target, local_window)
    grouped = aggregate_sources(result, groups, len(SOURCE_NAMES))
    result.update({"source_" + name: values for name, values in grouped.items()})
    result["source_groups"] = groups
    special = ~result["ordinary_keys"]
    contribution = result["contribution"]
    excluded_values = {
        "route_mass": result["attention"],
        "contribution_positive": np.maximum(contribution, 0),
        "contribution_negative": np.maximum(-contribution, 0),
        "value_energy": result["value_energy"],
    }
    for name, values in excluded_values.items():
        result["excluded_special_" + name] = values[..., special].sum(-1)
    invalid = [name for name, values in result.items() if not np.isfinite(values).all()]
    if invalid:
        raise FloatingPointError(f"{answer.response_id} target {target}: nonfinite {invalid}")
    return result


def measured_targets(model, answer, targets, settings, excluded, chunk_size):
    """Time actual replay work, excluding disk writes; the first target pays for prefill."""
    device = model.native.device
    iterator = iter_target_attributions(
        model,
        answer.token_ids,
        answer.prompt_length,
        targets,
        layers=settings["layers"],
        special_token_ids=excluded,
        prefill_chunk_size=chunk_size,
    )
    cached_tokens = 0
    with closing(iterator):
        for target in tqdm(targets, desc=f"targets {answer.response_id}", leave=False):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            arrays = next(iterator)
            arrays["capture_seconds"] = np.asarray(perf_counter() - started)
            query = int(arrays["query"])
            arrays["prefill_tokens"] = np.asarray(query - cached_tokens)
            cached_tokens = query + 1
            if device.type == "cuda":
                arrays["cuda_peak_allocated_bytes"] = np.asarray(
                    torch.cuda.max_memory_allocated(device)
                )
                arrays["cuda_peak_reserved_bytes"] = np.asarray(
                    torch.cuda.max_memory_reserved(device)
                )
            yield add_source_arrays(arrays, answer, settings["local_window"])


def save_answer(directory, answer, entry, observer, excluded, complete):
    metadata = {
        "record": entry["record"],
        "observer": observer,
        "generator": answer.generator,
        "scope": "observer_teacher_forced_target_sensitivity",
        "evidence_applicability": "not_measured",
        "source_names": SOURCE_NAMES,
        "prompt_length": answer.prompt_length,
        "token_ids": answer.token_ids.tolist(),
        "text": answer.text,
        "offsets": answer.offsets.tolist(),
        "targets": entry["targets"],
        "gold": entry["gold"],
        "special_token_ids": excluded,
        "complete": complete,
    }
    write_json(directory / "answer.json", metadata)


def capture_answer(model, answer, entry, output, settings, excluded, *, prefill_chunk_size=256):
    directory = answer_directory(output, entry)
    verify_cohort_answer(answer, entry)
    completed = directory / "answer.json"
    if completed.exists():
        saved = read_json(completed)
        if saved["token_ids"] != answer.token_ids.tolist():
            raise ValueError(f"{answer.response_id}: saved capture token IDs changed")
    else:
        save_answer(directory, answer, entry, settings["model"], excluded, False)
    pending = [
        target for target in target_plan(entry) if not (directory / f"{target:06d}.npz").exists()
    ]
    if pending:
        measurements = measured_targets(
            model, answer, pending, settings, excluded, prefill_chunk_size
        )
        with closing(measurements):
            for arrays in measurements:
                write_arrays(directory / f"{int(arrays['target']):06d}.npz", **arrays)
    save_answer(directory, answer, entry, settings["model"], excluded, True)


def input_locations(args, original, cohort):
    dataset = Path(args.dataset or original["dataset"]).expanduser()
    if dataset.suffix == ".jsonl":
        dataset = dataset.parent
    splits = sorted({entry["record"]["split"] for entry in cohort})
    caches = {split: str(args.cache or original[f"{split}_cache"]) for split in splits}
    return dataset, caches


def capture_settings(args, cohort_path, dataset, caches):
    return {
        "version": "short-span-target-contributions-v1",
        "purpose": "label_assisted_short_span_audit_not_detector_evaluation",
        "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "model": args.model,
        "revision": args.revision,
        "layers": args.layers,
        "dtype": args.dtype,
        "dataset": str(dataset),
        "caches": caches,
        "local_window": args.local_window,
        "source_names": SOURCE_NAMES,
        "natural_labels_used_for_target_selection": True,
        "contribution": "post_softmax_message_gate_derivative_of_target_log_odds",
        "value_energy": "sum_of_individual_edge_squared_projected_message_norms",
    }


def run(args):
    cohort_path = args.audit / "cohort.json"
    cohort = read_json(cohort_path)
    original = read_json(args.audit / "input_settings.json")
    args.model = args.model or original["tokenizer"]
    dataset, caches = input_locations(args, original, cohort)
    output = args.audit / "contributions"
    settings = capture_settings(args, cohort_path, dataset, caches)
    start_stage(output / "settings.json", settings, args.resume)
    selected = [entry for entry in cohort if entry["targets"]]
    if args.limit_answers is not None:
        selected = selected[: args.limit_answers]
    readers, model, excluded = {}, None, None
    for entry in tqdm(selected, desc="short-span answers", unit="answer"):
        root = caches[entry["record"]["split"]]
        if root not in readers:
            readers[root] = AuditInputs(
                root,
                dataset,
                index_path=original.get("index"),
                tokenizer=args.model,
                include_labels=False,
            )
        answer = readers[root].load_answer(str(entry["record"]["id"]))
        if model is None:
            model, tokenizer = load_model(args.model, args.revision, args.device, args.dtype)
            excluded = special_token_ids(tokenizer)
        verify_model_alignment(answer, tokenizer)
        capture_answer(
            model,
            answer,
            entry,
            output,
            settings,
            excluded,
            prefill_chunk_size=args.prefill_chunk_size,
        )
    summary = write_report(output, cohort)
    print(summary, flush=True)


def parse_layers(value):
    if ":" in value:
        start, stop = map(int, value.split(":"))
        layers = list(range(start, stop))
    else:
        layers = [int(layer) for layer in value.split(",")]
    if not layers or min(layers) < 0 or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be nonempty, unique, nonnegative indices")
    return layers


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--model", help="Observer checkpoint; defaults to input_settings.json tokenizer"
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--layers", type=parse_layers, default=list(range(8, 24)))
    parser.add_argument("--local-window", type=nonnegative_int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--limit-answers", type=positive_int)
    parser.add_argument(
        "--prefill-chunk-size",
        type=positive_int,
        default=256,
        help="Historical KV prefill chunk size; all prefix tokens remain available",
    )
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
