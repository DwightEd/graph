"""Capture four native conditions, freeze scores, then evaluate and auto-pack."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from ..inputs import validate_tokenizer
from .data import copy_annotations, prepare
from .scoring import BASELINES, PRIMARY, score_contrasts


def capture(args, settings):
    import torch
    from state_audit.model import load_model
    from .capture import collect_condition

    torch.set_num_threads(args.cpu_threads)
    model, tokenizer = None, None
    for index, response in enumerate(tqdm(settings["responses"], desc="contrast answers")):
        directory = args.output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        pending = [name for name in ("with_source", "without_source")
                   if not (directory / f"{name}_execution.json").exists()]
        if not pending:
            continue
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        validate_tokenizer(response, tokenizer)
        for condition in pending:
            measured = collect_condition(model, views[f"prompt_{condition}"], views["answer_ids"],
                views["units"], args.prefill_chunk_size, args.query_chunk_size, f"{response['id']} {condition}")
            write_arrays(directory / f"{condition}.npz", **measured)
            write_json(directory / f"{condition}_execution.json", dict(
                device=args.device, dtype=args.dtype, prefill_chunk_size=args.prefill_chunk_size,
                query_chunk_size=args.query_chunk_size, seconds=float(measured["seconds"]),
                peak_cuda_bytes=int(measured["peak_cuda_bytes"])))


def score_all(output, settings, protocol):
    coverage = []
    for index, response in enumerate(tqdm(settings["responses"], desc="freeze contrast scores")):
        directory = output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        baseline = read_arrays(directory / "baselines.npz")
        values = score_contrasts(read_arrays(directory / "with_source.npz"),
            read_arrays(directory / "without_source.npz"), views,
            {name: baseline[name] for name in BASELINES}, protocol["window"])
        write_arrays(directory / "scores.npz", **values)
        coverage.append(dict(response_id=response["id"], source_id=response["source_id"],
            planned_tokens=len(views["answer_ids"]), scored_tokens=len(values["target"]),
            unit_count=len(views["units"]), removed_prompt_tokens=len(views["removed_prompt_positions"])))
    write_json(output / "coverage.json", dict(status="complete", responses=coverage,
        selected_tokens=sum(row["planned_tokens"] for row in coverage),
        scored_tokens=sum(row["scored_tokens"] for row in coverage), dropped_tokens=0))


def pack(output):
    read_json(output / "protocol.json")
    archive = output.with_name(output.name + "_review.zip")
    temporary = archive.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and ".partial." not in path.name:
                bundle.write(path, path.relative_to(output))
    temporary.replace(archive)
    return str(archive)


def finish(args, settings, protocol):
    from .report import evaluate

    annotations = copy_annotations(args.output, Path(protocol["input"]), args.annotations)
    result = evaluate(args.output, settings, annotations, args.bootstrap)
    timing = [read_json(path) for path in sorted(args.output.glob("responses/*/*_execution.json"))]
    summary = dict(protocol=protocol, coverage=read_json(args.output / "coverage.json"),
        evaluation=result, capture_seconds=sum(row["seconds"] for row in timing),
        peak_cuda_bytes=max((row["peak_cuda_bytes"] for row in timing), default=0),
        score_units="nats/token for contrasts; native units for independent baselines")
    write_json(args.output / "summary.json", summary)
    return dict(output=str(args.output), primary_candidate=PRIMARY, review_archive=pack(args.output),
                scored_tokens=summary["coverage"]["scored_tokens"],
                all_error={name: {metric: value["all_error"][metric] for metric in ("auroc", "ap")}
                           for name, value in result.get("methods", {}).items()},
                evaluation_status=result["status"])


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing observable directory or its light/full ZIP")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("run", "capture", "score", "evaluate", "pack"), default="run")
    parser.add_argument("--max-unit-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200, help="Source bootstrap repetitions; 0 disables")
    parser.add_argument("--annotations", type=Path, help="Evaluation only; input annotations used by default")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_unit_tokens, args.window, args.prefill_chunk_size, args.query_chunk_size, args.cpu_threads) < 1:
        parser.error("Token windows, chunks and cpu-threads must be positive")
    if args.bootstrap < 0:
        parser.error("bootstrap must be nonnegative")
    if args.stage in ("run", "capture") and args.input is None:
        parser.error("--input is required for capture")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps(dict(review_archive=pack(args.output))))
        return
    if args.stage in ("run", "capture"):
        settings, protocol = prepare(args)
        capture(args, settings)
    else:
        settings = read_json(args.output / "settings.json")
        protocol = read_json(args.output / "protocol.json")
    if args.stage == "capture":
        print(json.dumps(dict(status="captured", output=str(args.output))))
        return
    if args.stage in ("run", "score"):
        score_all(args.output, settings, protocol)
    print(json.dumps(finish(args, settings, protocol), ensure_ascii=False))


if __name__ == "__main__":
    main()
