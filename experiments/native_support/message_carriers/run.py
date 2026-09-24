"""Capture sparse history-message interventions, freeze node vectors, evaluate and pack."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, start_stage, write_arrays, write_json

from ..evidence_contrast.data import copy_annotations
from ..evidence_contrast.run import pack
from ..inputs import validate_tokenizer
from .data import prepare
from .representation import assemble, feature_schema


def capture(args, settings):
    import torch
    from state_audit.model import load_model
    from .capture import capture_unit

    torch.set_num_threads(args.cpu_threads)
    model, tokenizer = None, None
    for index, response in enumerate(tqdm(settings["responses"], desc="carrier answers")):
        directory = args.output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        pending = [unit for unit in views["units"]
                   if not (directory / f"unit_{unit['start']:06d}" / "complete.json").is_file()]
        if not pending:
            continue
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        validate_tokenizer(response, tokenizer)
        for unit in tqdm(pending, desc=f"{response['id']} units", leave=False):
            destination = directory / f"unit_{unit['start']:06d}"
            started = perf_counter()
            if model.native.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.native.device)
            measured = capture_unit(model, views, unit, args.top_k,
                                    [args.seed, index, unit["start"]])
            write_arrays(destination / "measurements.npz", **measured)
            peak = torch.cuda.max_memory_allocated(model.native.device) if model.native.device.type == "cuda" else 0
            write_json(destination / "complete.json", dict(**unit, seconds=perf_counter() - started,
                peak_cuda_bytes=peak, device=args.device, dtype=args.dtype))


def score_all(output, settings):
    coverage = []
    for index, response in enumerate(tqdm(settings["responses"], desc="freeze node vectors")):
        directory = output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        measured = []
        for unit in views["units"]:
            destination = directory / f"unit_{unit['start']:06d}"
            read_json(destination / "complete.json")
            measured.append(read_arrays(destination / "measurements.npz"))
        schema = feature_schema(*measured[0]["eligible_counts"].shape)
        start_stage(output / "feature_schema.json", schema, resume=True)
        scores, nodes = assemble(views, read_arrays(directory / "baselines.npz"), measured)
        write_arrays(directory / "node_features.npz", **nodes)
        write_arrays(directory / "scores.npz", **scores)
        coverage.append(dict(response_id=response["id"], planned_tokens=len(views["answer_ids"]),
            represented_tokens=len(nodes["node_features"]), dimension=schema["dimension"],
            intervention_tokens=int((scores["selected_count"] > 0).sum()), units=len(measured),
            replay_source_gain_max_difference=float(abs(scores["source_full_replay"] - scores["source_full"]).max())))
    write_json(output / "coverage.json", dict(status="complete", responses=coverage,
        represented_tokens=sum(row["represented_tokens"] for row in coverage), dropped_tokens=0))


def finish(args, settings, protocol):
    from .report import evaluate

    coverage = read_json(args.output / "coverage.json")
    copy_annotations(args.output, Path(protocol["input"]), args.annotations)
    result = evaluate(args.output, settings)
    timings = [read_json(path) for path in args.output.glob("responses/*/unit_*/complete.json")]
    summary = dict(protocol=protocol, evaluation=result, coverage=coverage,
        capture_seconds=sum(row["seconds"] for row in timings),
        peak_cuda_bytes=max(row["peak_cuda_bytes"] for row in timings))
    write_json(args.output / "summary.json", summary)
    return dict(output=str(args.output), status=result["status"], review_archive=pack(args.output),
        represented_tokens=summary["coverage"]["represented_tokens"],
        all_error={name: {key: phases["all_error"][key] for key in ("auroc", "ap")}
                   for name, phases in result.get("methods", {}).items()})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed contrast/aggregation directory or review ZIP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "capture", "score", "evaluate", "pack"), default="run")
    parser.add_argument("--top-k", type=int, default=8, help="At most one history key per physical head")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--annotations", type=Path, help="Evaluation only; opened after all scores are saved")
    args = parser.parse_args(argv)
    if min(args.top_k, args.cpu_threads) < 1 or args.seed < 0:
        parser.error("top-k/threads must be positive and seed nonnegative")
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
        print(json.dumps(dict(output=str(args.output), status="captured", review_archive=pack(args.output))))
        return
    if args.stage in ("run", "score"):
        score_all(args.output, settings)
    print(json.dumps(finish(args, settings, protocol), ensure_ascii=False))


if __name__ == "__main__":
    main()
