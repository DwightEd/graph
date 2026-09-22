"""Native capture -> causal routing estimates -> frozen-score evaluation."""

import argparse
import json
from pathlib import Path

from state_audit.storage import read_json, start_stage, write_csv, write_json
from tqdm import tqdm

from .inputs import load_responses
from .pipeline import capture_response, pending_targets, score_response
from .report import write_report

EXAMPLE = Path(__file__).parent / "examples" / "prefixes.json"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "score", "optimize", "model", "compare", "evaluate"), default="run")
    parser.add_argument("--input", type=Path, default=EXAMPLE)
    parser.add_argument("--output", type=Path, default=Path("outputs/native_support_v1"))
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=8, help="Causal query rows per GPU forward; execution setting, safe to change on resume")
    parser.add_argument("--compress-cache", action="store_true", help="Compress token NPZ files to save disk space at the cost of CPU time")
    parser.add_argument("--window", type=int, default=16, help="Causal routing window; default retained from prior temporal code")
    parser.add_argument("--reference-output", type=Path, help="Independent cached reference for --stage model; default excludes each target source from input cohort")
    parser.add_argument("--annotations", type=Path, help="Evaluation only: response ID -> binary token labels")
    parser.add_argument("--dataset", type=Path, help="Official RAGTruth directory; prepare input and annotations automatically")
    parser.add_argument("--task", choices=("QA", "Summary", "Data2txt"), default="QA")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--generator", default="llama-2-7b-chat")
    parser.add_argument("--limit", type=int, default=4, help="Maximum official answers; default is a small pilot")
    parser.add_argument("--balanced", action="store_true", help="Diagnostic cohort: equal positive/negative answer counts, using labels only for selection")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_chunk_size < 1 or args.query_chunk_size < 1:
        parser.error("prefill and query chunk sizes must be positive")
    if args.window < 1:
        parser.error("window must be positive")
    if args.limit < 1 or (args.balanced and (args.limit < 2 or args.limit % 2)):
        parser.error("limit must be positive; balanced pilot needs an even limit of at least two")
    if args.stage == "prepare" and args.dataset is None:
        parser.error("prepare requires --dataset pointing to the official RAGTruth directory")
    if args.dataset is not None and args.stage in ("score", "optimize", "model", "compare", "evaluate"):
        parser.error("--dataset prepares new inputs: use --stage prepare or run, with a new output directory")
    if args.reference_output is not None and args.stage != "model":
        parser.error("--reference-output is only used by --stage model")
    return args


def prepare(args):
    manifest = None
    if args.dataset is not None:
        from .ragtruth import prepare_official
        model_name = args.model or read_json(args.input)["model"]
        manifest, annotations, cohort = prepare_official(args, model_name)
        responses = manifest["responses"]
    else:
        model_name, responses = load_responses(args.input)
    settings = {
        "version": "native-support-v1", "model": args.model or model_name,
        "device": args.device, "dtype": args.dtype, "responses": responses,
        "prefill_chunk_size": args.prefill_chunk_size,
        "readout": "observed_token_vs_native_highest_other", "labels_used": False,
        "gradients": False, "interventions": False,
    }
    if manifest is not None:
        settings["cohort"] = cohort
    start_stage(args.output / "settings.json", settings, args.resume)
    if manifest is not None:
        write_json(args.output / "input.json", manifest)
        write_json(args.output / "annotations.json", annotations)
    return settings


def run_capture(args, settings):
    from state_audit.model import load_model

    jobs = []
    for index, response in enumerate(settings["responses"]):
        directory = args.output / "responses" / f"{index:04d}"
        if pending_targets(response, directory):
            jobs.append((response, directory))
    if not jobs:
        return
    model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
    for response, directory in tqdm(jobs, desc="natural prefixes"):
        capture_response(
            model, tokenizer, response, directory, args.prefill_chunk_size,
            query_chunk_size=args.query_chunk_size, compress_cache=args.compress_cache,
        )


def run_score(output, settings):
    rows = []
    for index, response in enumerate(tqdm(settings["responses"], desc="support graphs")):
        directory = output / "responses" / f"{index:04d}"
        rows.extend(score_response(response, directory))
    write_csv(output / "tokens.csv", rows, list(rows[0]))
    write_report(output / "report.html", rows)
    summary = {
        "purpose": "label_free_native_support_detection_candidate",
        "responses": len(settings["responses"]), "scored_tokens": len(rows),
        "labels_used": False, "gradients": False, "interventions": False,
        "evaluation_performed_by_this_stage": False, "dataset_scope": "input_manifest_only",
        "max_abs_ledger_error": max(abs(row["ledger_error"]) for row in rows),
        "risk_definition": "minus_signed_prompt_support_with_causal_history_reuse",
    }
    write_json(output / "summary.json", summary)
    return summary


def evaluate_saved(args):
    from .comparison import DIRECTORY, evaluate_existing
    from .evaluate import evaluate
    from .optimize import DIRECTORY as FILTER_DIRECTORY
    from .optimize import evaluate_optimization
    from .state_model import DIRECTORY as STATE_DIRECTORY
    from .state_model import evaluate_state_model

    if (args.output / STATE_DIRECTORY / f"w{args.window}" / "summary.json").exists():
        return evaluate_state_model(args.output, args.annotations, args.window)
    if (args.output / FILTER_DIRECTORY / "features").exists():
        return evaluate_optimization(args.output, args.annotations, args.window)
    if (args.output / DIRECTORY / "summary.json").exists():
        return evaluate_existing(args.output, args.annotations)
    return evaluate(args.output, args.annotations)


def main(argv=None):
    from .comparison import run_comparison
    from .optimize import run_optimization

    args = arguments(argv)
    if args.stage == "evaluate":
        result = evaluate_saved(args)
    elif args.stage == "prepare":
        settings = prepare(args)
        result = {"status": "prepared", "responses": len(settings["responses"]),
                  "annotations": str(args.output / "annotations.json"), "model_run": False}
    elif args.stage == "compare":
        result = run_comparison(args.output, read_json(args.output / "settings.json"), args.annotations)
    elif args.stage == "model":
        from .state_model import run_state_model
        result = run_state_model(args.output, read_json(args.output / "settings.json"),
                                 args.annotations, args.window, args.reference_output)
    elif args.stage in ("score", "optimize"):
        result = run_optimization(args.output, read_json(args.output / "settings.json"), args.annotations, args.window)
    else:
        settings = prepare(args)
        run_capture(args, settings)
        result = run_optimization(args.output, settings, args.annotations, args.window)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
