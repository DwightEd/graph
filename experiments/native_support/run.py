"""Forward capture -> signed support graph -> per-token scores, without ablations."""

import argparse
import json
from pathlib import Path

from state_audit.storage import read_json, start_stage, write_csv, write_json
from tqdm import tqdm

from .inputs import load_responses
from .pipeline import capture_response, score_response
from .report import write_report

EXAMPLE = Path(__file__).parent / "examples" / "prefixes.json"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "score", "evaluate"), default="run")
    parser.add_argument("--input", type=Path, default=EXAMPLE)
    parser.add_argument("--output", type=Path, default=Path("outputs/native_support_v1"))
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--annotations", type=Path, help="Evaluation only: response ID -> binary token labels")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_chunk_size < 1:
        parser.error("prefill chunk size must be positive")
    if args.stage == "evaluate" and args.annotations is None:
        parser.error("evaluate requires --annotations")
    return args


def prepare(args):
    model_name, responses = load_responses(args.input)
    settings = {
        "version": "native-support-v1", "model": args.model or model_name,
        "device": args.device, "dtype": args.dtype, "responses": responses,
        "prefill_chunk_size": args.prefill_chunk_size,
        "readout": "observed_token_vs_native_highest_other", "labels_used": False,
        "gradients": False, "interventions": False,
    }
    start_stage(args.output / "settings.json", settings, args.resume)
    return settings


def run_capture(args, settings):
    from state_audit.model import load_model

    model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
    for index, response in enumerate(tqdm(settings["responses"], desc="natural prefixes")):
        directory = args.output / "responses" / f"{index:04d}"
        capture_response(model, tokenizer, response, directory, args.prefill_chunk_size)


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


def main(argv=None):
    args = arguments(argv)
    if args.stage == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.output, args.annotations)
    elif args.stage == "score":
        result = run_score(args.output, read_json(args.output / "settings.json"))
    else:
        settings = prepare(args)
        run_capture(args, settings)
        result = run_score(args.output, settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
