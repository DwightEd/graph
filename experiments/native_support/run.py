"""Native capture -> per-token observations -> frozen-score evaluation."""

import argparse
import json
from pathlib import Path

from state_audit.storage import read_json, start_stage, write_json
from tqdm import tqdm

from .inputs import load_responses
from .pipeline import capture_response, pending_targets

EXAMPLE = Path(__file__).parent / "examples" / "prefixes.json"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "score", "compare", "evaluate"), default="run")
    parser.add_argument("--input", type=Path, default=EXAMPLE)
    parser.add_argument("--output", type=Path, default=Path("outputs/native_support_v1"))
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=8, help="Causal query rows per GPU forward; execution setting, safe to change on resume")
    parser.add_argument("--compress-cache", action="store_true", help="Compress token NPZ files to save disk space at the cost of CPU time")
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
    if args.limit < 1 or (args.balanced and (args.limit < 2 or args.limit % 2)):
        parser.error("limit must be positive; balanced pilot needs an even limit of at least two")
    if args.stage == "prepare" and args.dataset is None:
        parser.error("prepare requires --dataset pointing to the official RAGTruth directory")
    if args.dataset is not None and args.stage in ("score", "compare", "evaluate"):
        parser.error("--dataset prepares new inputs: use --stage prepare or run, with a new output directory")
    return args


def save_settings(args, model_name, responses, cohort=None):
    settings = {
        "version": "native-support-v1", "model": args.model or model_name,
        "device": args.device, "dtype": args.dtype, "responses": responses,
        "prefill_chunk_size": args.prefill_chunk_size,
        "readout": "observed_token_vs_native_highest_other", "labels_used": False,
        "gradients": False, "interventions": False,
    }
    if cohort is not None:
        settings["cohort"] = cohort
    start_stage(args.output / "settings.json", settings, args.resume)
    return settings


def prepare(args):
    if args.dataset is None:
        model_name, responses = load_responses(args.input)
        return save_settings(args, model_name, responses)
    from .ragtruth import prepare_official
    model_name = args.model or read_json(args.input)["model"]
    manifest, annotations, cohort = prepare_official(args, model_name)
    settings = save_settings(args, model_name, manifest["responses"], cohort)
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


def main(argv=None):
    from .token_detection import evaluate_detection, run_detection

    args = arguments(argv)
    if args.stage == "evaluate":
        result = evaluate_detection(args.output, args.annotations)
    elif args.stage == "prepare":
        settings = prepare(args)
        result = {"status": "prepared", "responses": len(settings["responses"]),
                  "annotations": str(args.output / "annotations.json"), "model_run": False}
    elif args.stage == "compare":
        from .comparison import run_comparison
        result = run_comparison(args.output, read_json(args.output / "settings.json"), args.annotations)
    elif args.stage == "score":
        result = run_detection(args.output, read_json(args.output / "settings.json"), args.annotations)
    else:
        settings = prepare(args)
        run_capture(args, settings)
        result = run_detection(args.output, settings, args.annotations)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
