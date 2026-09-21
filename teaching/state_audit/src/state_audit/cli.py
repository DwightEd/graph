"""Argument parsing only; each command calls one named workflow."""

import argparse
import json
from pathlib import Path

from .analysis.audit import audit_run
from .dataset import convert_ragtruth
from .pipeline import audit_settings, capture_spec, generate_and_capture, load_run_model
from .storage import read_json


def generation_arguments(parser):
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--mode", choices=["generate", "replay"], default="generate")
    parser.add_argument("--template", choices=["chat", "raw"], default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1, help="Draws per distinct source/prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024, help="Hard limit; never truncate")
    parser.add_argument("--resume", action="store_true")


def audit_arguments(parser):
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--minimum-mass", type=float, default=0.25)
    parser.add_argument("--minimum-rise", type=float, default=0.15)
    parser.add_argument("--roles", action="store_true", help="Also run local pre-RoPE key swaps")


def parser():
    root = argparse.ArgumentParser(
        description="Resample, capture representations, intervene, and analyze"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("generate", "run"):
        command = commands.add_parser(name)
        generation_arguments(command)
        if name == "run":
            capture_arguments(command)
    capture = commands.add_parser("capture")
    capture.add_argument("--run", type=Path, required=True)
    capture_arguments(capture)
    capture.add_argument("--device")
    capture.add_argument("--resume", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit_arguments(audit)
    add_conversion(commands)
    add_intervention(commands)
    pair = commands.add_parser("pair")
    pair.add_argument("--run", type=Path, required=True)
    pair.add_argument("--reviews", type=Path)
    pair.add_argument("--output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--sample", type=int, default=0)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--family", choices=["llama", "mistral", "qwen2"], default="llama")
    demo.add_argument("--resume", action="store_true")
    return root


def add_conversion(commands):
    convert = commands.add_parser("convert-ragtruth")
    convert.add_argument("--data", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--split", choices=["train", "test"])
    convert.add_argument("--limit", type=int)


def capture_arguments(parser):
    from .model.sites import AXES

    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--representations", nargs="+", choices=list(AXES))
    parser.add_argument("--scope", choices=["response", "all"], default="response")


def add_intervention(commands):
    command = commands.add_parser("intervene")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--sample", type=int, default=0)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device")


def run_demo(args):
    from .demo import build_demo

    existing = args.output / "run.json"
    if existing.exists():
        if not args.resume or read_json(existing)["model_config"]["model_type"] != args.family:
            raise ValueError("Use --resume with the same model family, or a new output directory")
        data = args.output / "fixture" / "examples.jsonl"
        model = args.output / "fixture" / "model"
    else:
        data, model = build_demo(args.output / "fixture", args.family)
    command = [
        "run",
        "--data",
        str(data),
        "--model",
        str(model),
        "--output",
        str(args.output),
        "--mode",
        "replay",
        "--max-length",
        "256",
    ]
    if args.resume:
        command.append("--resume")
    generate_and_capture(parser().parse_args(command), full=True)
    settings = dict(window=4, minimum_mass=0.25, minimum_rise=0.15, roles=True)
    audit_run(args.output, args.output / "audit", settings)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("generate", "run"):
        generate_and_capture(args, full=args.command == "run")
    elif args.command == "capture":
        from .capture import capture_run

        capture_run(
            load_run_model(args.run, args.device), args.run, capture_spec(args), args.resume
        )
    elif args.command == "audit":
        audit_run(args.run, args.output or args.run / "audit", audit_settings(args))
    elif args.command == "convert-ragtruth":
        convert_ragtruth(args.data, args.output, args.split, args.limit)
    elif args.command == "intervene":
        from .experiments.interventions import run_plan

        result = run_plan(
            load_run_model(args.run, args.device), args.run, args.sample, args.plan, args.output
        )
        print(json.dumps(result, indent=2))
    elif args.command == "pair":
        from .pairing import import_reviews, pair_answers

        if args.reviews:
            import_reviews(args.run, args.reviews)
        result = pair_answers(args.run, args.output or args.run / "pairs.json")
        print(json.dumps(result, indent=2))
    elif args.command == "check":
        from .validation import check_trace

        print(json.dumps(check_trace(args.run, args.sample), indent=2))
    elif args.command == "demo":
        run_demo(args)
