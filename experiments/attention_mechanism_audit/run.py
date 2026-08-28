"""Command line entry point for the controlled grounding-mechanism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .data import load_pairs
from .evaluate import evaluate_artifact
from .replay import FrozenMarginReplay


DEFAULT_MODEL = (
    "/share/home/tm902089733300000/a903202310/lys/models/"
    "Meta-Llama-3.1-8B-Instruct"
)


def _torch_dtype(name: str):
    import torch

    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _run_audit(args: argparse.Namespace) -> None:
    pairs = load_pairs(args.pairs)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("the controlled pair manifest selected no rows")
    replay = FrozenMarginReplay.from_pretrained(
        args.model,
        device=args.device,
        torch_dtype=_torch_dtype(args.torch_dtype),
    )
    manifest = run_audit(
        _progress(pairs),
        replay,
        args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _progress(pairs):
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        print(f"audit pair {index}/{total}: {pair.sample_id}", flush=True)
        yield pair


def _statistic(name: str, value: dict[str, object]) -> None:
    if not value["available"]:
        print(f"{name:28s} unavailable")
        return
    print(
        f"{name:28s} "
        f"mean={value['source_equal_mean']:.6f} "
        f"95%CI=[{value['ci_low']:.6f}, {value['ci_high']:.6f}] "
        f"N={value['samples']} sources={value['sources']}"
    )


def _print_report(report: dict[str, object]) -> None:
    mechanisms = report["mechanisms"]
    select = mechanisms["select"]
    relay = mechanisms["relay"]
    override = mechanisms["override"]

    print("\n=== SELECT: total source-path effect ===")
    _statistic("relevant_gain", select["relevant_gain"])
    _statistic("select_contrast", select["select_contrast"])
    _statistic("select_success_rate", select["success_rate"])

    print("\n=== RELAY: select-success domain ===")
    _statistic("history_prior_support", relay["history_prior_support"])
    _statistic("history_evidence_relay", relay["history_evidence_relay"])
    _statistic("self_lock_rate", relay["self_lock_rate"])

    print("\n=== OVERRIDE: select-success domain ===")
    _statistic("question_prior_strength", override["question_prior_strength"])
    _statistic("prior_capture", override["prior_capture"])
    _statistic("capture_failure_rate", override["capture_failure_rate"])


def _run_evaluate(args: argparse.Namespace) -> None:
    report = evaluate_artifact(
        args.artifact,
        args.output,
        bootstrap_replicates=args.bootstrap,
        seed=args.seed,
    )
    _print_report(report)
    print(f"\nFull report: {Path(args.output)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen SELECT--RELAY--OVERRIDE causal audit"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="run the seven frozen replays")
    audit.add_argument("--pairs", type=Path, required=True)
    audit.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--device", default="cuda")
    audit.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    audit.add_argument("--limit", type=int)
    audit.set_defaults(handler=_run_audit)

    evaluate = commands.add_parser("evaluate", help="summarize fixed effects")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1_000)
    evaluate.add_argument("--seed", type=int, default=20260828)
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
