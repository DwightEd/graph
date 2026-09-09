"""Command adapter for the counterfactual last-crossing mediation study."""

import argparse
from pathlib import Path

from .counterfactual_study import CounterfactualConfig, CounterfactualStudy


def _doses(value):
    try:
        doses = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("doses must be comma-separated numbers") from exc
    if not doses:
        raise argparse.ArgumentTypeError("at least two doses are required")
    return doses


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--pairs", type=Path, required=True)
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--phase", choices=CounterfactualStudy.PHASES, default="all"
    )
    command.add_argument("--device", default="cuda:0")
    command.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument("--window", type=int, default=10)
    command.add_argument("--gain", type=float, default=0.10)
    command.add_argument("--local-floor", type=float, default=0.50)
    command.add_argument("--event-row-budget", type=int, default=2)
    command.add_argument("--carrier-budget", type=int, default=1)
    command.add_argument("--doses", type=_doses, default=(0.0, 0.5, 1.0))
    return command


def run(args):
    config = CounterfactualConfig(
        pairs=args.pairs,
        model=args.model,
        output=args.output,
        device=args.device,
        dtype=args.dtype,
        query_chunk=args.query_chunk,
        window=args.window,
        gain=args.gain,
        local_floor=args.local_floor,
        event_row_budget=args.event_row_budget,
        carrier_budget=args.carrier_budget,
        doses=args.doses,
    )
    return CounterfactualStudy(config).run(args.phase)
