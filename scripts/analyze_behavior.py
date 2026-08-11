#!/usr/bin/env python3
"""Run original-threshold token behavior case studies or onset alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior_analysis import BehaviorAnalysis


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-root", type=Path, required=True, help="Canonical attention split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, help="Verified original-threshold graph split")
    parser.add_argument("--tau", type=float, help="Threshold for on-the-fly original graph construction")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single", help="Analyze one response")
    _common(single)
    single.add_argument("--sample-id", required=True)
    single.add_argument("--control-sample-id")
    single.add_argument("--pre-window", type=int, default=8)
    single.add_argument("--post-window", type=int, default=8)
    align = commands.add_parser("align", help="Exploratory label-conditioned onset alignment")
    _common(align)
    align.add_argument("--radius", type=int, default=12)
    align.add_argument("--run-policy", choices=("first", "all"), default="first")
    align.add_argument("--no-controls", action="store_true")
    align.add_argument("--max-events", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis = BehaviorAnalysis(args.split_root, args.output_dir, args.graph_root, args.tau)
    if args.command == "single":
        result = analysis.single(args.sample_id, control_sample_id=args.control_sample_id, pre_window=args.pre_window, post_window=args.post_window)
    else:
        result = analysis.align(radius=args.radius, run_policy=args.run_policy, controls=not args.no_controls, max_events=args.max_events)
    print({"output_dir": str(args.output_dir), **result})


if __name__ == "__main__":
    main()
