"""Command-line entry point for constraint-control graph anomaly detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from control_graph.evaluation import DetectionEvaluator, EvaluationConfig
from control_graph.pipeline import (
    BuildConfig,
    BuildGraphDataset,
    DetectGraphAnomalies,
    DetectionConfig,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build causal control graphs")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    detect = commands.add_parser("detect", help="fit and score graph anomalies")
    detect.add_argument("--graphs", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--fit-split", default="train")
    detect.add_argument("--score-split", default="test")

    evaluate = commands.add_parser("evaluate", help="evaluate frozen scores")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260910)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build":
        result = BuildGraphDataset(BuildConfig(args.input, args.output)).run()
    elif args.command == "detect":
        result = DetectGraphAnomalies(
            DetectionConfig(args.graphs, args.output, args.fit_split, args.score_split)
        ).run()
    else:
        result = DetectionEvaluator(
            EvaluationConfig(args.scores, args.labels, args.output, args.bootstrap, args.seed)
        ).run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
