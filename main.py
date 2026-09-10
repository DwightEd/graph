"""Command-line entry point for constraint-control graph anomaly detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from control_graph.audit import AttentionAuditScorer, AuditScoreConfig
from control_graph.audit_evaluation import (
    AttentionAuditEvaluator,
    AuditEvaluationConfig,
)
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

    audit_score = commands.add_parser(
        "audit-score", help="score completed attention-audit v3 traces without labels"
    )
    audit_score.add_argument("--audit", type=Path, required=True)
    audit_score.add_argument("--output", type=Path, required=True)
    audit_score.add_argument("--completed-only", action="store_true")

    audit_evaluate = commands.add_parser(
        "audit-evaluate", help="evaluate frozen audit scores with label sidecars"
    )
    audit_evaluate.add_argument("--audit", type=Path, required=True)
    audit_evaluate.add_argument("--scores", type=Path, required=True)
    audit_evaluate.add_argument("--output", type=Path, required=True)
    audit_evaluate.add_argument("--bootstrap", type=int, default=1000)
    audit_evaluate.add_argument("--seed", type=int, default=20260910)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build":
        result = BuildGraphDataset(BuildConfig(args.input, args.output)).run()
    elif args.command == "detect":
        result = DetectGraphAnomalies(
            DetectionConfig(args.graphs, args.output, args.fit_split, args.score_split)
        ).run()
    elif args.command == "evaluate":
        result = DetectionEvaluator(
            EvaluationConfig(args.scores, args.labels, args.output, args.bootstrap, args.seed)
        ).run()
    elif args.command == "audit-score":
        result = AttentionAuditScorer(
            AuditScoreConfig(args.audit, args.output, args.completed_only)
        ).run()
    else:
        result = AttentionAuditEvaluator(
            AuditEvaluationConfig(
                args.audit, args.scores, args.output, args.bootstrap, args.seed
            )
        ).run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
