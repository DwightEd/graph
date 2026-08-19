"""CLI for the causal token-level routing-basin detector."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .detector import DetectorConfig
from .experiment import evaluate_scores, fit_reference, score_dataset
from .routing import RoutingFeatureConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit an unlabeled routing reference")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--limit", type=int)
    fit.add_argument("--calibration-fraction", type=float, default=0.2)
    fit.add_argument("--ridge", type=float, default=1e-2)
    fit.add_argument("--smoothing-decay", type=float, default=0.9)
    fit.add_argument("--student-df", type=float, default=4.0)
    fit.add_argument("--threshold-quantile", type=float, default=0.95)
    _feature_arguments(fit)

    score = commands.add_parser("score", help="freeze held-out token scores")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate", help="open labels after scores are frozen"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _feature_arguments(parser):
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--prompt-bins", type=int, default=8)
    parser.add_argument("--lag-bins", type=int, default=6)
    parser.add_argument("--recent-lag-max", type=int, default=4)
    parser.add_argument("--anchor-run-cap", type=int, default=8)
    parser.add_argument("--operator-sketch-width", type=int, default=512)
    parser.add_argument("--block-rows", type=int, default=8192)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split, device=args.device, verify_hashes=True
        )
        result = fit_reference(
            dataset,
            args.output,
            detector_config=DetectorConfig(
                calibration_fraction=args.calibration_fraction,
                ridge=args.ridge,
                smoothing_decay=args.smoothing_decay,
                student_df=args.student_df,
                threshold_quantile=args.threshold_quantile,
            ),
            feature_config=RoutingFeatureConfig(
                window=args.window,
                prompt_bins=args.prompt_bins,
                lag_bins=args.lag_bins,
                recent_lag_max=args.recent_lag_max,
                anchor_run_cap=args.anchor_run_cap,
                operator_sketch_width=args.operator_sketch_width,
                block_rows=args.block_rows,
            ),
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root, device=args.device, verify_hashes=True
        )
        result = score_dataset(
            dataset, args.reference, args.output, limit=args.limit
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        report = evaluate_scores(dataset, args.scores, args.output_dir)
        result = {
            "report": f"{args.output_dir}/report.json",
            "labels_read": True,
            **report["primary"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
