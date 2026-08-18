"""CLI for the RR-only signal decomposition and collapse audit."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .components import RRSignalConfig
from .experiment import (
    evaluate_rr_signal_audit,
    fit_rr_signal_audit,
    score_rr_signal_audit,
)
from .geometry import RRGeometryConfig


def _signal_arguments(parser):
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lag-bins", type=int, default=8)
    parser.add_argument("--local-lag-max", type=int, default=4)
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--causal-position-bins", type=int, default=10)


def _geometry_arguments(parser):
    parser.add_argument("--relative-position-bins", type=int, default=4)
    parser.add_argument("--reservoir-rows", type=int, default=4096)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--min-condition-rows", type=int, default=32)
    parser.add_argument("--trim-fraction", type=float, default=0.90)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260818)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit",
        help="fit unlabeled RR decomposition references and coordination controls",
    )
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--limit", type=int)
    _signal_arguments(fit)
    _geometry_arguments(fit)

    score = commands.add_parser(
        "score",
        help="freeze held-out RR decomposition and collapse scores without labels",
    )
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate",
        help="open labels only after scores are frozen and write mechanism reports",
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--onset-window", type=int, default=4)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def _signal_config(args):
    return RRSignalConfig(
        top_k=args.top_k,
        lag_bins=args.lag_bins,
        local_lag_max=args.local_lag_max,
        anchor_count=args.anchor_count,
        block_rows=args.block_rows,
        causal_position_bins=args.causal_position_bins,
    )


def _geometry_config(args):
    return RRGeometryConfig(
        relative_position_bins=args.relative_position_bins,
        reservoir_rows=args.reservoir_rows,
        pca_dim=args.pca_dim,
        min_condition_rows=args.min_condition_rows,
        trim_fraction=args.trim_fraction,
        calibration_fraction=args.calibration_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split,
            device=args.device,
            verify_hashes=True,
        )
        result = fit_rr_signal_audit(
            dataset,
            args.output,
            signal_config=_signal_config(args),
            geometry_config=_geometry_config(args),
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=True,
        )
        result = score_rr_signal_audit(
            dataset,
            args.reference,
            args.output,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        report = evaluate_rr_signal_audit(
            dataset,
            args.scores,
            args.output_dir,
            onset_window=args.onset_window,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        result = {
            "output": f"{args.output_dir}/evaluation.json",
            "labels_read": True,
            "tokens": report["tokens"],
            "positive_tokens": report["positive_tokens"],
            "prevalence": report["prevalence"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
