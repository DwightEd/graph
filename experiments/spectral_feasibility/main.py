"""CLI for the fully unsupervised RR spectral-subspace experiment."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .experiment import (
    evaluate_score_artifact,
    fit_spectral_reference,
    score_spectral_dataset,
)
from .representations import SpectralConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit", help="fit the label-free RR-only robust spectral reference"
    )
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--top-k", type=int, default=5)
    fit.add_argument("--position-bins", type=int, default=4)
    fit.add_argument("--pca-dim", type=int, default=32)
    fit.add_argument("--reference-per-sample", type=int, default=6)
    fit.add_argument("--trim-fraction", type=float, default=0.90)
    fit.add_argument("--calibration-fraction", type=float, default=0.25)
    fit.add_argument("--split-seed", type=int, default=20260815)
    fit.add_argument("--channel-tail-fraction", type=float, default=0.05)
    fit.add_argument("--attribution-topk", type=int, default=8)
    fit.add_argument("--block-rows", type=int, default=8192)
    fit.add_argument("--limit", type=int)

    score = commands.add_parser(
        "score", help="score RR spectral escape from the frozen reference"
    )
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate", help="open token labels only after RR scores are frozen"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split, device=args.device, verify_hashes=True
        )
        config = SpectralConfig(
            top_k=args.top_k,
            block_rows=args.block_rows,
            position_bins=args.position_bins,
            pca_dim=args.pca_dim,
            reference_per_sample=args.reference_per_sample,
            trim_fraction=args.trim_fraction,
            calibration_fraction=args.calibration_fraction,
            split_seed=args.split_seed,
            channel_tail_fraction=args.channel_tail_fraction,
            attribution_topk=args.attribution_topk,
        )
        result = fit_spectral_reference(
            dataset,
            args.output,
            config=config,
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root, device=args.device, verify_hashes=True
        )
        result = score_spectral_dataset(
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
        report = evaluate_score_artifact(dataset, args.scores, args.output)
        result = {
            "output": args.output,
            **report["metrics"],
            "components": report["components"],
            "labels_read": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
