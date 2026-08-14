"""CLI for label-blind spectral attention-graph feasibility experiments."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .experiment import (
    collect_representations,
    evaluate_score_artifact,
    save_representation_artifact,
    score_representation_artifacts,
)
from .representations import SpectralConfig


def _scales(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one heat scale is required")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser(
        "extract",
        help="build spectral token vectors without labels",
    )
    extract.add_argument("--split-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--sample-id", action="append")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--heat-scales", type=_scales, default=(0.25, 0.5, 1.0, 2.0, 4.0))
    extract.add_argument("--block-rows", type=int, default=4096)

    score = commands.add_parser(
        "score",
        help="fit an unlabeled train reference and score frozen test vectors",
    )
    score.add_argument("--train-features", required=True)
    score.add_argument("--test-features", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--trim-fraction", type=float, default=0.90)
    score.add_argument("--ridge", type=float, default=1e-3)

    evaluate = commands.add_parser(
        "evaluate",
        help="open labels only after spectral scores are frozen",
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "extract":
        dataset = open_research_dataset(args.split_root, device=args.device)
        config = SpectralConfig(
            heat_scales=tuple(args.heat_scales),
            block_rows=args.block_rows,
        )
        artifact = collect_representations(
            dataset,
            config=config,
            sample_ids=args.sample_id,
            limit=args.limit,
        )
        save_representation_artifact(artifact, args.output)
        result = {
            "output": args.output,
            "samples": int(artifact["sample_count"]),
            "tokens": int(len(artifact["features"])),
            "feature_dim": int(artifact["features"].shape[1]),
            "labels_read": False,
        }
    elif args.command == "score":
        result = score_representation_artifacts(
            args.train_features,
            args.test_features,
            args.output,
            trim_fraction=args.trim_fraction,
            ridge=args.ridge,
        )
        result["labels_read"] = False
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            retain_embedded_labels=True,
        )
        report = evaluate_score_artifact(dataset, args.scores, args.output)
        result = {
            "output": args.output,
            **report["metrics"],
            "labels_read": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
