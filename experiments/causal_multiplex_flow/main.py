"""Command-line interface for Causal Multiplex Routing Prediction."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .events import EventConfig
from .experiment import TrainConfig, evaluate_cmrp, fit_cmrp, score_cmrp
from .model import ModelConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="train and calibrate CMRP without labels")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--limit", type=int)
    fit.add_argument("--block-rows", type=int, default=8192)
    fit.add_argument("--max-prompt-events", type=int, default=16)
    fit.add_argument("--max-rr-events", type=int, default=32)
    fit.add_argument("--hidden-dim", type=int, default=64)
    fit.add_argument("--channel-embedding-dim", type=int, default=8)
    fit.add_argument("--relation-embedding-dim", type=int, default=4)
    fit.add_argument("--lag-frequencies", type=int, default=4)
    fit.add_argument("--negatives", type=int, default=8)
    fit.add_argument("--dropout", type=float, default=0.10)
    fit.add_argument("--weight-loss-weight", type=float, default=0.10)
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-5)
    fit.add_argument("--gradient-clip", type=float, default=1.0)
    fit.add_argument("--calibration-fraction", type=float, default=0.25)
    fit.add_argument("--seed", type=int, default=20260817)

    score = commands.add_parser("score", help="freeze label-free CMRP test scores")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate", help="open labels after the score artifact is frozen"
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
            args.train_split,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=False,
        )
        result = fit_cmrp(
            dataset,
            args.output_dir,
            event_config=EventConfig(
                block_rows=args.block_rows,
                max_prompt_events_per_token=args.max_prompt_events,
                max_rr_events_per_token=args.max_rr_events,
            ),
            model_config=ModelConfig(
                hidden_dim=args.hidden_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                relation_embedding_dim=args.relation_embedding_dim,
                lag_frequencies=args.lag_frequencies,
                negatives_per_edge=args.negatives,
                dropout=args.dropout,
                weight_loss_weight=args.weight_loss_weight,
                seed=args.seed,
            ),
            train_config=TrainConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip=args.gradient_clip,
                calibration_fraction=args.calibration_fraction,
                seed=args.seed,
            ),
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=False,
        )
        result = score_cmrp(
            dataset,
            args.reference,
            args.output,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=True,
        )
        report = evaluate_cmrp(dataset, args.scores, args.output)
        result = {
            "output": args.output,
            "labels_read": True,
            **(report["metrics"] or {}),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
