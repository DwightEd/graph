"""CLI for the learnable causal attention Set-Flow experiment."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .calibration import CalibrationConfig
from .config import SetFlowModelConfig, SourceSetConfig, TrainingConfig
from .experiment import evaluate_setflow, fit_setflow, score_setflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="label-free masked Set-Flow training")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--limit", type=int)
    _add_source_args(fit)
    _add_model_args(fit)
    _add_training_args(fit)
    fit.add_argument("--min-condition-rows", type=int, default=32)

    score = commands.add_parser("score", help="freeze held-out Set-Flow scores")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser("evaluate", help="post-hoc token-label evaluation")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _add_source_args(parser):
    parser.add_argument("--max-route-sources", type=int, default=32)
    parser.add_argument("--max-memory-sources", type=int, default=16)
    parser.add_argument("--route-mass-coverage", type=float, default=0.98)
    parser.add_argument("--block-rows", type=int, default=8192)


def _add_model_args(parser):
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--scalar-fourier-dim", type=int, default=16)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument("--induced-points", type=int, default=8)
    parser.add_argument("--set-blocks", type=int, default=2)
    parser.add_argument("--head-mixer-layers", type=int, default=2)
    parser.add_argument("--depth-mixer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--element-mask-probability", type=float, default=0.20)
    parser.add_argument("--head-mask-probability", type=float, default=0.20)
    parser.add_argument("--layer-mask-probability", type=float, default=0.15)


def _add_training_args(parser):
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--reference-per-sample", type=int, default=8)
    parser.add_argument("--latent-trim-fraction", type=float, default=0.90)
    parser.add_argument("--deterministic-masks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(args.train_split, device=args.device)
        result = fit_setflow(
            dataset,
            args.output_dir,
            source_config=SourceSetConfig(
                max_route_sources=args.max_route_sources,
                max_memory_sources=args.max_memory_sources,
                route_mass_coverage=args.route_mass_coverage,
                block_rows=args.block_rows,
            ),
            model_config=SetFlowModelConfig(
                hidden_dim=args.hidden_dim,
                scalar_fourier_dim=args.scalar_fourier_dim,
                set_heads=args.set_heads,
                induced_points=args.induced_points,
                set_blocks=args.set_blocks,
                head_mixer_layers=args.head_mixer_layers,
                depth_mixer_layers=args.depth_mixer_layers,
                dropout=args.dropout,
                element_mask_probability=args.element_mask_probability,
                head_mask_probability=args.head_mask_probability,
                layer_mask_probability=args.layer_mask_probability,
            ),
            training_config=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation=args.gradient_accumulation,
                gradient_clip_norm=args.gradient_clip_norm,
                calibration_fraction=args.calibration_fraction,
                reference_per_sample=args.reference_per_sample,
                latent_trim_fraction=args.latent_trim_fraction,
                deterministic_masks=args.deterministic_masks,
                seed=args.seed,
            ),
            calibration_config=CalibrationConfig(
                min_condition_rows=args.min_condition_rows,
                latent_trim_fraction=args.latent_trim_fraction,
            ),
            device=args.device,
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(args.split_root, device=args.device)
        result = score_setflow(
            dataset,
            args.reference,
            args.output,
            device=args.device,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            retain_embedded_labels=True,
        )
        report = evaluate_setflow(dataset, args.scores, args.output)
        result = {
            "output": args.output,
            "labels_read": True,
            **(report["metrics"] or {}),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
