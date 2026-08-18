"""CLI for Mechanism-Guided Causal Attention Set-Flow."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .calibration import CalibrationConfig
from .config import (
    CorruptionConfig,
    SetFlowModelConfig,
    SourceSetConfig,
    TrainingConfig,
)
from .experiment import evaluate_setflow, fit_setflow, score_setflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="label-free mechanism-guided training")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--limit", type=int)
    _add_source_args(fit)
    _add_model_args(fit)
    _add_corruption_args(fit)
    _add_training_args(fit)
    fit.add_argument("--min-condition-rows", type=int, default=32)

    score = commands.add_parser("score", help="freeze held-out learned energies")
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
    parser.add_argument("--materialize-query-chunk-size", type=int, default=64)


def _add_model_args(parser):
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--scalar-fourier-dim", type=int, default=16)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument("--induced-points", type=int, default=8)
    parser.add_argument("--set-blocks", type=int, default=2)
    parser.add_argument("--head-mixer-layers", type=int, default=2)
    parser.add_argument("--depth-mixer-layers", type=int, default=2)
    parser.add_argument("--energy-hidden-multiplier", type=int, default=2)
    parser.add_argument("--projector-hidden-multiplier", type=int, default=2)
    parser.add_argument("--set-row-chunk-size", type=int, default=4096)
    parser.add_argument("--mixer-token-chunk-size", type=int, default=512)
    parser.add_argument("--disable-activation-checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)


def _add_corruption_args(parser):
    parser.add_argument("--token-span-min", type=int, default=4)
    parser.add_argument("--token-span-max", type=int, default=24)
    parser.add_argument("--layer-span-min", type=int, default=4)
    parser.add_argument("--layer-span-max", type=int, default=12)
    parser.add_argument("--selected-head-fraction", type=float, default=0.50)
    parser.add_argument("--collapse-power", type=float, default=4.0)
    parser.add_argument("--self-reinforce-power", type=float, default=2.0)
    parser.add_argument("--locality-window", type=int, default=4)
    parser.add_argument("--corruption-margin", type=float, default=1.0)
    parser.add_argument("--clean-keep-fraction", type=float, default=0.90)


def _add_training_args(parser):
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--clean-energy-weight", type=float, default=1.0)
    parser.add_argument("--corrupt-energy-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--type-weight", type=float, default=0.50)
    parser.add_argument("--clean-recovery-weight", type=float, default=1.0)
    parser.add_argument("--context-recovery-weight", type=float, default=0.50)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--disable-cuda-memory-profile", action="store_true")
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
                materialize_query_chunk_size=args.materialize_query_chunk_size,
            ),
            model_config=SetFlowModelConfig(
                hidden_dim=args.hidden_dim,
                scalar_fourier_dim=args.scalar_fourier_dim,
                set_heads=args.set_heads,
                induced_points=args.induced_points,
                set_blocks=args.set_blocks,
                head_mixer_layers=args.head_mixer_layers,
                depth_mixer_layers=args.depth_mixer_layers,
                energy_hidden_multiplier=args.energy_hidden_multiplier,
                projector_hidden_multiplier=args.projector_hidden_multiplier,
                set_row_chunk_size=args.set_row_chunk_size,
                mixer_token_chunk_size=args.mixer_token_chunk_size,
                activation_checkpointing=not args.disable_activation_checkpointing,
                dropout=args.dropout,
            ),
            corruption_config=CorruptionConfig(
                token_span_min=args.token_span_min,
                token_span_max=args.token_span_max,
                layer_span_min=args.layer_span_min,
                layer_span_max=args.layer_span_max,
                selected_head_fraction=args.selected_head_fraction,
                collapse_power=args.collapse_power,
                self_reinforce_power=args.self_reinforce_power,
                locality_window=args.locality_window,
                margin=args.corruption_margin,
                clean_keep_fraction=args.clean_keep_fraction,
            ),
            training_config=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation=args.gradient_accumulation,
                gradient_clip_norm=args.gradient_clip_norm,
                calibration_fraction=args.calibration_fraction,
                ema_momentum=args.ema_momentum,
                clean_energy_weight=args.clean_energy_weight,
                corrupt_energy_weight=args.corrupt_energy_weight,
                ranking_weight=args.ranking_weight,
                type_weight=args.type_weight,
                clean_recovery_weight=args.clean_recovery_weight,
                context_recovery_weight=args.context_recovery_weight,
                variance_weight=args.variance_weight,
                covariance_weight=args.covariance_weight,
                precision=args.precision,
                profile_cuda_memory=not args.disable_cuda_memory_profile,
                seed=args.seed,
            ),
            calibration_config=CalibrationConfig(
                min_condition_rows=args.min_condition_rows
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