"""Command-line boundary for fit, score, and post-hoc evaluation."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .config import (
    CalibrationConfig,
    ChangeConfig,
    DeBruijnConfig,
    GraphConfig,
)
from .evaluation import evaluate_scores
from .experiment import (
    ExperimentConfig,
    fit_reference,
    score_split,
    visualize_scored_sample,
)
from .spectral_bridge import build_rr_hybrid


def _sample_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample-id",
        action="append",
        help="exact sample ID; repeat to select several samples",
    )
    parser.add_argument("--limit", type=int)


def _fit_configuration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--recent-lag", type=int, default=4)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--soft-top-k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cusum-slack", type=float, default=0.5)
    parser.add_argument(
        "--prompt-lineage-drop-weight",
        type=float,
        default=0.0,
        help="prompt-lineage diagnostic is excluded from the score by default",
    )
    parser.add_argument("--rupture-decay", type=float, default=0.95)
    parser.add_argument("--feedback-ema-decay", type=float, default=0.9)
    parser.add_argument("--scale-floor", type=float, default=1e-3)
    parser.add_argument("--channel-fraction", type=float, default=0.2)
    parser.add_argument("--fusion-fraction", type=float, default=0.2)
    parser.add_argument("--reference-size", type=int, default=12000)
    parser.add_argument("--top-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)


def _experiment_config(args) -> ExperimentConfig:
    return ExperimentConfig(
        graph=GraphConfig(
            block_rows=args.block_rows,
            recent_lag=args.recent_lag,
        ),
        debruijn=DeBruijnConfig(
            order=args.order,
            soft_top_k=args.soft_top_k,
            alpha=args.alpha,
        ),
        change=ChangeConfig(
            cusum_slack=args.cusum_slack,
            prompt_lineage_drop_weight=args.prompt_lineage_drop_weight,
            rupture_decay=args.rupture_decay,
            feedback_ema_decay=args.feedback_ema_decay,
            scale_floor=args.scale_floor,
        ),
        calibration=CalibrationConfig(
            channel_fraction=args.channel_fraction,
            fusion_fraction=args.fusion_fraction,
            reference_size=args.reference_size,
            top_channels=args.top_channels,
            seed=args.seed,
        ),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Causal typed-path De Bruijn hallucination routing"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit a label-free reference")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--reference", required=True)
    fit.add_argument("--device", default="cpu")
    _sample_selection(fit)
    _fit_configuration(fit)

    score = commands.add_parser("score", help="freeze held-out token scores")
    score.add_argument("--test-split", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--save-channel-sidecars", action="store_true")
    score.add_argument("--sidecar-dir")
    _sample_selection(score)

    evaluate = commands.add_parser(
        "evaluate",
        help="open labels only after the score artifact is frozen",
    )
    evaluate.add_argument("--test-split", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=42)

    visualize = commands.add_parser(
        "visualize",
        help="render one sample without opening labels",
    )
    visualize.add_argument("--test-split", required=True)
    visualize.add_argument("--reference", required=True)
    visualize.add_argument("--scores", required=True)
    visualize.add_argument("--sample-id", required=True)
    visualize.add_argument("--output", required=True)
    visualize.add_argument("--token-index", type=int)
    visualize.add_argument("--device", default="cpu")

    hybrid = commands.add_parser(
        "hybrid",
        help="combine path score with the one allowed causal RR residual",
    )
    hybrid.add_argument("--path-scores", required=True)
    hybrid.add_argument("--rr-scores", required=True)
    hybrid.add_argument("--output", required=True)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        result = fit_reference(
            dataset,
            args.reference,
            config=_experiment_config(args),
            sample_ids=args.sample_id,
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.test_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        result = score_split(
            dataset,
            args.reference,
            args.output,
            sample_ids=args.sample_id,
            limit=args.limit,
            save_channel_sidecars=args.save_channel_sidecars,
            sidecar_dir=args.sidecar_dir,
        )
    elif args.command == "evaluate":
        dataset = open_research_dataset(
            args.test_split,
            device="cpu",
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        result = evaluate_scores(
            dataset,
            args.scores,
            args.output,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    elif args.command == "visualize":
        dataset = open_research_dataset(
            args.test_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        result = visualize_scored_sample(
            dataset,
            args.reference,
            args.scores,
            sample_id=args.sample_id,
            output_path=args.output,
            token_index=args.token_index,
        )
    elif args.command == "hybrid":
        result = build_rr_hybrid(
            args.path_scores,
            args.rr_scores,
            args.output,
        )
    else:  # pragma: no cover - argparse makes this unreachable.
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
