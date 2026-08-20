"""Command-line interface for the attention phenomenology audit."""

from __future__ import annotations

import argparse

from .causal_head_model import TrainingConfig
from .config import PhenomenologyConfig
from .distribution_validation import (
    DistributionValidationConfig,
    validate_composition_distributions,
)
from .evaluation import evaluate_scores
from .experiment import fit_reference, score_split
from .head_model_experiment import (
    HeadModelExperimentConfig,
    HeadResolvedExperiment,
)
from .majorization_detector import MajorizationDetectorConfig
from .majorization_validation import run_majorization_validation


def _add_config(parser: argparse.ArgumentParser) -> None:
    defaults = PhenomenologyConfig()
    parser.add_argument(
        "--null-prompt-position-bins",
        type=int,
        default=defaults.null_prompt_position_bins,
    )
    parser.add_argument(
        "--null-response-lag-bins",
        type=int,
        default=defaults.null_response_lag_bins,
    )
    parser.add_argument(
        "--recent-response-tokens",
        type=int,
        default=defaults.recent_response_tokens,
    )
    parser.add_argument(
        "--causal-position-bins",
        type=int,
        default=defaults.causal_position_bins,
    )
    parser.add_argument(
        "--reference-minimum-scale",
        type=float,
        default=defaults.reference_minimum_scale,
    )
    parser.add_argument(
        "--maximum-standardized-value",
        type=float,
        default=defaults.maximum_standardized_value,
    )
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)


def _config(arguments) -> PhenomenologyConfig:
    return PhenomenologyConfig(
        null_prompt_position_bins=arguments.null_prompt_position_bins,
        null_response_lag_bins=arguments.null_response_lag_bins,
        recent_response_tokens=arguments.recent_response_tokens,
        causal_position_bins=arguments.causal_position_bins,
        reference_minimum_scale=arguments.reference_minimum_scale,
        maximum_standardized_value=arguments.maximum_standardized_value,
        block_rows=arguments.block_rows,
        random_seed=arguments.seed,
    )


def _distribution_config(arguments) -> DistributionValidationConfig:
    representations = (
        tuple(arguments.representation)
        if arguments.representation
        else ("role", "provenance")
    )
    return DistributionValidationConfig(
        representations=representations,
        fit_reservoir_rows=arguments.fit_reservoir_rows,
        validation_reservoir_rows=arguments.validation_reservoir_rows,
        minimum_group_rows=arguments.minimum_group_rows,
        pseudocounts=(
            tuple(arguments.pseudocount)
            if arguments.pseudocount
            else (1e-6, 1e-4, 1e-3)
        ),
        simulation_rows=arguments.simulation_rows,
        random_seed=arguments.seed,
    )


def _majorization_config(arguments) -> MajorizationDetectorConfig:
    return MajorizationDetectorConfig(
        history_decay=arguments.history_decay,
        majorization_tolerance=arguments.majorization_tolerance,
        fit_tokens_per_sample=arguments.fit_tokens_per_sample,
        minimum_scale=arguments.minimum_scale,
        maximum_standardized_value=arguments.maximum_standardized_value,
    )


def _head_experiment_config(arguments) -> HeadModelExperimentConfig:
    return HeadModelExperimentConfig(
        validation_fraction=arguments.validation_fraction,
        reuse_top_k=arguments.reuse_top_k,
        recent_response_tokens=arguments.recent_response_tokens,
        block_rows=arguments.block_rows,
        train_limit=arguments.train_limit,
        test_limit=arguments.test_limit,
        seed=arguments.seed,
    )


def _head_training_config(arguments) -> TrainingConfig:
    return TrainingConfig(
        hidden_dim=arguments.hidden_dim,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        dropout=arguments.dropout,
        forecast_weight=arguments.forecast_weight,
        patience=arguments.patience,
        maximum_standardized_value=arguments.maximum_standardized_value,
        seed=arguments.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit unlabeled routing references")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--reservoir-rows", type=int, default=2048)
    fit.add_argument("--limit", type=int)
    _add_config(fit)

    score = commands.add_parser("score", help="freeze test mechanism fields")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--limit", type=int)
    score.add_argument("--no-rewire", action="store_true")
    score.add_argument("--detail-sample-id", action="append", default=[])
    _add_config(score)

    evaluate = commands.add_parser(
        "evaluate", help="unlock labels for post-hoc hypothesis tests"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--score-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--onset-window", type=int, default=4)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260819)

    distributions = commands.add_parser(
        "validate-distributions",
        help="compare Dirichlet and logistic-normal fits without opening labels",
    )
    distributions.add_argument("--fit-split", required=True)
    distributions.add_argument("--validation-split", required=True)
    distributions.add_argument("--output-dir", required=True)
    distributions.add_argument("--device", default="cpu")
    distributions.add_argument(
        "--representation",
        action="append",
        choices=("role", "provenance"),
    )
    distributions.add_argument("--fit-reservoir-rows", type=int, default=1024)
    distributions.add_argument(
        "--validation-reservoir-rows", type=int, default=1024
    )
    distributions.add_argument("--minimum-group-rows", type=int, default=128)
    distributions.add_argument("--pseudocount", type=float, action="append")
    distributions.add_argument("--simulation-rows", type=int, default=4096)
    distributions.add_argument("--fit-limit", type=int)
    distributions.add_argument("--validation-limit", type=int)
    _add_config(distributions)

    majorization = commands.add_parser(
        "validate-majorization",
        help="validate causal majorization, Hill spectra, and route states",
    )
    majorization.add_argument("--train-split", required=True)
    majorization.add_argument("--test-split", required=True)
    majorization.add_argument("--output-dir", required=True)
    majorization.add_argument("--device", default="cpu")
    majorization.add_argument("--history-decay", type=float, default=0.9)
    majorization.add_argument(
        "--majorization-tolerance", type=float, default=1e-6
    )
    majorization.add_argument("--fit-tokens-per-sample", type=int, default=128)
    majorization.add_argument("--minimum-scale", type=float, default=0.01)
    majorization.add_argument(
        "--maximum-standardized-value", type=float, default=10.0
    )
    majorization.add_argument("--block-rows", type=int, default=8192)
    majorization.add_argument("--fit-limit", type=int)
    majorization.add_argument("--test-limit", type=int)
    majorization.add_argument("--bootstrap-replicates", type=int, default=200)
    majorization.add_argument("--seed", type=int, default=20260820)

    head_model = commands.add_parser(
        "train-head-model",
        help="train a head-preserving layer and causal-time token detector",
    )
    head_model.add_argument("--train-split", required=True)
    head_model.add_argument("--test-split", required=True)
    head_model.add_argument("--output-dir", required=True)
    head_model.add_argument("--device", default="cpu")
    head_model.add_argument("--validation-fraction", type=float, default=0.2)
    head_model.add_argument("--reuse-top-k", type=int, default=5)
    head_model.add_argument("--recent-response-tokens", type=int, default=4)
    head_model.add_argument("--block-rows", type=int, default=8192)
    head_model.add_argument("--train-limit", type=int)
    head_model.add_argument("--test-limit", type=int)
    head_model.add_argument("--hidden-dim", type=int, default=16)
    head_model.add_argument("--epochs", type=int, default=20)
    head_model.add_argument("--batch-size", type=int, default=2)
    head_model.add_argument("--learning-rate", type=float, default=1e-3)
    head_model.add_argument("--weight-decay", type=float, default=1e-4)
    head_model.add_argument("--dropout", type=float, default=0.0)
    head_model.add_argument("--forecast-weight", type=float, default=0.5)
    head_model.add_argument("--patience", type=int, default=5)
    head_model.add_argument(
        "--maximum-standardized-value", type=float, default=10.0
    )
    head_model.add_argument("--seed", type=int, default=20260820)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "fit":
        fit_reference(
            train_split=arguments.train_split,
            output=arguments.output,
            device=arguments.device,
            config=_config(arguments),
            reservoir_rows=arguments.reservoir_rows,
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        score_split(
            split_root=arguments.split_root,
            reference_path=arguments.reference,
            output_dir=arguments.output_dir,
            device=arguments.device,
            config=_config(arguments),
            rewire=not arguments.no_rewire,
            detail_sample_ids=tuple(arguments.detail_sample_id),
            limit=arguments.limit,
        )
    elif arguments.command == "validate-distributions":
        validate_composition_distributions(
            fit_split=arguments.fit_split,
            validation_split=arguments.validation_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            phenomenology_config=_config(arguments),
            validation_config=_distribution_config(arguments),
            fit_limit=arguments.fit_limit,
            validation_limit=arguments.validation_limit,
        )
    elif arguments.command == "evaluate":
        evaluate_scores(
            split_root=arguments.split_root,
            score_dir=arguments.score_dir,
            output_dir=arguments.output_dir,
            onset_window=arguments.onset_window,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
    elif arguments.command == "validate-majorization":
        result = run_majorization_validation(
            train_split=arguments.train_split,
            test_split=arguments.test_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
            detector_config=_majorization_config(arguments),
            block_rows=arguments.block_rows,
            fit_limit=arguments.fit_limit,
            test_limit=arguments.test_limit,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
        print(f"done: {arguments.output_dir}/evaluation.json")
        print(
            "current AUROC:",
            result["current_detection"]["auroc"],
            "next-token AUROC:",
            result["forecast"]["horizon_1"]["auroc"],
        )
    else:
        result = HeadResolvedExperiment(
            experiment_config=_head_experiment_config(arguments),
            training_config=_head_training_config(arguments),
        ).run(
            train_split=arguments.train_split,
            test_split=arguments.test_split,
            output_dir=arguments.output_dir,
            device=arguments.device,
        )
        print(f"done: {arguments.output_dir}/evaluation.json")
        print(
            "test AUROC:",
            result["test"]["current"]["auroc"],
            "next-token AUROC:",
            result["test"]["forecast_1"]["auroc"],
        )


if __name__ == "__main__":
    main()
