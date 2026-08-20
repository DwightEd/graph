from experiments.attention_phenomenology.main import (
    _config,
    _distribution_config,
    _head_experiment_config,
    _head_training_config,
    build_parser,
)


def test_cli_uses_only_current_mechanism_parameters():
    arguments = build_parser().parse_args(
        [
            "fit",
            "--train-split",
            "train",
            "--output",
            "reference.npz",
            "--null-prompt-position-bins",
            "6",
            "--null-response-lag-bins",
            "7",
            "--recent-response-tokens",
            "5",
            "--reference-minimum-scale",
            "0.01",
            "--maximum-standardized-value",
            "8",
        ]
    )

    config = _config(arguments)

    assert config.null_prompt_position_bins == 6
    assert config.null_response_lag_bins == 7
    assert config.recent_response_tokens == 5
    assert config.reference_minimum_scale == 0.01
    assert config.maximum_standardized_value == 8.0


def test_distribution_command_builds_its_own_validation_config():
    arguments = build_parser().parse_args(
        [
            "validate-distributions",
            "--fit-split",
            "train",
            "--validation-split",
            "validation",
            "--output-dir",
            "outputs",
            "--fit-reservoir-rows",
            "32",
            "--validation-reservoir-rows",
            "48",
            "--pseudocount",
            "0.0001",
        ]
    )

    config = _distribution_config(arguments)

    assert config.fit_reservoir_rows == 32
    assert config.validation_reservoir_rows == 48
    assert config.pseudocounts == (0.0001,)


def test_head_model_command_exposes_small_explicit_configs():
    arguments = build_parser().parse_args(
        [
            "train-head-model",
            "--train-split",
            "train",
            "--test-split",
            "test",
            "--output-dir",
            "outputs",
            "--reuse-top-k",
            "3",
            "--validation-fraction",
            "0.25",
            "--hidden-dim",
            "12",
            "--epochs",
            "7",
        ]
    )

    experiment = _head_experiment_config(arguments)
    training = _head_training_config(arguments)

    assert experiment.reuse_top_k == 3
    assert experiment.validation_fraction == 0.25
    assert training.hidden_dim == 12
    assert training.epochs == 7
