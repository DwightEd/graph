from experiments.attention_phenomenology.main import _config, build_parser


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
