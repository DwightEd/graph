from pathlib import Path

from experiments.attention_mechanism_audit.run import DEFAULT_MODEL, parser


def test_capture_cli_uses_the_exact_frozen_model_path():
    args = parser().parse_args(
        [
            "capture",
            "--split-root",
            "cache/train",
            "--source-info",
            "source.jsonl",
            "--output",
            "traces",
        ]
    )

    assert args.model == Path(DEFAULT_MODEL)
    assert args.predictor_chunk == 128
    assert args.intervention_batch == 3


def test_evaluate_cli_accepts_multiple_physical_shards_once():
    args = parser().parse_args(
        [
            "evaluate",
            "--input",
            "train/traces",
            "cache/train",
            "--input",
            "test/traces",
            "cache/test",
            "--output",
            "report.json",
        ]
    )

    assert args.input == [
        ["train/traces", "cache/train"],
        ["test/traces", "cache/test"],
    ]
    assert args.output == Path("report.json")
    assert vars(args).keys().isdisjoint({"split_name", "combine", "probe", "epochs"})


def test_plot_sample_searches_the_same_saved_inputs():
    args = parser().parse_args(
        [
            "plot-sample",
            "--input",
            "train/traces",
            "cache/train",
            "--input",
            "test/traces",
            "cache/test",
            "--sample-id",
            "11907",
            "--output",
            "sample.png",
        ]
    )

    assert args.sample_id == "11907"
    assert args.output == Path("sample.png")
