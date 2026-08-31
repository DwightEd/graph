from pathlib import Path

import pytest

pytest.importorskip("torch")

from experiments.attention_mechanism_audit.run import (
    DEFAULT_MODEL,
    SCORE_ORDER,
    _print_report,
    parser,
)


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
    assert "trace_level" not in vars(args)


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
            "--input",
            "test/traces",
            "--sample-id",
            "11907",
            "--output",
            "sample.png",
        ]
    )

    assert args.sample_id == "11907"
    assert args.input == [Path("train/traces"), Path("test/traces")]
    assert args.output == Path("sample.png")


def test_report_prints_without_bootstrap_intervals(capsys):
    metric = {
        "auroc": 0.5,
        "auprc": 0.2,
        "auprc_lift": 1.0,
        "auroc_ci95": [None, None],
        "auprc_ci95": [None, None],
    }
    _print_report(
        {
            "samples": 1,
            "sources": 1,
            "tokens": 2,
            "hallucinated_tokens": 1,
            "prevalence": 0.5,
            "capture_complete": False,
            "primary_score": SCORE_ORDER[0],
            "detection": {name: metric for name in SCORE_ORDER},
        }
    )
    output = capsys.readouterr().out
    assert "PARTIAL-QA" in output
    assert output.count("CI=n/a") == 8
