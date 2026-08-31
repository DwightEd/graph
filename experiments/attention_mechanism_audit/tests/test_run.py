from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("torch")

import experiments.attention_mechanism_audit.run as run


def test_all_cli_uses_the_shared_cache_and_exact_frozen_model():
    args = run.parser().parse_args(
        [
            "all",
            "--cache",
            "cache",
            "--source-info",
            "source.jsonl",
            "--output",
            "results",
        ]
    )

    assert args.model == run.DEFAULT_MODEL
    assert args.cache == Path("cache")
    assert args.source_info == Path("source.jsonl")
    assert args.output == Path("results")
    assert args.device == "cuda:0"
    assert args.dtype == "bfloat16"
    assert args.limit is None


def test_all_command_evaluates_each_task_from_the_same_saved_traces(
    tmp_path, monkeypatch
):
    shared = [
        (tmp_path / "traces/train", tmp_path / "cache/train"),
        (tmp_path / "traces/test", tmp_path / "cache/test"),
    ]
    captures = []
    evaluations = []

    def capture_all(**kwargs):
        captures.append(kwargs)
        return {task: shared for task in run.TASK_TYPES}

    def evaluate_all(*, inputs, task_type, output, bootstrap, seed):
        evaluations.append((inputs, task_type, output, bootstrap, seed))
        return {"token_scores": "scores.npz", "figures": "figures"}

    monkeypatch.setattr(run, "capture_all", capture_all)
    monkeypatch.setattr(run, "evaluate_all", evaluate_all)
    monkeypatch.setattr(run, "_print_report", lambda *_args: None)

    args = Namespace(
        cache=tmp_path / "cache",
        source_info=tmp_path / "source.jsonl",
        model=tmp_path / "model",
        output=tmp_path / "results",
        device="cuda:0",
        dtype="bfloat16",
        limit=None,
        bootstrap=10,
        seed=7,
    )
    run._all(args)

    assert len(captures) == 1
    assert captures[0]["split_roots"] == (
        tmp_path / "cache/train",
        tmp_path / "cache/test",
    )
    assert [call[1] for call in evaluations] == list(run.TASK_TYPES)
    assert all(call[0] is shared for call in evaluations)
    assert [call[2] for call in evaluations] == [
        tmp_path / "results" / task.casefold() / "report.json"
        for task in run.TASK_TYPES
    ]


def test_plot_sample_searches_the_same_saved_inputs():
    args = run.parser().parse_args(
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
    run._print_report(
        {
            "samples": 1,
            "sources": 1,
            "tokens": 2,
            "hallucinated_tokens": 1,
            "prevalence": 0.5,
            "capture_complete": False,
            "task_type": "Summary",
            "primary_score": run.SCORE_ORDER[0],
            "detection": {name: metric for name in run.SCORE_ORDER},
        },
    )
    output = capsys.readouterr().out
    assert "PARTIAL-SUMMARY" in output
    assert output.count("CI=n/a") == 8
