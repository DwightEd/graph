import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from experiments.head_resolved_shortcut_route import run


def metric(auroc: float | None = 0.5) -> dict:
    return {
        "auroc": auroc,
        "average_precision": None if auroc is None else 0.25,
        "auroc_ci95": [None, None],
        "average_precision_ci95": [None, None],
    }


def test_all_cli_exposes_only_supported_full_sequence_capture_options():
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
    assert args.top_k == 64
    assert args.cover_mass == 0.95
    assert not hasattr(args, "predictor_chunk")
    assert not hasattr(args, "logit_chunk")


def test_all_captures_once_then_evaluates_each_task(tmp_path, monkeypatch):
    shared = [
        (tmp_path / "shortcut_route_state/train", tmp_path / "cache/train"),
        (tmp_path / "shortcut_route_state/test", tmp_path / "cache/test"),
    ]
    captures = []
    evaluations = []

    def capture_all(**kwargs):
        captures.append(kwargs)
        return {task: shared for task in run.TASK_TYPES}

    def evaluate_all(*, inputs, task_type, output, bootstrap, seed, allow_partial):
        evaluations.append((inputs, task_type, output, bootstrap, seed, allow_partial))
        return {
            "task_type": task_type,
            "samples": 1,
            "sources": 1,
            "tokens": 2,
            "hallucinated_tokens": 1,
            "prevalence": 0.5,
            "detection": {name: metric() for name in run.SCORE_ORDER},
            "frozen_axes": "frozen_axes.npz",
        }

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
        top_k=64,
        cover_mass=0.95,
        bootstrap=10,
        seed=7,
    )
    run._all_command(args)

    assert len(captures) == 1
    assert captures[0]["split_roots"] == (
        tmp_path / "cache/train",
        tmp_path / "cache/test",
    )
    assert captures[0]["top_k"] == 64
    assert captures[0]["cover_mass"] == 0.95
    assert [call[1] for call in evaluations] == list(run.TASK_TYPES)
    assert all(call[0] == shared for call in evaluations)
    assert not any(call[5] for call in evaluations)
    assert [call[2] for call in evaluations] == [
        tmp_path / "results" / run.REPORT_DIRECTORY / task.casefold() / "report.json"
        for task in run.TASK_TYPES
    ]


def test_evaluate_command_pairs_state_and_label_shards(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run,
        "_evaluate_tasks",
        lambda **kwargs: calls.append(kwargs),
    )
    args = Namespace(
        task="QA",
        state_root=tmp_path / "state",
        cache=tmp_path / "cache",
        report_root=None,
        bootstrap=20,
        seed=9,
        allow_partial=True,
    )

    run._evaluate_command(args)

    assert calls[0]["tasks"] == ("QA",)
    assert calls[0]["inputs"] == [
        (
            tmp_path / "state" / run.STATE_DIRECTORY / "train",
            tmp_path / "cache" / "train",
        ),
        (
            tmp_path / "state" / run.STATE_DIRECTORY / "test",
            tmp_path / "cache" / "test",
        ),
    ]
    assert calls[0]["report_root"] == tmp_path / "state" / run.REPORT_DIRECTORY
    assert calls[0]["allow_partial"] is True


def test_report_prints_only_preregistered_axes(capsys):
    run._print_report(
        {
            "task_type": "Summary",
            "samples": 2,
            "sources": 1,
            "tokens": 3,
            "hallucinated_tokens": 1,
            "prevalence": 1 / 3,
            "detection": {
                name: metric(None if index == 2 else 0.5)
                for index, name in enumerate(run.SCORE_ORDER)
            },
        }
    )

    output = capsys.readouterr().out
    assert "COMPLETE-CAPTURE SUMMARY SHORTCUT-ROUTE ASSOCIATION" in output
    assert output.count("axis      ") == 3
    assert "negative_prompt_source_dispersion_support" in output
    assert "veto channels are raw audits only" in output
    assert "structural controls" in output
    assert "dual_register" not in output


def test_one_click_script_is_cwd_independent_and_keeps_tracebacks(tmp_path):
    script = Path(run.__file__).with_name("run_all.sh")
    completed = subprocess.run(
        [script, "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage: run.py all" in completed.stdout
    source = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" not in source
    assert "|| exit $?" in source
