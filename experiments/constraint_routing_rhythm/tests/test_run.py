from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.constraint_routing_rhythm import run


def test_cli_has_only_the_three_foreground_commands() -> None:
    command_parser = run.parser()

    for command in ("analyze", "evaluate", "all"):
        assert command_parser.parse_args([command]).command == command
    with pytest.raises(SystemExit):
        command_parser.parse_args(["collect"])


def test_analysis_defaults_follow_the_method_and_historical_inputs() -> None:
    args = run.parser().parse_args(["analyze"])

    assert args.model == run.DEFAULT_MODEL
    assert args.cache == run.DEFAULT_CACHE
    assert args.source_info == run.DEFAULT_SOURCE_INFO
    assert args.dtype == "bfloat16"
    assert args.device == "cuda:0"
    assert args.audit_limit == 0
    assert args.audit_seed == 2026
    assert args.plot_limit == 4
    assert args.head_quantile == 0.3
    assert args.query_chunk == 128
    assert (args.window, args.horizon_low, args.horizon_high) == (10, 10, 100)
    assert args.carrier_quantile == 0.75
    assert args.mass_floor == 1e-6
    assert args.max_carriers == 8
    assert args.limit is None and args.max_events is None


def test_smoke_supplies_only_missing_limits(monkeypatch, tmp_path: Path) -> None:
    fake_model = object()
    fake_tokenizer = object()
    observed = {}

    def fake_load(path, device, dtype):
        observed["load"] = (path, device, dtype)
        return fake_model, fake_tokenizer

    def fake_analyze(model, tokenizer, **kwargs):
        observed["call"] = (model, tokenizer, kwargs)
        return {"QA": 1}

    monkeypatch.setattr(run, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(run, "analyze_split", fake_analyze)
    args = run.parser().parse_args(
        ["analyze", "--smoke", "--max-events", "5", "--output", str(tmp_path)]
    )

    counts = run.analyze_command(args)

    assert counts == {"QA": 1}
    assert observed["load"] == (run.DEFAULT_MODEL, "cuda:0", "bfloat16")
    model, tokenizer, kwargs = observed["call"]
    assert (model, tokenizer) == (fake_model, fake_tokenizer)
    assert kwargs["split_root"] == run.DEFAULT_CACHE / "test"
    assert kwargs["source_info"] == run.DEFAULT_SOURCE_INFO
    assert kwargs["output_root"] == tmp_path
    assert kwargs["limit"] == 1
    assert kwargs["max_events"] == 5
    assert kwargs["model_id"] == run.DEFAULT_MODEL.name
    assert kwargs["audit_seed"] == 2026
    assert kwargs["mass_floor"] == 1e-6
    assert kwargs["run_config"]["smoke"] is True
    assert kwargs["run_config"]["max_events"] == 5
    assert kwargs["run_config"]["dataset_root"] == str(
        (run.DEFAULT_CACHE / "test").resolve()
    )
    assert kwargs["run_config"]["mass_floor"] == 1e-6


def test_default_smoke_output_does_not_poison_full_results() -> None:
    full = run.parser().parse_args(["all"])
    smoke = run.parser().parse_args(["all", "--smoke"])

    assert run.output_root(smoke) == run.output_root(full) / "smoke"

    explicit_limit = run.parser().parse_args(["analyze", "--smoke", "--limit", "3"])
    assert run.smoke_limits(explicit_limit) == (3, 8)


def test_evaluate_uses_frozen_results_and_test_labels(
    monkeypatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    observed = {}

    def fake_evaluate(**kwargs):
        observed.update(kwargs)
        return {"QA": {"constraint_deficit": {"tokens": 3}}}

    monkeypatch.setattr(run, "evaluate_results", fake_evaluate)
    args = run.parser().parse_args(
        [
            "evaluate",
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--bootstrap",
            "17",
            "--seed",
            "19",
        ]
    )

    reports = run.evaluate_command(args)

    assert reports["QA"]["constraint_deficit"]["tokens"] == 3
    assert observed == {
        "result_root": output / "results",
        "dataset_root": cache / "test",
        "output_root": output / "reports",
        "bootstrap": 17,
        "seed": 19,
    }


def test_all_releases_cuda_between_analysis_and_evaluation(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(run, "analyze_command", lambda _args: events.append("analyze"))
    monkeypatch.setattr(run, "release_cuda", lambda: events.append("release"))
    monkeypatch.setattr(
        run,
        "evaluate_command",
        lambda _args: events.append("evaluate") or {"QA": {}},
    )

    result = run.all_command(run.parser().parse_args(["all"]))

    assert events == ["analyze", "release", "evaluate"]
    assert result == {"QA": {}}


def test_model_loader_is_local_and_honors_dtype_and_device(monkeypatch) -> None:
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"] = (path, kwargs)
            return FakeModel()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"] = (path, kwargs)
            return "tokenizer"

    monkeypatch.setattr(run, "AutoModelForCausalLM", FakeModelFactory)
    monkeypatch.setattr(run, "AutoTokenizer", FakeTokenizerFactory)

    model, tokenizer = run.load_model_and_tokenizer(
        Path("/models/llama"), "cuda:3", "float16"
    )

    assert isinstance(model, FakeModel) and tokenizer == "tokenizer"
    assert calls["model"] == (
        "/models/llama",
        {
            "local_files_only": True,
            "dtype": torch.float16,
            "attn_implementation": "eager",
        },
    )
    assert calls["tokenizer"] == (
        "/models/llama",
        {"local_files_only": True},
    )
    assert calls["device"] == "cuda:3" and calls["eval"] is True
