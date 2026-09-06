from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from experiments.reanchor_flow.flow import FlowSignal
from experiments.reanchor_flow.route_plan import RouteBudget
from experiments.reanchor_flow.subset_data import SampleRecord
from experiments.reanchor_flow.worlds import TargetContrast


class RecordingTqdm:
    instances: ClassVar[list[RecordingTqdm]] = []
    writes: ClassVar[list[str]] = []

    def __init__(self, iterable, **options) -> None:
        self.iterable = iterable
        self.options = options
        self.completed = 0
        self.postfix: list[str] = []
        self.instances.append(self)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.completed += 1

    def set_postfix_str(self, value: str, **_options) -> None:
        self.postfix.append(value)

    @classmethod
    def write(cls, value: str) -> None:
        cls.writes.append(value)


def _config(tmp_path):
    from experiments.reanchor_flow.subset import SubsetRunConfig

    return SubsetRunConfig(
        model_id="model",
        model_dtype="float32",
        tokenizer_id="tokenizer",
        dataset_root=str(tmp_path / "cache"),
        source_info=str(tmp_path / "source.jsonl"),
        split="test",
        tasks=("QA",),
        samples_per_task=1,
        explicit_sample_ids=(),
        selection_seed=2026,
        targets_per_sample=1,
        target_policy="uncertain",
        max_response_tokens=128,
        signal=FlowSignal.MESSAGE,
        carrier_scope="response",
        coverage=0.9,
        query_chunk=8,
        route_budget=RouteBudget(),
        local_window=10,
    )


def test_subset_cli_defaults_to_budgeted_discovery() -> None:
    from experiments.reanchor_flow.run import (
        parser,
        route_plan_summary,
        subset_config_from_args,
        subset_output_root,
    )

    args = parser().parse_args(["subset"])
    config = subset_config_from_args(
        args,
        SimpleNamespace(name_or_path="tiny-llama"),
        "test",
    )

    assert args.carrier_scope == "response"
    assert config.route_budget == RouteBudget(
        edges_per_head=2,
        max_rows=256,
        root_candidates=4,
        hub_candidates=8,
        corridor_edges=64,
        confirm=False,
    )
    assert subset_output_root(args).name == "native_mechanism_v3"
    summary = route_plan_summary(args)
    assert "mode=candidate-discovery" in summary
    assert "edges_per_head=2" in summary
    assert "max_route_rows=256" in summary


def test_top_level_help_names_the_current_renderable_artifact_schema() -> None:
    from experiments.reanchor_flow.run import parser

    assert "schema-3" in parser().format_help()


def test_audit_all_defaults_to_all_samples_full_responses_and_bounded_targets() -> None:
    from experiments.reanchor_flow.run import (
        parser,
        selected_splits,
        subset_config_from_args,
        validate_args,
    )

    args = parser().parse_args(["audit-all"])
    validate_args(args)
    assert selected_splits(args) == ("train", "test")
    assert args.samples_per_task == 0
    assert args.targets_per_sample == 3
    assert args.target_policy == "reanchor-window"
    assert args.evaluate
    config = subset_config_from_args(
        args, SimpleNamespace(name_or_path="model"), "test"
    )
    assert config.max_response_tokens is None
    assert not config.route_budget.confirm

    scan_args = parser().parse_args(["audit-all", "--scan-only"])
    scan_config = subset_config_from_args(
        scan_args, SimpleNamespace(name_or_path="model"), "test"
    )
    assert scan_config.scan_only
    assert scan_config.manifest_value() == config.manifest_value()


def test_audit_all_loads_model_once_and_evaluates_each_finished_split(
    tmp_path,
    monkeypatch,
) -> None:
    from experiments.reanchor_flow import run

    args = run.parser().parse_args(["audit-all", "--output", str(tmp_path)])
    model, tokenizer = object(), SimpleNamespace(name_or_path="model")
    calls = []

    def load(*_args):
        calls.append("model")
        return model, tokenizer

    def capture(received_model, received_tokenizer, output, config):
        assert received_model is model and received_tokenizer is tokenizer
        assert config.samples_per_task == 0 and config.max_response_tokens is None
        calls.append(f"capture {config.split}")
        return {"samples": 2, "targets": 6, "resumed": 0, "confirmed": 0}

    def evaluate(dataset, output, *, plot):
        assert plot
        calls.append(f"evaluate {output.name}")
        return {"groups": {}}

    monkeypatch.setattr(run, "load_model", load)
    monkeypatch.setattr(run, "run_subset_split", capture)
    monkeypatch.setattr(run, "evaluate_subset_split", evaluate)
    monkeypatch.setattr(run, "_render_scans", lambda *_args: 2)
    monkeypatch.setattr(run, "clear_memory", lambda: None)

    reports = run.audit_subset(args)

    assert calls == [
        "model",
        "capture train",
        "evaluate train",
        "capture test",
        "evaluate test",
    ]
    assert set(reports) == {"train", "test"}


def test_cli_prints_not_run_for_missing_confirmation_rate() -> None:
    from experiments.reanchor_flow.run import confirmation_rate

    assert confirmation_rate(None) == "not-run"
    assert confirmation_rate(0.5) == "0.5000"


def test_resumed_target_advances_target_progress(tmp_path, monkeypatch) -> None:
    from experiments.reanchor_flow import subset

    RecordingTqdm.instances.clear()
    monkeypatch.setattr(subset, "tqdm", RecordingTqdm)
    monkeypatch.setattr(
        subset,
        "NativeAuditMetadata",
        lambda **values: SimpleNamespace(**values),
    )
    monkeypatch.setattr(subset, "validate_native_audit", lambda *_args: None)

    target = TargetContrast(4, 9, 8, "observed-vs-runner")
    world = SimpleNamespace(sample_id="q1", targets=(target,), target_selection=())
    record = SampleRecord("q1", "source-a", "QA", "generator")
    output = tmp_path / "output"
    destination = output / "audits" / "QA" / "q1" / "q4_a9_b8_message.npz"
    destination.parent.mkdir(parents=True)
    np.savez_compressed(destination, corridor_confirmed=False)
    config = _config(tmp_path)
    identity = subset._audit_identity(
        world,
        record,
        target,
        0,
        destination,
        output,
        config,
    )
    manifest = {"audits": {"q1:q4_a9_b8_message": identity}}
    counts = {"samples": 0, "targets": 0, "resumed": 0, "confirmed": 0}

    subset._run_world_targets(
        object(),
        world,
        record,
        output,
        manifest,
        output / "run_manifest.json",
        counts,
        config,
    )

    progress = RecordingTqdm.instances[-1]
    assert progress.options["unit"] == "target"
    assert progress.completed == 1
    assert progress.postfix == ["resumed"]
    assert counts["resumed"] == 1


def test_subset_split_advances_sample_progress(tmp_path, monkeypatch) -> None:
    from experiments.reanchor_flow import subset

    RecordingTqdm.instances.clear()
    record = SampleRecord("q1", "source-a", "QA", "generator")
    dataset = SimpleNamespace(manifest={"split": "test"}, spec={})
    world = SimpleNamespace(
        sample_id="q1",
        tokenizer_id="tokenizer",
        targets=(),
        target_selection=(),
    )
    monkeypatch.setattr(subset, "tqdm", RecordingTqdm)
    monkeypatch.setattr(subset, "open_research_dataset", lambda *_args, **_kw: dataset)
    monkeypatch.setattr(
        subset,
        "load_source_info",
        lambda _path: {"source-a": {"task_type": "QA"}},
    )
    monkeypatch.setattr(subset, "inspect_records", lambda *_args, **_kw: (record,))
    monkeypatch.setattr(subset, "select_records", lambda *_args, **_kw: (record,))
    monkeypatch.setattr(subset, "load_world_from_dataset", lambda *_args, **_kw: world)
    monkeypatch.setattr(subset, "save_native_world", lambda *_args: None)
    monkeypatch.setattr(subset, "_run_world_targets", lambda *_args: None)

    counts = subset.run_subset_split(
        object(),
        SimpleNamespace(name_or_path="tokenizer"),
        tmp_path / "output",
        _config(tmp_path),
    )

    progress = RecordingTqdm.instances[-1]
    assert progress.options["unit"] == "sample"
    assert progress.completed == 1
    assert progress.postfix == ["QA/q1"]
    assert counts["samples"] == 1
    manifest = json.loads(
        (tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["subset_manifest_schema"] == 3


def test_computed_target_reports_internal_phases(tmp_path, monkeypatch) -> None:
    from experiments.reanchor_flow import subset

    RecordingTqdm.instances.clear()
    RecordingTqdm.writes.clear()
    monkeypatch.setattr(subset, "tqdm", RecordingTqdm)
    monkeypatch.setattr(
        subset,
        "NativeAuditMetadata",
        lambda **values: SimpleNamespace(**values),
    )
    monkeypatch.setattr(subset, "save_native_audit", lambda *_args: None)
    monkeypatch.setattr(subset, "validate_native_audit", lambda *_args: None)

    def audit(*_args, on_phase, **_kwargs):
        on_phase("capture")
        on_phase("plan")
        return SimpleNamespace(
            corridor_confirmed=False,
            selected_root_unit_id=2,
            selected_root_confirmed=False,
            corridor=SimpleNamespace(count=7),
            plan=SimpleNamespace(hubs=(object(), object())),
            effect=SimpleNamespace(restoration_error=float("nan")),
        )

    monkeypatch.setattr(subset, "audit_native_target", audit)
    target = TargetContrast(4, 9, 8, "observed-vs-runner")
    world = SimpleNamespace(sample_id="q1", targets=(target,), target_selection=())
    record = SampleRecord("q1", "source-a", "QA", "generator")
    output = tmp_path / "output"
    counts = {"samples": 0, "targets": 0, "resumed": 0, "confirmed": 0}

    subset._run_world_targets(
        object(),
        world,
        record,
        output,
        {"audits": {}},
        output / "run_manifest.json",
        counts,
        _config(tmp_path),
    )

    progress = RecordingTqdm.instances[-1]
    assert progress.completed == 1
    assert progress.postfix == ["capture", "plan", "computed"]
    assert counts["targets"] == 1
    assert "corridor_edges=7 hubs=2 exact=not-run" in RecordingTqdm.writes[-1]


def test_subset_plot_decodes_artifact_tokens_for_source_labels(
    tmp_path, monkeypatch
) -> None:
    from experiments.reanchor_flow import run

    output = tmp_path / "output"
    artifact = output / "audits" / "QA" / "sample" / "target.npz"
    artifact.parent.mkdir(parents=True)
    np.savez_compressed(artifact, token_ids=np.asarray([11, 12, 13]))
    calls = []
    monkeypatch.setattr(run, "tqdm", lambda iterable, **_options: iterable)
    monkeypatch.setattr(
        run,
        "render_artifact",
        lambda path, output_path=None, token_labels=None: calls.append(
            (path, output_path, token_labels)
        ),
    )
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda values: [f"tok-{value}" for value in values]
    )

    count = run._render_subset(output, tokenizer)

    assert count == 1
    assert calls == [(artifact, None, ["tok-11", "tok-12", "tok-13"])]
