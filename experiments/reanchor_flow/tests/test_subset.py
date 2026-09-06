from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.reanchor_timeline import StructuralReanchorEvent
from experiments.reanchor_flow.route_plan import RouteBudget
from experiments.reanchor_flow.subset_data import (
    SampleRecord,
    inspect_records,
    reanchor_target_positions,
    select_records,
)
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model


class FakeSample:
    def __init__(self, record: SampleRecord) -> None:
        self.record = record
        self.released = 0

    @property
    def source_id(self):
        return self.record.source_id

    @property
    def task_type(self):
        return self.record.task_type

    @property
    def generator_model(self):
        return self.record.generator_model

    def release_attention(self):
        self.released += 1


class FakeDataset:
    def __init__(self, records) -> None:
        self.sample_ids = [record.sample_id for record in records]
        self.samples = {record.sample_id: FakeSample(record) for record in records}

    def __getitem__(self, sample_id):
        return self.samples[sample_id]

    def labels(self):
        raise AssertionError("capture opened labels")

    def prepare_evaluation_labels(self, *_):
        raise AssertionError("capture opened labels")


def records():
    return (
        SampleRecord("q1", "source-a", "QA", "g"),
        SampleRecord("q2", "source-a", "QA", "g"),
        SampleRecord("q3", "source-b", "QA", "g"),
        SampleRecord("s1", "source-c", "Summary", "g"),
        SampleRecord("d1", "source-d", "Data2txt", "g"),
    )


def test_subset_selection_is_source_diverse_deterministic_and_label_free() -> None:
    dataset = FakeDataset(records())
    inspected = inspect_records(dataset)
    first = select_records(
        inspected,
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    second = select_records(
        inspected,
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    assert first == second
    assert len({record.source_id for record in first}) == 2
    assert all(sample.released == 1 for sample in dataset.samples.values())


def test_explicit_subset_preserves_requested_order() -> None:
    selected = select_records(
        records(),
        tasks=("QA", "Summary"),
        samples_per_task=1,
        seed=0,
        sample_ids=("s1", "q2"),
    )
    assert [record.sample_id for record in selected] == ["s1", "q2"]


def test_absolute_cache_model_identity_does_not_fall_back_to_basename(
    tmp_path,
) -> None:
    from experiments.reanchor_flow.subset import _model_matches

    requested = tmp_path / "requested" / "same-name"
    exact = SimpleNamespace(spec={"model_path": str(requested)})
    another = SimpleNamespace(
        spec={"model_path": str(tmp_path / "another" / "same-name")}
    )
    legacy = SimpleNamespace(spec={"model_path": "same-name"})
    assert _model_matches(exact, requested)
    assert not _model_matches(another, requested)
    assert _model_matches(legacy, requested)


def test_manifest_uses_schema_and_plain_identity_for_resume(tmp_path) -> None:
    from experiments.reanchor_flow.artifacts import save_json
    from experiments.reanchor_flow.subset import open_manifest

    path = tmp_path / "output" / "run_manifest.json"
    config = {"method": "budgeted_head_resolved_route_audit_v3", "local_window": 7}
    manifest = open_manifest(path, config, (records()[0],))
    assert manifest["subset_manifest_schema"] == 3
    assert manifest["config"] == config
    assert not any("sha256" in key for key in manifest)
    save_json(path, manifest)
    assert open_manifest(path, config, (records()[0],))["samples"] == {}
    with pytest.raises(ValueError, match="another subset configuration"):
        open_manifest(path, {**config, "local_window": 8}, (records()[0],))


def test_subset_cli_needs_no_pair_and_keeps_corridor_contract() -> None:
    from experiments.reanchor_flow.run import parser, subset_config_from_args

    command_parser = parser()
    subset = command_parser.parse_args(["subset"])
    assert subset.flow_signal == "message"
    assert subset.samples_per_task == 1
    assert subset.targets_per_sample == 1
    assert subset.target_policy == "reanchor"
    assert subset.carrier_scope == "response"
    assert subset.local_window == 10
    assert not hasattr(subset, "pair")
    config = subset_config_from_args(
        subset,
        SimpleNamespace(name_or_path="tiny-llama"),
        "test",
    )
    assert config.tokenizer_id == "tiny-llama"
    assert config.signal.value == "message"
    assert config.carrier_scope == "response"
    assert config.tasks == ("QA", "Summary", "Data2txt")
    with pytest.raises(SystemExit):
        command_parser.parse_args(["corridor"])


def test_subset_argument_validation() -> None:
    from experiments.reanchor_flow.run import parser, validate_args

    command_parser = parser()
    args = command_parser.parse_args(["subset", "--samples-per-task", "0"])
    with pytest.raises(ValueError, match="samples-per-task"):
        validate_args(args)
    args = command_parser.parse_args(["subset", "--split", "all", "--sample-id", "one"])
    with pytest.raises(ValueError, match="concrete"):
        validate_args(args)


def test_reanchor_window_count_is_a_hard_target_row_budget() -> None:
    events = (
        StructuralReanchorEvent(8, 4, 0, 1, 0, 0.9, 2),
        StructuralReanchorEvent(13, 9, 1, 0, 2, 0.7, 1),
    )
    available = torch.arange(4, 17)

    centers = reanchor_target_positions(
        events,
        available,
        count=2,
        policy="reanchor",
    )
    window = reanchor_target_positions(
        events,
        available,
        count=3,
        policy="reanchor-window",
    )

    assert [(position, offset) for position, _, offset in centers] == [(8, 0), (13, 0)]
    assert [(position, offset) for position, _, offset in window] == [
        (8, 0),
        (7, -1),
        (9, 1),
    ]


def test_reanchor_no_event_falls_back_and_records_origin(monkeypatch) -> None:
    from experiments.reanchor_flow import subset_data

    cache = SimpleNamespace(
        query=torch.tensor([4, 5, 6]),
        target=torch.tensor([11, 12, 13]),
        runner=torch.tensor([21, 22, 23]),
    )
    observed_checkpoint_layers = []

    def fake_baseline(*_args, **kwargs):
        observed_checkpoint_layers.extend(kwargs["checkpoint_layers"])
        return cache

    monkeypatch.setattr(subset_data, "baseline_forward", fake_baseline)
    monkeypatch.setattr(
        subset_data,
        "capture_source_location_buckets",
        lambda *_args, **_kwargs: SimpleNamespace(transport=torch.ones(2, 2, 2, 4)),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[object(), object()]))

    targets, selections = subset_data.freeze_target_plan(
        model,
        torch.arange(8),
        5,
        count=1,
        policy="reanchor",
        query_chunk=2,
        units=object(),
        evidence_unit_id=(0,),
        local_window=3,
    )

    assert observed_checkpoint_layers == [0, 1]
    assert targets[0].query_position == 5
    assert "reanchor_fallback" in targets[0].origin
    assert "evenly_spaced" in targets[0].origin
    assert selections[0].fallback
    assert not selections[0].has_event
    assert not selections[0].is_center


def test_reanchor_target_plan_persists_the_shared_structural_event(monkeypatch) -> None:
    from experiments.reanchor_flow import subset_data

    cache = SimpleNamespace(
        query=torch.tensor([4, 5, 6, 7]),
        target=torch.tensor([11, 12, 13, 14]),
        runner=torch.tensor([21, 22, 23, 24]),
    )
    transport = torch.zeros(2, 2, 3, 4)
    transport[..., :3] = torch.tensor([0.05, 0.05, 0.10])
    transport[..., 3] = 0.80
    transport[0, 1, 1] = torch.tensor([0.80, 0.05, 0.05, 0.10])
    source_position = torch.zeros_like(transport, dtype=torch.int32)
    source_unit = torch.zeros_like(transport, dtype=torch.int32)
    source_position[0, 1, 1, 0] = 2
    source_unit[0, 1, 1, 0] = 1
    source_location = SimpleNamespace(
        transport=transport,
        source_position=source_position,
        source_unit_id=source_unit,
    )
    monkeypatch.setattr(subset_data, "baseline_forward", lambda *_a, **_k: cache)
    monkeypatch.setattr(
        subset_data,
        "capture_source_location_buckets",
        lambda *_args, **_kwargs: source_location,
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[object(), object()]))

    targets, selections = subset_data.freeze_target_plan(
        model,
        torch.arange(9),
        5,
        count=1,
        policy="reanchor",
        query_chunk=2,
        units=object(),
        evidence_unit_id=(1,),
        local_window=3,
    )

    assert targets[0].query_position == 6
    assert selections[0].is_center
    assert selections[0].center_position == 6
    assert selections[0].layer == 0
    assert selections[0].head == 1
    assert selections[0].source_kind == "prompt_evidence"
    assert selections[0].source_position == 2
    assert selections[0].source_unit_id == 1
    assert selections[0].score > 0


def test_reanchor_budget_covering_short_response_still_marks_event_center(
    monkeypatch,
) -> None:
    from experiments.reanchor_flow import subset_data

    cache = SimpleNamespace(
        query=torch.tensor([4, 5, 6]),
        target=torch.tensor([11, 12, 13]),
        runner=torch.tensor([21, 22, 23]),
    )
    transport = torch.zeros(1, 2, 2, 4)
    transport[..., :3] = torch.tensor([0.05, 0.05, 0.10])
    transport[..., 3] = 0.80
    transport[0, 1, 1] = torch.tensor([0.85, 0.05, 0.05, 0.05])
    source_position = torch.zeros_like(transport, dtype=torch.int32)
    source_unit = torch.zeros_like(transport, dtype=torch.int32)
    source_position[0, 1, 1, 0] = 2
    source_unit[0, 1, 1, 0] = 1
    source_location = SimpleNamespace(
        transport=transport,
        source_position=source_position,
        source_unit_id=source_unit,
    )
    checkpoint_layers = []

    def fake_baseline(*_args, **kwargs):
        checkpoint_layers.extend(kwargs["checkpoint_layers"])
        return cache

    monkeypatch.setattr(subset_data, "baseline_forward", fake_baseline)
    monkeypatch.setattr(
        subset_data,
        "capture_source_location_buckets",
        lambda *_args, **_kwargs: source_location,
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[object()]))

    targets, selections = subset_data.freeze_target_plan(
        model,
        torch.arange(8),
        5,
        count=3,
        policy="reanchor-window",
        query_chunk=2,
        units=object(),
        evidence_unit_id=(1,),
        local_window=3,
    )

    assert checkpoint_layers == [0]
    assert [target.query_position for target in targets] == [4, 5, 6]
    assert [selection.is_center for selection in selections] == [False, False, True]
    assert selections[2].center_position == 6
    assert selections[2].source_kind == "prompt_evidence"
    assert all(not selection.fallback for selection in selections)
    assert "center_q6" in targets[2].origin


def test_subset_pipeline_resumes_valid_native_audit(tmp_path, monkeypatch) -> None:
    from experiments.reanchor_flow import subset, subset_data

    pair = paired_world()

    class AuditSample(FakeSample):
        def attention(self):
            return SimpleNamespace(
                token_ids=pair.clean_token_ids,
                response_idx=pair.response_start,
            )

    class AuditDataset(FakeDataset):
        def __init__(self):
            record = SampleRecord("q1", "source-a", "QA", "generator")
            self.spec = {}
            self.manifest = {"split": "test"}
            self.sample_ids = [record.sample_id]
            self.samples = {record.sample_id: AuditSample(record)}

    dataset = AuditDataset()

    def open_dataset(*_args, **kwargs):
        assert kwargs["retain_embedded_labels"] is False
        return dataset

    monkeypatch.setattr(subset, "open_research_dataset", open_dataset)
    monkeypatch.setattr(
        subset,
        "load_source_info",
        lambda _path: {
            "source-a": {
                "task_type": "QA",
                "prompt": "unused",
                "source_info": {"passages": "unused"},
            }
        },
    )
    monkeypatch.setattr(
        subset_data,
        "build_source_units",
        lambda *_args, **_kwargs: pair.units,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    model = tiny_model()
    tokenizer = SimpleNamespace(name_or_path="tiny-llama")
    config = subset.SubsetRunConfig(
        model_id=str((tmp_path / "tiny-llama").resolve()),
        model_dtype="float32",
        tokenizer_id="tiny-llama",
        dataset_root=str(cache.resolve()),
        source_info=str(source.resolve()),
        split="test",
        tasks=("QA",),
        samples_per_task=1,
        explicit_sample_ids=(),
        selection_seed=2026,
        targets_per_sample=1,
        target_policy="reanchor",
        max_response_tokens=None,
        signal=subset.FlowSignal.MESSAGE,
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        route_budget=RouteBudget(
            edges_per_head=2,
            max_rows=32,
            root_candidates=2,
            hub_candidates=1,
            corridor_edges=8,
            confirm=False,
        ),
        local_window=3,
        saved_edges=4,
    )
    real_audit = subset.audit_native_target
    seen_audit_options = {}

    def record_audit_options(*args, **kwargs):
        seen_audit_options.update(kwargs)
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(subset, "audit_native_target", record_audit_options)
    first = subset.run_subset_split(model, tokenizer, output, config)
    assert dataset.verify_hashes is True
    assert seen_audit_options["local_window"] == 3
    assert first == {
        "samples": 1,
        "targets": 1,
        "resumed": 0,
        "confirmed": first["confirmed"],
    }
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["subset_manifest_schema"] == 3
    assert manifest["analysis_complete"]
    assert manifest["labels_used_for_capture"] is False
    assert manifest["config"]["local_window"] == 3
    assert "config_sha256" not in manifest
    audit_entry = next(iter(manifest["audits"].values()))
    assert "sha256" not in audit_entry
    result_path = output / audit_entry["result"]
    with np.load(result_path, allow_pickle=False) as stored:
        assert int(stored["subset_audit_schema"]) == 3
        assert int(stored["local_window"]) == 3
        assert int(stored["edge_saved_count"]) == len(stored["edge_layer"])
        assert int(stored["corridor_edge_count"]) == len(
            stored["frozen_corridor_layer"]
        )
        assert int(stored["edge_in_frozen_corridor"].sum()) == int(
            stored["corridor_edge_count"]
        )
        assert bool(stored["target_reanchor_fallback"])

    monkeypatch.setattr(
        subset,
        "audit_native_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume recomputed a completed audit")
        ),
    )
    second = subset.run_subset_split(model, tokenizer, output, config)
    assert second["targets"] == 1
    assert second["resumed"] == 1
    with pytest.raises(ValueError, match="another subset configuration"):
        subset.run_subset_split(
            model,
            tokenizer,
            output,
            replace(config, local_window=4),
        )
