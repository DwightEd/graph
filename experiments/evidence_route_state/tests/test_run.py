from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.evidence_route_state import run
from experiments.evidence_route_state.capture import PromptRouteControl
from experiments.evidence_route_state.data import TASK_TYPES, PromptUnits, RouteSample
from experiments.evidence_route_state.graph import GraphSequence
from experiments.evidence_route_state.messages import PromptCarriers


def graph(tokens: int = 4) -> GraphSequence:
    return GraphSequence(
        query_position=torch.arange(tokens) + 9,
        prediction_position=torch.arange(tokens) + 10,
        node_embedding=torch.arange(tokens * 4 * 3).reshape(tokens, 4, 3).float(),
        residual_gram=torch.ones(tokens, 3, 4, 4),
        head_write_gram=torch.ones(tokens, 2, 2, 4, 4),
        route_topology=torch.ones(tokens, 2, 2, 4, 7),
        mlp_relation=torch.ones(tokens, 2, 5),
        margin_contribution=torch.ones(tokens, 4),
        valid=torch.tensor([False, False, True, True]),
    )


def route_sample(split: str, task: str, number: int) -> RouteSample:
    return RouteSample(
        sample_id=f"{split}-{task}-{number}",
        source_id=f"source-{split}-{task}-{number}",
        split=split,
        task_type=task,
        data_source="fixture",
        generator_model="generator",
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        response_start=2,
        prompt_units=PromptUnits(
            token_unit_id=torch.tensor([0, 1]),
            evidence_name=("evidence",),
            evidence_char_span=torch.tensor([[0, 1]]),
        ),
    )


def fake_trace():
    layers, tokens, heads = 2, 4, 2

    def carriers():
        return PromptCarriers(
            effective_sources=torch.ones(layers, tokens),
            effective_rank=torch.ones(layers, tokens),
            anchor_source=torch.zeros(layers, tokens, heads, dtype=torch.int32),
        )

    return SimpleNamespace(
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        response_start=2,
        graph=graph(),
        target_logprob=torch.zeros(tokens),
        target_confidence=torch.ones(tokens),
        target_margin=torch.zeros(tokens),
        prompt_route=PromptRouteControl(carriers(), carriers()),
        attention_write_error=torch.zeros(layers, tokens),
        register_closure_error=torch.zeros(layers + 1, tokens),
    )


def test_saved_capture_round_trips_the_complete_graph_sequence(tmp_path):
    path = tmp_path / "capture.npz"
    sample = route_sample("train", "QA", 0)
    run.save_capture(path, sample, fake_trace())

    restored = run.load_sequence(path)

    for name in run.GRAPH_FLOAT_FIELDS:
        np.testing.assert_allclose(
            run.array(getattr(restored, name)),
            run.array(getattr(fake_trace().graph, name)),
            rtol=1e-3,
            atol=1e-3,
        )
    np.testing.assert_array_equal(restored.query_position, [9, 10, 11, 12])
    np.testing.assert_array_equal(restored.prediction_position, [10, 11, 12, 13])


def test_saved_gram_matrices_do_not_overflow_float16(tmp_path):
    path = tmp_path / "capture.npz"
    sample = route_sample("train", "QA", 0)
    trace = fake_trace()
    trace.graph.residual_gram.fill_(100_000)
    trace.graph.head_write_gram.fill_(-100_000)

    run.save_capture(path, sample, trace)
    restored = run.load_sequence(path)

    assert restored.residual_gram.dtype == np.float32
    assert restored.head_write_gram.dtype == np.float32
    assert np.isfinite(restored.residual_gram).all()
    assert np.isfinite(restored.head_write_gram).all()


def test_capture_limit_is_per_task_and_per_physical_split(tmp_path, monkeypatch):
    def samples(split_root, _source_info, _tokenizer):
        split = Path(split_root).name
        for number in range(2):
            for task in TASK_TYPES:
                yield route_sample(split, task, number)

    class Replay:
        def __init__(self):
            self.seen = []

        def capture(self, token_ids, response_start, evidence_mask, **_kwargs):
            self.seen.append(
                (token_ids.tolist(), response_start, evidence_mask.tolist())
            )
            return fake_trace()

    monkeypatch.setattr(run, "iter_route_samples", samples)
    args = SimpleNamespace(
        cache=tmp_path / "cache",
        source_info=tmp_path / "source.jsonl",
        output=tmp_path / "output",
        limit=1,
        predictor_chunk=2,
    )
    replay = Replay()

    records = run.capture_all(args, replay, object())

    assert len(replay.seen) == 2 * len(TASK_TYPES)
    assert all(
        len(records[task][split]) == 1
        for task in TASK_TYPES
        for split in ("train", "test")
    )
    assert all(
        record["path"].find("graph_sequences") >= 0
        for task in TASK_TYPES
        for split in ("train", "test")
        for record in records[task][split]
    )


def test_reference_and_calibration_sources_are_deterministic_and_disjoint():
    records = [
        {"sample_id": f"sample-{index}", "source_id": f"source-{index // 2}"}
        for index in range(32)
    ]

    reference, calibration = run.split_reference(records)
    repeated = run.split_reference(list(reversed(records)))
    reference_sources = {record["source_id"] for record in reference}
    calibration_sources = {record["source_id"] for record in calibration}

    assert reference_sources.isdisjoint(calibration_sources)
    assert len(reference_sources) == 12
    assert len(calibration_sources) == 4
    assert {record["sample_id"] for record in reference} == {
        record["sample_id"] for record in repeated[0]
    }
    assert {record["sample_id"] for record in calibration} == {
        record["sample_id"] for record in repeated[1]
    }


def test_reference_split_rejects_a_one_source_smoke_run():
    with np.testing.assert_raises_regex(ValueError, "--limit 2 or greater"):
        run.split_reference([{"sample_id": "sample", "source_id": "source"}])
