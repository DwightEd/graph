from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.evidence_route_state import run
from experiments.evidence_route_state.capture import RouteMessageReplay
from experiments.evidence_route_state.data import TASK_TYPES, PromptUnits, RouteSample


def route_sample(split: str, task: str, number: int) -> RouteSample:
    return RouteSample(
        sample_id=f"{split}-{task}-{number}",
        source_id=f"source-{split}-{task}-{number}",
        split=split,
        task_type=task,
        data_source="fixture",
        generator_model="generator",
        token_ids=torch.tensor([1, 2, 3]),
        response_start=2,
        prompt_units=PromptUnits(
            token_unit_id=torch.tensor([0, 1]),
            evidence_name=("evidence",),
            evidence_char_span=torch.tensor([[0, 1]]),
        ),
    )


def test_capture_limit_is_applied_once_per_task_in_each_physical_split(
    tmp_path, monkeypatch
):
    def samples(split_root, _source_info, _tokenizer):
        split = Path(split_root).name
        for number in range(2):
            for task in TASK_TYPES:
                yield route_sample(split, task, number)

    captured = []

    def capture(_replay, sample, output, **_kwargs):
        captured.append(sample.sample_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, present=True)

    monkeypatch.setattr(run, "iter_route_samples", samples)
    monkeypatch.setattr(run, "capture_sample", capture)
    args = SimpleNamespace(
        cache=tmp_path / "cache",
        source_info=tmp_path / "source.jsonl",
        output=tmp_path / "output",
        limit=1,
        predictor_chunk=2,
        logit_chunk=2,
        route_coverage=0.9,
        graph_edges_per_head=2,
    )

    records = run.capture_all(args, replay=object(), tokenizer=object())

    assert len(captured) == 2 * len(TASK_TYPES)
    for task in TASK_TYPES:
        assert len(records[task]["train"]) == 1
        assert len(records[task]["test"]) == 1
    index = (args.output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == 2 * len(TASK_TYPES)


def test_both_physical_splits_fit_each_other_and_every_sample_is_scored(
    tmp_path, monkeypatch
):
    records = {
        task: {
            "train": [
                {
                    "sample_id": f"{task}-train-a",
                    "source_id": f"{task}-train-source-a",
                    "generator_model": "generator",
                },
                {
                    "sample_id": f"{task}-train-b",
                    "source_id": f"{task}-train-source-b",
                    "generator_model": "generator",
                },
            ],
            "test": [
                {
                    "sample_id": f"{task}-test-a",
                    "source_id": f"{task}-test-source-a",
                    "generator_model": "generator",
                }
            ],
        }
        for task in TASK_TYPES
    }
    folds = []
    frozen_samples = {}
    events = []

    def score_fold(fit, score, model_path):
        folds.append(
            (
                tuple(record["sample_id"] for record in fit),
                tuple(record["sample_id"] for record in score),
                Path(model_path).name,
            )
        )
        return [dict(record) for record in score]

    def freeze(scored, path):
        task = Path(path).parent.name
        frozen_samples[task] = {record["sample_id"] for record in scored}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"frozen before labels")
        events.append((task, "freeze"))

    def evaluate(_scored, frozen_path, _report_path, *, task_type, **_kwargs):
        assert Path(frozen_path).read_bytes() == b"frozen before labels"
        events.append((task_type.casefold(), "evaluate"))
        return {"task_type": task_type}

    monkeypatch.setattr(
        run.RouteMessageReplay, "from_pretrained", lambda *_a, **_k: object()
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(run, "capture_all", lambda *_a, **_k: records)
    monkeypatch.setattr(run, "score_fold", score_fold)
    monkeypatch.setattr(run, "freeze_scores", freeze)
    monkeypatch.setattr(run, "evaluate_scores", evaluate)
    monkeypatch.setattr(run, "print_report", lambda _report: None)

    args = SimpleNamespace(
        model=tmp_path / "model",
        device="cpu",
        dtype="bfloat16",
        output=tmp_path / "output",
        cache=tmp_path / "cache",
        source_info=tmp_path / "source.jsonl",
        predictor_chunk=2,
        graph_edges_per_head=2,
        route_coverage=0.9,
        bootstrap=0,
        seed=7,
    )
    run.run_all(args)

    assert len(folds) == 2 * len(TASK_TYPES)
    for task_index, task in enumerate(TASK_TYPES):
        train = tuple(record["sample_id"] for record in records[task]["train"])
        test = tuple(record["sample_id"] for record in records[task]["test"])
        forward, reverse = folds[2 * task_index : 2 * task_index + 2]
        assert forward[:2] == (train, test)
        assert reverse[:2] == (test, train)
        assert frozen_samples[task.casefold()] == {*train, *test}
        assert events[2 * task_index : 2 * task_index + 2] == [
            (task.casefold(), "freeze"),
            (task.casefold(), "evaluate"),
        ]


def test_control_training_split_is_deterministic_source_disjoint_and_three_to_one():
    records = [
        {"sample_id": f"sample-{index}", "source_id": f"source-{index // 2}"}
        for index in range(32)
    ]

    nuisance, calibration = run.control_training_split(records)
    repeated = run.control_training_split(list(reversed(records)))
    nuisance_sources = {record["source_id"] for record in nuisance}
    calibration_sources = {record["source_id"] for record in calibration}

    assert nuisance_sources.isdisjoint(calibration_sources)
    assert len(nuisance_sources) == 12
    assert len(calibration_sources) == 4
    assert (
        {record["sample_id"] for record in nuisance},
        {record["sample_id"] for record in calibration},
    ) == (
        {record["sample_id"] for record in repeated[0]},
        {record["sample_id"] for record in repeated[1]},
    )


def test_score_fold_builds_hmm_observations_from_calibrated_lineage_volume(
    tmp_path, monkeypatch
):
    tokens, layers, heads = 6, 2, 2

    def record(index: int, shift: float) -> dict:
        position = (np.arange(tokens) + 0.5) / tokens
        base = 1.0 + 0.3 * position + shift
        route_log_volume = np.vstack((base, 0.8 * base))
        path = tmp_path / f"capture-{index}.npz"
        arrays = {
            "response_start": np.asarray(10 + index),
            "route_log_volume": route_log_volume,
            "raw_route_contraction": np.full(tokens, 0.99),
            "takeover": np.linspace(0.0, 1.0, tokens),
            "valid": np.array([False, False, True, True, True, True]),
            "prediction_position": np.arange(tokens) + 11 + index,
            "target_logprob": np.full(tokens, -0.5),
            "functional_prompt_effective_sources": np.exp(route_log_volume),
            "functional_prompt_effective_rank": np.ones((layers, tokens)),
            "functional_prompt_anchor_source": np.zeros(
                (layers, tokens, heads), dtype=np.int32
            ),
            "attention_prompt_effective_sources": np.exp(route_log_volume),
            "attention_prompt_effective_rank": np.ones((layers, tokens)),
            "attention_prompt_anchor_source": np.zeros(
                (layers, tokens, heads), dtype=np.int32
            ),
        }
        for control in run.TOPOLOGY_CONTROLS:
            arrays[f"{control}_route_log_volume"] = route_log_volume
            arrays[f"{control}_takeover"] = arrays["takeover"]
            arrays[f"{control}_valid"] = arrays["valid"]
        np.savez(path, **arrays)
        return {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "path": str(path),
        }

    fit = [record(index, 0.05 * index) for index in range(8)]
    scored = [record(8, -1.0)]
    fitted_observations = []

    class Detector:
        transition_ = np.full((3, 3), 1 / 3)

        def fit(self, observations, _valid):
            fitted_observations.extend(observations)
            return self

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        def score(self, observation, _valid):
            return observation[:, 0]

        def independent_score(self, observation, _valid):
            return observation[:, 1]

        def expected_dwell_time(self):
            return np.full(3, 1.5)

    monkeypatch.setattr(run, "StickyRouteHMM", Detector)
    output = run.score_fold(fit, scored, tmp_path / "model.npz")

    assert len(fitted_observations) == len(fit) * (1 + len(run.TOPOLOGY_CONTROLS))
    assert all(
        np.nanmax(observation[:, 0]) <= 1.0 for observation in fitted_observations
    )
    valid = output[0]["valid"]
    np.testing.assert_array_equal(
        output[0]["captured_posterior"][valid],
        output[0]["route_contraction"][valid],
    )
    np.testing.assert_array_equal(
        output[0]["raw_route_contraction"],
        np.full(tokens, 0.99),
    )
    assert not np.allclose(output[0]["route_contraction"], 0.99)


def test_capture_sample_uses_dense_lineage_and_is_chunk_invariant(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    replay = RouteMessageReplay(model)
    sample = RouteSample(
        sample_id="sample",
        source_id="source",
        split="test",
        task_type="QA",
        data_source="fixture",
        generator_model="generator",
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        response_start=3,
        prompt_units=PromptUnits(
            token_unit_id=torch.tensor([0, 1, 1]),
            evidence_name=("passage",),
            evidence_char_span=torch.tensor([[0, 2]]),
        ),
    )

    paths = []
    for chunk in (1, 5):
        path = tmp_path / f"capture-{chunk}.npz"
        run.capture_sample(
            replay,
            sample,
            path,
            predictor_chunk=chunk,
            logit_chunk=2,
            route_coverage=0.9,
            graph_edges_per_head=2,
        )
        paths.append(path)

    with np.load(paths[0]) as left, np.load(paths[1]) as right:
        for name in (
            "route_log_volume",
            "takeover",
            "endpoint_rewire_route_log_volume",
            "endpoint_rewire_takeover",
            "weight_shuffle_route_log_volume",
            "weight_shuffle_takeover",
        ):
            np.testing.assert_allclose(left[name], right[name], rtol=2e-4, atol=2e-5)
        assert left["graph_edge_head"].dtype == np.int16
        assert left["graph_edge_source"].dtype == np.int32
