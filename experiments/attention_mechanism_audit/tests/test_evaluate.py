import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import experiments.attention_mechanism_audit.evaluate as evaluate


def _artifact() -> dict:
    return {
        "token_ids": torch.tensor([10, 11, 12, 13, 14]),
        "response_start": 3,
        "trace": {
            "total_message_magnitude": torch.tensor([[1.0, 4.0], [1.0, 4.0]]),
            "evidence_message_magnitude": torch.tensor([[1.0, 3.0], [1.0, 1.0]]),
            "response_message_magnitude": torch.tensor([[0.0, 1.0], [0.0, 3.0]]),
            "source_message_entropy": torch.tensor(
                [[0.0, np.log(2)], [0.0, np.log(4)]]
            ),
            "top_source_index": torch.tensor(
                [[[0, 1], [1, 2]], [[0, 2], [2, 3]]], dtype=torch.int32
            ),
            "top_source_magnitude": torch.ones(2, 2, 2),
        },
        "score_inputs": {
            "full_logprob": torch.tensor([-1.0, -2.0]),
            "no_evidence_logprob": torch.tensor([-1.5, -1.0]),
            "no_response_logprob": torch.tensor([-1.2, -4.0]),
            "no_evidence_response_margin": torch.tensor([0.2, 0.9]),
        },
    }


def test_four_scores_are_the_fixed_mechanism_equations():
    scores = evaluate.token_scores(_artifact())

    np.testing.assert_allclose(scores["causal_route_capture"], [-0.3, 3.0])
    np.testing.assert_allclose(scores["routing_imbalance"], [-1.0, 0.0])
    np.testing.assert_allclose(scores["source_dispersion"], [0.0, 0.75])
    np.testing.assert_allclose(
        scores["message_independent_preference"],
        [0.2, 0.9],
    )


def test_auc_direction_is_never_flipped_after_reading_labels():
    label = np.asarray([0, 1, 0, 1], dtype=bool)
    source = np.asarray(["a", "a", "b", "b"])
    scores = {name: np.asarray([0.9, 1.0, 0.0, 0.1]) for name in evaluate.SCORE_ORDER}
    scores["source_dispersion"] *= -1

    result = evaluate.detection_summary(
        label,
        scores,
        source,
        bootstrap=0,
        seed=1,
    )

    assert result["causal_route_capture"]["auroc"] == 0.75
    np.testing.assert_allclose(
        result["causal_route_capture"]["auprc"],
        5 / 6,
    )
    assert result["source_dispersion"]["auroc"] == 0.25


def test_physical_shards_are_pooled_before_one_evaluation(tmp_path, monkeypatch):
    task_type = "Summary"
    label = np.asarray([0, 1], dtype=bool)

    def shard(name: str, score: list[float]) -> dict[str, np.ndarray]:
        return {
            "label": label,
            "sample_id": np.asarray([name, name]),
            "source_id": np.asarray([name, name]),
            "token_index": np.asarray([0, 1], dtype=np.int32),
            "response_length": np.asarray([2, 2], dtype=np.int32),
            **{
                metric: np.asarray(score, dtype=np.float32)
                for metric in evaluate.SCORE_ORDER
            },
        }

    pieces = {
        "train": shard("train", [0.9, 1.0]),
        "test": shard("test", [0.0, 0.1]),
    }
    events = []
    monkeypatch.setattr(
        evaluate,
        "_load_scores",
        lambda traces, task: (
            events.append(f"score:{traces.name}:{task}") or traces.name
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_add_labels",
        lambda name, _cache: events.append(f"label:{name}") or pieces[name],
    )
    plotted = []
    monkeypatch.setattr(
        evaluate,
        "plot_population",
        lambda *args: plotted.append(args),
    )
    for name in pieces:
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": evaluate.SCHEMA,
                    "version": evaluate.VERSION,
                    "split_root": str(Path(f"cache/{name}").resolve()),
                    "task_types": ["QA", "Summary", "Data2txt"],
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "report.json"
    report = evaluate.evaluate_all(
        inputs=[(tmp_path / "train", "cache/train"), (tmp_path / "test", "cache/test")],
        task_type=task_type,
        output=output,
        bootstrap=0,
    )

    assert report["task_type"] == task_type
    assert report["samples"] == 2
    assert report["tokens"] == 4
    assert report["physical_cache_shards"] == 2
    assert report["capture_complete"] is True
    assert report["detection"]["causal_route_capture"]["auroc"] == 0.75
    np.testing.assert_allclose(
        report["detection"]["causal_route_capture"]["auprc"],
        5 / 6,
    )
    assert "by_split" not in report
    assert len(plotted) == 1
    assert events == [
        "score:train:Summary",
        "score:test:Summary",
        "label:train",
        "label:test",
    ]
    assert output.is_file()
    assert (tmp_path / "token_scores.npz").is_file()
    json.loads(output.read_text(encoding="utf-8"))


def test_sample_plot_consumes_only_the_compact_trace(tmp_path, monkeypatch):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def convert_ids_to_tokens(self, values):
            if isinstance(values, list):
                return [f"t{value}" for value in values]
            return f"t{values}"

    seen = {}
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer),
    )
    monkeypatch.setattr(evaluate, "_load_manifest", lambda *_args: {})
    monkeypatch.setattr(
        evaluate,
        "load_index",
        lambda _root: [{"sample_id": "sample", "path": "sample.pt"}],
    )
    monkeypatch.setattr(evaluate.torch, "load", lambda *_args, **_kwargs: _artifact())
    monkeypatch.setattr(
        evaluate,
        "plot_sample_dashboard",
        lambda record, layers, output: seen.update(
            record=record, layers=layers, output=output
        ),
    )

    result = evaluate.plot_saved_sample(
        inputs=[tmp_path / "traces"],
        sample_id="sample",
        model_path=tmp_path / "model",
        output=tmp_path / "sample.png",
    )

    assert result["sample_id"] == "sample"
    assert seen["record"]["token_text"] == ["t13", "t14"]
    assert seen["record"]["source_flow"].shape == (4, 2)
    assert "label" not in seen["record"]
    assert seen["layers"]["routing_imbalance"].shape == (2, 2)
