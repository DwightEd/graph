import json

import numpy as np
import torch

import experiments.attention_mechanism_audit.evaluate as evaluate
from experiments.attention_mechanism_audit.capture import HISTORY, ROLE_NAMES
from experiments.attention_mechanism_audit.data import EVIDENCE


def _artifact() -> dict:
    edge = torch.zeros(2, 2, 1, len(ROLE_NAMES))
    edge[:, 0, 0, EVIDENCE] = 1
    edge[0, 1, 0, EVIDENCE] = 3
    edge[0, 1, 0, HISTORY] = 1
    edge[1, 1, 0, EVIDENCE] = 1
    edge[1, 1, 0, HISTORY] = 3
    return {
        "trace": {
            "role_edge_magnitude": edge,
            "source_message_entropy": torch.tensor(
                [[0.0, np.log(2)], [0.0, np.log(4)]]
            ),
            "source_role": torch.tensor(
                [[EVIDENCE, -1, -1, -1], [EVIDENCE, HISTORY, 2, 3]]
            ),
        },
        "mechanism": {
            "evidence_message_effect": torch.tensor([0.5, -1.0]),
            "response_message_effect": torch.tensor([0.2, 2.0]),
            "evidence_response_removed_margin": torch.tensor([0.2, 0.9]),
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
    scores = {
        name: np.asarray([0.9, 1.0, 0.0, 0.1])
        for name in evaluate.SCORE_ORDER
    }
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
    label = np.asarray([0, 1], dtype=bool)

    def shard(name: str, score: list[float]) -> dict[str, np.ndarray]:
        return {
            "label": label,
            "sample_id": np.asarray([name, name]),
            "source_id": np.asarray([name, name]),
            "split": np.asarray([name, name]),
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
    monkeypatch.setattr(
        evaluate,
        "_load_input",
        lambda traces, _cache: pieces[traces.name],
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
            json.dumps({"complete": True}),
            encoding="utf-8",
        )

    output = tmp_path / "report.json"
    report = evaluate.evaluate_all(
        inputs=[(tmp_path / "train", "cache/train"), (tmp_path / "test", "cache/test")],
        output=output,
        bootstrap=0,
    )

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
    assert output.is_file()
    assert (tmp_path / "token_scores.npz").is_file()
    json.loads(output.read_text(encoding="utf-8"))
