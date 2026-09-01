import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import experiments.attention_mechanism_audit.evaluate as evaluate


def _artifact(tokens: int = 2) -> dict:
    layers, heads, roles = 3, 2, 4
    attention = torch.ones(layers, tokens, heads, roles)
    edge = attention.clone()
    write = attention.clone()
    entropy = torch.full((layers, tokens, heads), 0.25)
    coherence = torch.full((layers, tokens, roles), 0.5)
    token_ids = torch.arange(tokens + 3)
    full = torch.arange(tokens, dtype=torch.float32) - 1
    return {
        "token_ids": token_ids,
        "response_start": 3,
        "trace": {
            "role_attention_mass": attention,
            "edge_role_energy": edge,
            "head_role_write_norm": write,
            "head_source_entropy": entropy,
            "role_head_coherence": coherence,
            "top_source_index": torch.zeros(layers, tokens, 1, dtype=torch.int32),
            "top_source_magnitude": torch.ones(layers, tokens, 1),
        },
        "score_inputs": {
            "full_logprob": full,
            "full_margin": torch.ones(tokens),
            "no_evidence_logprob": full - 1,
            "no_evidence_margin": torch.zeros(tokens),
            "no_history_logprob": full - 2,
            "no_history_margin": torch.zeros(tokens),
            "no_evidence_history_logprob": full - 4,
            "no_evidence_history_margin": torch.full((tokens,), 0.75),
        },
    }


def test_rich_audit_preserves_self_head_structure_and_factorial_equations():
    artifact = _artifact(3)
    artifact["trace"]["edge_role_energy"][:, :, 0] = torch.tensor(
        [4.0, 2.0, 1.0, 1.0]
    )
    artifact["trace"]["edge_role_energy"][:, :, 1] = torch.tensor(
        [4.0, 2.0, 1.0, 1.0]
    )
    layers = evaluate.layer_audit_metrics(artifact)
    audit = evaluate.token_audit_metrics(artifact)

    np.testing.assert_allclose(layers["edge_evidence_share"], 0.5)
    np.testing.assert_allclose(layers["edge_predictor_self_share"], 0.125)
    np.testing.assert_allclose(layers["edge_route_balance"], -3 / 7)
    np.testing.assert_allclose(layers["source_coherence_evidence"], 0.25)
    np.testing.assert_allclose(layers["edge_attention_gain_evidence"], 0.25)
    np.testing.assert_allclose(layers["edge_head_role_jsd"], 0.0, atol=1e-12)
    np.testing.assert_allclose(layers["edge_route_velocity"][:, 0], 0.0)
    # full/noE/noR/noER offsets are 0/1/2/4: E=1.5, R=2.5, I=-1.
    np.testing.assert_allclose(audit["causal_evidence_support"], 1.5)
    np.testing.assert_allclose(audit["causal_history_support"], 2.5)
    np.testing.assert_allclose(audit["causal_interaction"], -1.0)
    np.testing.assert_allclose(audit["remaining_context_margin"], 0.75)
    assert "attention_response_history_share_layer_shift" in audit
    assert "write_route_velocity_mean" in audit
    assert "head_coherence_predictor_self_late" in audit


def test_auc_direction_is_never_flipped_after_reading_labels():
    label = np.asarray([0, 1, 0, 1], dtype=bool)
    source = np.asarray(["a", "a", "b", "b"])
    scores = {
        name: np.asarray([0.9, 1.0, 0.0, 0.1])
        for name in evaluate.SCORE_ORDER
    }
    scores["static_state"] *= -1

    result = evaluate.detection_summary(
        label, scores, source, bootstrap=0, seed=1
    )

    assert result["mechanism_innovation"]["auroc"] == 0.75
    np.testing.assert_allclose(result["mechanism_innovation"]["auprc"], 5 / 6)
    assert result["static_state"]["auroc"] == 0.25


def test_group_audit_matches_labels_only_within_position_cells():
    arrays = {
        "label": np.asarray([0, 1, 0, 1], dtype=bool),
        "sample_id": np.asarray(["a", "a", "b", "b"]),
        "source_id": np.asarray(["s1", "s1", "s2", "s2"]),
        "token_index": np.asarray([0, 1, 0, 1]),
        "response_length": np.asarray([20, 20, 20, 20]),
        "edge_route_velocity_mean": np.asarray([0.0, 2.0, 1.0, 5.0]),
    }
    report = evaluate.group_difference_audit(
        arrays,
        ("edge_route_velocity_mean",),
        position_bin=16,
        bootstrap=0,
        seed=1,
    )
    result = report["metrics"]["edge_route_velocity_mean"]
    assert result["hallucinated_minus_correct"] == 3.0
    assert result["sources"] == 2
    assert report["hallucinated_token_coverage"] == 1.0


def test_onset_difference_in_difference_removes_local_background_drift():
    length = 100
    token = np.arange(length)
    label = np.zeros(length, dtype=bool)
    label[55:57] = True
    value = token * 0.1
    value[55:] += 2.0
    result = evaluate._onset_difference_in_difference(
        value,
        label,
        np.repeat("sample", length),
        np.repeat("source", length),
        token,
        np.repeat(length, length),
        position_bin=16,
        window=1,
        bootstrap=0,
        seed=1,
    )
    np.testing.assert_allclose(
        result["onset_change_minus_matched_correct_change"], 2.0
    )
    assert result["onsets"] == 1
    assert result["matched_correct_pivots"] > 0


def _write_shard(root: Path, split: str, sample: str) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": evaluate.SCHEMA,
                "version": evaluate.VERSION,
                "task_types": ["QA", "Summary", "Data2txt"],
                "split": split,
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "index.jsonl").write_text(
        json.dumps(
            {
                "sample_id": sample,
                "source_id": sample,
                "task_type": "Summary",
                "path": f"{sample}.pt",
                "response_tokens": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_pool_score_freeze_then_labels_is_the_only_evaluation_order(
    tmp_path, monkeypatch
):
    train, test = tmp_path / "train", tmp_path / "test"
    _write_shard(train, "train", "train_sample")
    _write_shard(test, "test", "test_sample")
    events = []

    def score_records(records, *, seed):
        assert all("split_root" not in record for record in records)
        events.append(("score", tuple(record["physical_shard"] for record in records)))
        return {
            "train_sample": {
                name: np.asarray([0.9, 1.0], dtype=np.float32)
                for name in evaluate.SCORE_ORDER
            },
            "test_sample": {
                name: np.asarray([0.0, 0.1], dtype=np.float32)
                for name in evaluate.SCORE_ORDER
            },
        }, {"crossfit_complete": True, "seed": seed}

    monkeypatch.setattr(evaluate, "score_records", score_records)
    monkeypatch.setattr(evaluate, "_load_artifact", lambda _path: _artifact())
    original_write = evaluate._write_frozen

    def write_frozen(path, arrays):
        assert "label" not in arrays
        events.append(("freeze", len(arrays["mechanism_innovation"])))
        return original_write(path, arrays)

    monkeypatch.setattr(evaluate, "_write_frozen", write_frozen)

    class Sample:
        def release_attention(self):
            pass

    class Prepared:
        def response_labels(self, _sample):
            return torch.tensor([False, True])

    class Dataset:
        def prepare_evaluation_labels(self, ids):
            return Prepared()

        def __getitem__(self, _sample):
            return Sample()

    def open_dataset(path, **kwargs):
        assert kwargs["retain_embedded_labels"] is True
        events.append(("label", Path(path).name))
        return Dataset()

    monkeypatch.setattr(evaluate, "open_research_dataset", open_dataset)
    monkeypatch.setattr(evaluate, "plot_population", lambda *_args: None)
    output = tmp_path / "result" / "report.json"
    report = evaluate.evaluate_all(
        inputs=[(train, "cache/train"), (test, "cache/test")],
        task_type="Summary",
        output=output,
        bootstrap=0,
        seed=7,
    )

    assert events == [
        ("score", (0, 1)),
        ("freeze", 4),
        ("label", "train"),
        ("label", "test"),
    ]
    assert report["primary_score"] == "mechanism_innovation"
    assert report["control_scores"] == ["static_state", "confidence"]
    assert report["detection_estimand"] == "token_micro"
    assert report["detection_bootstrap_unit"] == "source_id_cluster"
    assert report["physical_cache_shards"] == 2
    assert report["detection"]["mechanism_innovation"]["auroc"] == 0.75
    assert report["labels_used_during"].endswith("after_score_freeze")
    frozen = output.with_name("frozen_scores.npz")
    assert frozen.is_file()
    with np.load(frozen) as arrays:
        assert "label" not in arrays
    assert report["frozen_scores_sha256"] == hashlib.sha256(
        frozen.read_bytes()
    ).hexdigest()
    with np.load(output.with_name("token_scores.npz")) as arrays:
        assert arrays["label"].tolist() == [False, True, False, True]
    json.loads(output.read_text(encoding="utf-8"))


def test_smoke_capture_does_not_report_zero_scores_as_detection():
    arrays = {
        "label": np.asarray([False, True]),
        "sample_id": np.asarray(["s", "s"]),
        "source_id": np.asarray(["source", "source"]),
    }
    scores = {name: np.zeros(2) for name in evaluate.SCORE_ORDER}
    report = evaluate.build_report(
        task_type="QA",
        arrays=arrays,
        scores=scores,
        detector={
            "mechanism_scores_available": False,
            "reason": "three sources required",
        },
        bootstrap=0,
        seed=1,
    )

    assert all(
        result["auroc"] is None for result in report["detection"].values()
    )
    assert all(
        result["unavailable_reason"] == "three sources required"
        for result in report["detection"].values()
    )


def test_label_length_must_match_frozen_response(tmp_path, monkeypatch):
    records = [
        {
            "sample_id": "s",
            "physical_shard": 0,
            "split_root": tmp_path,
        }
    ]
    frozen = {"response_length": np.asarray([2, 2])}

    class Sample:
        def release_attention(self):
            pass

    class Prepared:
        def response_labels(self, _sample):
            return torch.tensor([True])

    class Dataset:
        def prepare_evaluation_labels(self, _ids):
            return Prepared()

        def __getitem__(self, _sample):
            return Sample()

    monkeypatch.setattr(evaluate, "open_research_dataset", lambda *_a, **_k: Dataset())
    with pytest.raises(ValueError, match="frozen-score/label length mismatch"):
        evaluate._load_labels(records, frozen)


def test_sample_plot_consumes_only_the_new_compact_trace(tmp_path, monkeypatch):
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
    monkeypatch.setattr(evaluate, "_load_artifact", lambda *_args: _artifact())
    monkeypatch.setattr(
        evaluate,
        "plot_sample_dashboard",
        lambda record, layers, output: seen.update(
            record=record, layers=layers, output=output
        ),
    )
    result = evaluate.plot_saved_sample(
        inputs=[tmp_path / "state"],
        sample_id="sample",
        model_path=tmp_path / "model",
        output=tmp_path / "sample.png",
    )
    assert result["sample_id"] == "sample"
    assert seen["record"]["token_text"] == ["t3", "t4"]
    assert seen["record"]["source_flow"].shape == (2, 2)
    assert "route_interaction" in seen["record"]
    assert "history_support" in seen["record"]
    assert seen["layers"]["edge_route_balance"].shape == (3, 2)
    assert "edge_predictor_self_share" in seen["layers"]
    assert "label" not in seen["record"]
