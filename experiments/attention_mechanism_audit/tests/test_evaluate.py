import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.attention_mechanism_audit import evaluate


def _artifact(tokens: int = 2) -> dict:
    layers, heads, roles = 3, 2, 4
    attention = torch.ones(layers, tokens, heads, roles)
    edge = attention.clone()
    write = attention.clone()
    entropy = torch.full((layers, tokens, heads, roles), 0.25)
    top1 = torch.full((layers, tokens, heads, roles), 0.75)
    coherence = torch.full((layers, tokens, roles), 0.5)
    token_ids = torch.arange(tokens + 3)
    full = torch.arange(tokens, dtype=torch.float32) - 1
    trace = {
        "attention_role_mass": attention,
        "edge_role_mass": edge,
        "head_role_write_norm": write,
        "role_head_coherence": coherence,
        "route_source_index": torch.full(
            (layers, tokens, heads, 2, 1), -1, dtype=torch.int32
        ),
        "route_source_magnitude": torch.zeros(layers, tokens, heads, 2, 1),
        "route_source_remainder": torch.zeros(layers, tokens, heads, 2),
        "route_source_cover_size": torch.zeros(
            layers, tokens, heads, 2, dtype=torch.int16
        ),
        "pathway_effect_norm": torch.ones(layers, tokens, 3, 5),
        "pathway_mlp_projection": torch.tensor([0.1, 0.2, 0.3])
        .expand(layers, tokens, 3)
        .clone(),
        "pathway_pre_output_gain": torch.full((layers, tokens, 3), 1.25),
        "pathway_pre_output_cosine": torch.full((layers, tokens, 3), 0.8),
        "pathway_residual_error": torch.zeros(layers, tokens, 4),
        "pathway_valid": torch.ones(layers, tokens, 3, dtype=torch.bool),
        "pathway_cosine_valid": torch.ones(layers, tokens, 3, dtype=torch.bool),
    }
    trace["route_source_index"][..., 0, 0] = 0
    trace["route_source_magnitude"][..., 0, 0] = 1
    trace["route_source_cover_size"][..., 0] = 1
    for family in ("attention", "edge"):
        trace[f"{family}_role_source_entropy"] = entropy.clone()
        trace[f"{family}_role_top1"] = top1.clone()
        trace[f"{family}_role_anchor_index"] = torch.zeros(
            layers, tokens, heads, roles, dtype=torch.int32
        )
        trace[f"{family}_role_effective_rank"] = torch.full(
            (layers, tokens, roles), 1.5
        )
        trace[f"{family}_role_effective_routes"] = torch.full(
            (layers, tokens, roles), 2.0
        )
    return {
        "token_ids": token_ids,
        "response_start": 3,
        "evidence_mask": torch.tensor([True, False, False]),
        "trace": trace,
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


def test_rich_audit_preserves_head_routes_pathways_and_factorial_equations():
    artifact = _artifact(3)
    artifact["trace"]["edge_role_mass"][:, :, 0] = torch.tensor([4.0, 2.0, 1.0, 1.0])
    artifact["trace"]["edge_role_mass"][:, :, 1] = torch.tensor([4.0, 2.0, 1.0, 1.0])
    layers = evaluate.layer_audit_metrics(artifact)
    audit = evaluate.token_audit_metrics(artifact)

    np.testing.assert_allclose(layers["edge_evidence_mass_share"], 0.5)
    np.testing.assert_allclose(layers["edge_predictor_self_mass_share"], 0.125)
    np.testing.assert_allclose(layers["edge_evidence_effective_routes"], 2.0)
    np.testing.assert_allclose(layers["edge_evidence_effective_rank"], 1.5)
    np.testing.assert_allclose(layers["edge_evidence_head_entropy"], 0.25)
    np.testing.assert_allclose(layers["edge_evidence_head_top1"], 0.75)
    np.testing.assert_allclose(layers["evidence_within_head_cancellation"], 0.75)
    np.testing.assert_allclose(layers["pathway_evidence_mlp_projection"], 0.1)
    # full/noE/noR/noER offsets are 0/1/2/4: E=1.5, R=2.5, I=-1.
    np.testing.assert_allclose(audit["causal_evidence_support"], 1.5)
    np.testing.assert_allclose(audit["causal_history_support"], 2.5)
    np.testing.assert_allclose(audit["causal_interaction"], -1.0)
    np.testing.assert_allclose(audit["direct_evidence_support_with_history"], 1.0)
    np.testing.assert_allclose(audit["history_support_under_direct_evidence_cut"], 3.0)
    np.testing.assert_allclose(audit["unsupported_history_takeover_raw"], 2.0)
    np.testing.assert_allclose(audit["edge_evidence_head_cover_size_mean"], 1)
    np.testing.assert_allclose(audit["edge_evidence_anchor_persistence_mean"][1:], 1)
    assert "attention_response_history_mass_share_layer_shift" in audit
    assert "edge_evidence_route_contraction" in audit
    assert "head_coherence_evidence_layer_shift" in audit
    assert "head_coherence_predictor_self_mean" not in audit
    assert audit["edge_evidence_route_contraction__valid"].dtype == np.bool_
    assert len(audit) < 105
    assert not any(name.endswith(("_early", "_late")) for name in audit)


def test_per_head_route_summaries_are_weighted_by_role_mass():
    artifact = _artifact()
    trace = artifact["trace"]
    trace["edge_role_mass"][:, :, 0, 0] = 9
    trace["edge_role_mass"][:, :, 1, 0] = 1
    trace["edge_role_source_entropy"][:, :, 0, 0] = 0.1
    trace["edge_role_source_entropy"][:, :, 1, 0] = 0.9

    layers = evaluate.layer_audit_metrics(artifact)

    np.testing.assert_allclose(layers["edge_evidence_head_entropy"], 0.18)
    np.testing.assert_allclose(layers["edge_evidence_head_entropy_spread"], 0.24)


def test_pathway_ratios_ignore_layers_without_a_defined_contrast():
    artifact = _artifact()
    artifact["trace"]["pathway_mlp_projection"][:, :, 0] = torch.tensor(
        [10.0, 2.0, 4.0]
    )[:, None]
    artifact["trace"]["pathway_valid"][:, :, 0] = torch.tensor([False, True, True])[
        :, None
    ]

    audit = evaluate.token_audit_metrics(artifact)

    np.testing.assert_allclose(audit["pathway_evidence_mlp_projection_mean"], 3)
    np.testing.assert_allclose(audit["pathway_evidence_mlp_projection_layer_shift"], 0)
    np.testing.assert_allclose(audit["pathway_evidence_valid_mean"], 2 / 3)


def test_auc_direction_is_never_flipped_after_reading_labels():
    label = np.asarray([0, 1, 0, 1], dtype=bool)
    source = np.asarray(["a", "a", "b", "b"])
    scores = {name: np.asarray([0.9, 1.0, 0.0, 0.1]) for name in evaluate.SCORE_ORDER}
    scores[evaluate.SCORE_ORDER[1]] *= -1

    result = evaluate.detection_summary(label, scores, source, bootstrap=0, seed=1)

    assert result[evaluate.PRIMARY_SCORE]["auroc"] == 0.75
    np.testing.assert_allclose(
        result[evaluate.PRIMARY_SCORE]["average_precision"], 5 / 6
    )
    assert result[evaluate.SCORE_ORDER[1]]["auroc"] == 0.25


def test_group_audit_matches_labels_only_within_position_cells():
    arrays = {
        "label": np.asarray([0, 1, 0, 1], dtype=bool),
        "sample_id": np.asarray(["a", "a", "b", "b"]),
        "source_id": np.asarray(["s1", "s1", "s2", "s2"]),
        "token_index": np.asarray([0, 1, 0, 1]),
        "response_length": np.asarray([20, 20, 20, 20]),
        "detection_valid": np.ones(4, dtype=bool),
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


def test_history_audits_exclude_the_pre_history_prefix():
    arrays = {
        "label": np.asarray([False, True, False, True]),
        "sample_id": np.repeat("sample", 4),
        "source_id": np.repeat("source", 4),
        "token_index": np.arange(4),
        "response_length": np.repeat(20, 4),
        "detection_valid": np.asarray([False, False, True, True]),
        "unsupported_history_takeover_raw": np.asarray([0.0, 100.0, 0.0, 2.0]),
    }
    report = evaluate.group_difference_audit(
        arrays,
        ("unsupported_history_takeover_raw",),
        position_bin=16,
        bootstrap=0,
        seed=1,
    )
    result = report["metrics"]["unsupported_history_takeover_raw"]

    assert result["token_scope"] == "strict_history_eligible"
    assert result["hallucinated_mean"] == 2.0
    assert result["hallucinated_minus_correct"] == 2.0
    assert report["strict_history_eligible_tokens"] == 2


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
    np.testing.assert_allclose(result["onset_change_minus_matched_correct_change"], 2.0)
    assert result["onsets"] == 1
    assert result["matched_correct_pivots"] > 0


def _write_shard(root: Path, split: str, sample: str) -> None:
    root.mkdir(parents=True)
    capture_spec = {"route_cover_mass": 0.8}
    artifact_contract = {
        "schema": evaluate.SCHEMA,
        "version": evaluate.VERSION,
        "capture_spec": capture_spec,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": evaluate.SCHEMA,
                "version": evaluate.VERSION,
                "task_types": ["QA", "Summary", "Data2txt"],
                "split": split,
                "split_identity": f"identity-{split}",
                "observer_identity": "observer",
                "model_dtype": "torch.bfloat16",
                "capture_spec": capture_spec,
                "source_identity": "source",
                "labels_used": False,
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
                "generator_model": "generator",
                "path": f"{sample}.pt",
                "response_tokens": 2,
                "prompt_tokens": 3,
                "evidence_tokens": 1,
                "artifact_contract": artifact_contract,
                "token_ids_sha256": evaluate.token_ids_sha256(np.arange(5)),
                "evidence_mask_sha256": "evidence-digest",
                "target_response_sha256": evaluate.target_response_sha256(
                    np.arange(5), 3
                ),
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
                **{
                    name: np.asarray([0.9, 1.0], dtype=np.float32)
                    for name in evaluate.SCORE_ORDER
                },
                "detection_valid": np.ones(2, dtype=bool),
            },
            "test_sample": {
                **{
                    name: np.asarray([0.0, 0.1], dtype=np.float32)
                    for name in evaluate.SCORE_ORDER
                },
                "detection_valid": np.ones(2, dtype=bool),
            },
        }, {"crossfit_complete": True, "seed": seed}

    monkeypatch.setattr(evaluate, "score_records", score_records)
    monkeypatch.setattr(evaluate, "_load_artifact", lambda *_args: _artifact())
    original_write = evaluate._write_frozen

    def write_frozen(path, arrays):
        assert "label" not in arrays
        events.append(("freeze", len(arrays[evaluate.PRIMARY_SCORE])))
        return original_write(path, arrays)

    monkeypatch.setattr(evaluate, "_write_frozen", write_frozen)

    class Sample:
        source_id = ""
        task_type = "Summary"
        generator_model = "generator"

        def __init__(self, sample_id):
            self.source_id = sample_id

        def attention(self):
            return SimpleNamespace(
                token_ids=torch.arange(5),
                response_idx=3,
            )

        def release_attention(self):
            pass

    class Prepared:
        def response_labels(self, _sample):
            return torch.tensor([False, True])

    class Dataset:
        def prepare_evaluation_labels(self, ids):
            return Prepared()

        def __getitem__(self, sample):
            return Sample(sample)

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
    assert report["primary_score"] == evaluate.SCORE_ORDER[0]
    assert report["control_scores"] == list(evaluate.SCORE_ORDER[1:])
    assert report["detection_estimand"] == "token_micro"
    assert "strict_history_eligible" in report["detection_token_scope"]
    assert report["detection_bootstrap_unit"] == "source_id_cluster"
    assert report["evaluated_tokens"] == 4
    assert report["evaluated_positives"] == 2
    assert report["physical_cache_shards"] == 2
    assert report["detection"][evaluate.PRIMARY_SCORE]["auroc"] == 0.75
    assert report["labels_used_during"].endswith("after_score_freeze")
    frozen = output.with_name("frozen_scores.npz")
    assert frozen.is_file()
    with np.load(frozen) as arrays:
        assert "label" not in arrays
        assert arrays["detection_valid"].all()
    assert (
        report["frozen_scores_sha256"]
        == hashlib.sha256(frozen.read_bytes()).hexdigest()
    )
    with np.load(output.with_name("token_scores.npz")) as arrays:
        assert arrays["label"].tolist() == [False, True, False, True]
    json.loads(output.read_text(encoding="utf-8"))


def test_pool_rejects_mixed_observer_identity(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    _write_shard(train, "train", "train_sample")
    _write_shard(test, "test", "test_sample")
    manifest_path = test / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["observer_identity"] = "different-observer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="different scientific identity"):
        evaluate._pool_records(
            [(train, "cache/train"), (test, "cache/test")], "Summary"
        )


def test_smoke_capture_does_not_report_zero_scores_as_detection():
    arrays = {
        "label": np.asarray([False, True]),
        "sample_id": np.asarray(["s", "s"]),
        "source_id": np.asarray(["source", "source"]),
        "detection_valid": np.asarray([True, True]),
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

    for name in evaluate.SCORE_ORDER[:-1]:
        assert report["detection"][name]["auroc"] is None
        assert (
            report["detection"][name]["unavailable_reason"] == "three sources required"
        )
    assert report["detection"]["confidence"]["auroc"] == 0.5


def test_all_detectors_share_the_strict_history_validity_mask():
    arrays = {
        "label": np.asarray([True, True, False, True]),
        "sample_id": np.asarray(["s", "s", "s", "s"]),
        "source_id": np.asarray(["source", "source", "source", "source"]),
        "detection_valid": np.asarray([False, False, True, True]),
    }
    scores = {name: np.asarray([1.0, 0.0, 0.0, 1.0]) for name in evaluate.SCORE_ORDER}
    report = evaluate.build_report(
        task_type="QA",
        arrays=arrays,
        scores=scores,
        detector={"mechanism_scores_available": True},
        bootstrap=0,
        seed=1,
    )

    assert report["tokens"] == 4
    assert report["hallucinated_tokens"] == 3
    assert report["evaluated_tokens"] == 2
    assert report["evaluated_positives"] == 1
    assert report["prevalence"] == 0.5
    assert all(result["auroc"] == 1 for result in report["detection"].values())


def test_label_length_must_match_frozen_response(tmp_path, monkeypatch):
    records = [
        {
            "sample_id": "s",
            "source_id": "source",
            "task_type": "QA",
            "physical_shard": 0,
            "split_root": tmp_path,
        }
    ]
    frozen = {"response_length": np.asarray([2, 2])}

    class Sample:
        source_id = "source"
        task_type = "QA"
        generator_model = None

        def attention(self):
            return SimpleNamespace(token_ids=torch.arange(5), response_idx=3)

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


def test_sample_identity_checks_metadata_and_exact_target_token_ids():
    attention = SimpleNamespace(token_ids=torch.arange(5), response_idx=3)
    sample = SimpleNamespace(
        source_id="source",
        task_type="Summary",
        generator_model="generator",
        attention=lambda: attention,
    )
    record = {
        "source_id": "source",
        "task_type": "Summary",
        "generator_model": "generator",
        "target_response_sha256": evaluate.target_response_sha256(torch.arange(5), 3),
        "token_ids_sha256": evaluate.token_ids_sha256(torch.arange(5)),
    }
    assert evaluate._validate_sample_identity(record, sample) is attention

    changed = {**record, "target_response_sha256": "not-the-target-digest"}
    with pytest.raises(ValueError, match="response token IDs changed"):
        evaluate._validate_sample_identity(changed, sample)

    prompt_changed = SimpleNamespace(
        token_ids=torch.tensor([9, 1, 2, 3, 4]), response_idx=3
    )
    changed_sample = SimpleNamespace(
        **{**sample.__dict__, "attention": lambda: prompt_changed}
    )
    with pytest.raises(ValueError, match="prompt or response token IDs changed"):
        evaluate._validate_sample_identity(record, changed_sample)


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
    graph = object()
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
    monkeypatch.setattr(evaluate, "build_graph", lambda *_args: graph)
    monkeypatch.setattr(
        evaluate,
        "plot_sample_dashboard",
        lambda record, layers, route_graph, output: seen.update(
            record=record, layers=layers, graph=route_graph, output=output
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
    assert seen["graph"] is graph
    assert "source_flow" not in seen["record"]
    assert "route_interaction" in seen["record"]
    assert "history_support" in seen["record"]
    assert seen["layers"]["edge_evidence_head_entropy"].shape == (3, 2, 2)
    assert "edge_predictor_self_mass_share" in seen["layers"]
    assert "label" not in seen["record"]
