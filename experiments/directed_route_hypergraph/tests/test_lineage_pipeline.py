from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.directed_route_hypergraph.lineage_pipeline import (
    SCORE_SCHEMA,
    TRACE_SCHEMA,
    conditional_high_tail,
    encoded_to_token_graph,
    evaluate_scores,
    export_trace,
    load_frozen_labels,
    onset_diagnostics,
    require_artifact,
    score_traces,
    source_balanced_high_tail,
    trace_graph,
)
from experiments.directed_route_hypergraph.routing_lineage import UNRESOLVED
from experiments.grounded_route.artifacts import (
    EncodedTokenGraph,
    load_npz,
    merge_embedding_index,
    save_npz,
    save_embedding_index,
    save_encoded_graph,
    sha256,
)
from experiments.grounded_route.tests.helpers import make_graph


def encoded(sample_id: str, source_id: str, *, embedding_value: float):
    graph = replace(
        make_graph(layers=3, heads=2, response_count=5),
        sample_id=sample_id,
        source_id=source_id,
    ).canonicalize()
    # The lineage pipeline must not use either learned tensor in this object.
    output = SimpleNamespace(
        node_embedding=torch.full((graph.token_count, 7), embedding_value),
        lineage=torch.full(
            (graph.response_count, graph.layer_count, graph.head_count, 3),
            embedding_value,
        ),
    )
    return graph, EncodedTokenGraph.from_output(graph, output)


def bundle(root, split: str, scope: str, samples):
    graph_dir = root / "graphs"
    graph_dir.mkdir(parents=True)
    paths = []
    hashes = []
    for number, sample in enumerate(samples):
        relative = f"graphs/{number:08d}.pt"
        save_encoded_graph(root / relative, sample)
        paths.append(relative)
        hashes.append(sha256(root / relative))
    index_path = root / "index.npz"
    metadata = {
        "dataset_manifest_sha256": "a" * 64,
        "split": split,
        "scope": scope,
        "encoded_graph_sample_ids": [sample.sample_id for sample in samples],
        "encoded_graph_paths": paths,
        "encoded_graph_sha256": hashes,
    }
    if split == "test":
        metadata.update(
            audit_scope="selected_samples",
            reserved_source_ids=("train-source",),
            test_source_ids=tuple(sorted({sample.source_id for sample in samples})),
            test_sample_ids=tuple(sample.sample_id for sample in samples),
        )
    save_embedding_index(index_path, merge_embedding_index(samples), **metadata)
    return index_path


def test_export_trace_keeps_complete_rows_and_predecessor_alignment(tmp_path):
    graph, sample = encoded("sample-a", "source-a", embedding_value=999.0)
    index = bundle(tmp_path / "bundle", "test", "all", [sample])
    output = tmp_path / "trace.npz"

    report = export_trace(index, output, seed=11)
    trace = require_artifact(output, TRACE_SCHEMA)

    assert report["nodes"] == graph.response_count
    assert report["available_nodes"] == graph.response_count - 1
    assert trace["ordered_representation"].shape == (
        graph.response_count,
        graph.layer_count * 4,
    )
    assert trace["routing_representation"].shape == (
        graph.response_count,
        graph.layer_count * 17,
    )
    assert trace["routing_entropy_lower"].shape == (
        graph.response_count,
        graph.layer_count,
    )
    assert trace["routing_entropy_upper"].shape == trace["routing_entropy_lower"].shape
    assert trace["routing_concentration_lower"].shape == (
        graph.response_count,
        graph.layer_count,
    )
    assert trace["routing_concentration_upper"].shape == trace[
        "routing_concentration_lower"
    ].shape
    assert trace["routing_role_mass"].shape == (
        graph.response_count,
        graph.layer_count,
        4,
    )
    assert trace["routing_head_role_std"].shape == trace["routing_role_mass"].shape
    assert trace["routing_role_js"].shape == (
        graph.response_count,
        graph.layer_count,
    )
    assert not trace["available"][0]
    assert trace["available"][1:].all()
    assert trace["predictor_token_index"].tolist() == [-1, 0, 1, 2, 3]
    np.testing.assert_array_equal(
        trace["response_token_id"], graph.response_token_ids.numpy()
    )
    np.testing.assert_array_equal(
        trace["ordered_lineage"][0], np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    assert trace["ordered_representation"][0].reshape(-1, 4)[
        :, UNRESOLVED
    ].tolist() == [1.0, 1.0, 1.0]
    assert not trace["routing_entropy_lower"][0].any()
    assert not trace["routing_entropy_upper"][0].any()
    assert not trace["routing_role_mass"][0].any()
    routing = trace["routing_representation"].reshape(
        graph.response_count,
        graph.layer_count,
        17,
    )
    np.testing.assert_allclose(routing[..., :4], trace["ordered_representation"].reshape(
        graph.response_count,
        graph.layer_count,
        4,
    ))
    np.testing.assert_allclose(
        routing[..., 4:6],
        np.stack((trace["routing_entropy_lower"], trace["routing_entropy_upper"]), axis=-1),
    )
    assert bool(np.asarray(trace["drift_observed"]).item())
    assert bool(np.asarray(trace["dispersion_observed"]).item())
    assert not bool(np.asarray(trace["parametric_bias_observed"]).item())
    assert not np.isclose(trace["ordered_representation"], 999.0).any()
    assert "last_layer" in trace["controls"].astype(str).tolist()
    assert trace["posthoc_same_token_lineage"].shape == (graph.response_count, 4)


def test_artifact_boundary_rejects_misalignment_and_reversed_bounds(tmp_path):
    _, sample = encoded("sample-a", "source-a", embedding_value=1.0)
    index = bundle(tmp_path / "bundle", "test", "all", [sample])
    output = tmp_path / "trace.npz"
    export_trace(index, output)
    arrays = load_npz(output)

    predictor = arrays["predictor_token_index"].copy()
    predictor[1] = 99
    misaligned = tmp_path / "misaligned.npz"
    save_npz(misaligned, **{**arrays, "predictor_token_index": predictor})
    with pytest.raises(ValueError, match="predecessor-query aligned"):
        require_artifact(misaligned, TRACE_SCHEMA)

    lower = arrays["routing_entropy_lower"].copy()
    upper = arrays["routing_entropy_upper"].copy()
    lower[1, 0] = 1.0
    upper[1, 0] = 0.0
    reversed_bounds = tmp_path / "reversed-bounds.npz"
    save_npz(
        reversed_bounds,
        **{
            **arrays,
            "routing_entropy_lower": lower,
            "routing_entropy_upper": upper,
        },
    )
    with pytest.raises(ValueError, match="lower bound exceeds"):
        require_artifact(reversed_bounds, TRACE_SCHEMA)


def test_source_balanced_tail_gives_each_source_equal_weight():
    reference = np.asarray([0.0] * 9 + [10.0])
    source = np.asarray(["long"] * 9 + ["short"])

    probability = source_balanced_high_tail(reference, source, np.asarray([5.0]))

    # long: (0 exceed + 1)/(9 + 1); short: (1 + 1)/(1 + 1)
    np.testing.assert_allclose(probability, [0.55])


def test_random_layer_order_is_fixed_across_samples():
    first = replace(make_graph(layers=4, heads=2), sample_id="first")
    second = replace(make_graph(layers=4, heads=2), sample_id="second")

    first_trace = trace_graph(first, seed=31)
    second_trace = trace_graph(second, seed=31)

    np.testing.assert_array_equal(
        first_trace["random_layer_layer_order"],
        second_trace["random_layer_layer_order"],
    )


def test_conditional_calibration_uses_absolute_bins_then_task_fallback():
    probability, rows, sources, fallback = conditional_high_tail(
        calibration_score=np.asarray([0.0, 4.0, 10.0]),
        calibration_source=np.asarray(["a", "b", "c"]),
        calibration_task=np.asarray(["QA", "QA", "Other"]),
        calibration_position=np.asarray([1, 2, 1]),
        test_score=np.asarray([3.0, 3.0]),
        test_task=np.asarray(["QA", "QA"]),
        test_position=np.asarray([3, 40]),
        position_bin_width=16,
        minimum_reference_sources=2,
    )

    assert rows.tolist() == [2, 2]
    assert sources.tolist() == [2, 2]
    assert fallback.tolist() == [0, 1]
    np.testing.assert_allclose(probability, [0.75, 0.75])


def test_score_freezes_complete_label_free_control_rows(tmp_path):
    _, calibration_a = encoded("cal-a", "cal-source-a", embedding_value=1.0)
    _, calibration_b = encoded("cal-b", "cal-source-b", embedding_value=2.0)
    _, test = encoded("test-a", "test-source", embedding_value=3.0)
    calibration_index = bundle(
        tmp_path / "calibration", "train", "calibration", [calibration_a, calibration_b]
    )
    test_index = bundle(tmp_path / "test", "test", "all", [test])
    calibration_trace = tmp_path / "calibration-trace.npz"
    test_trace = tmp_path / "test-trace.npz"
    score_path = tmp_path / "scores.npz"
    export_trace(calibration_index, calibration_trace, seed=17)
    export_trace(test_index, test_trace, seed=17)

    report = score_traces(
        calibration_trace,
        test_trace,
        score_path,
        position_bin_width=2,
    )
    scores = require_artifact(score_path, SCORE_SCHEMA)

    assert report["labels_read"] is False
    assert len(scores["score"]) == test.response_count
    assert scores["score"][0] == 0.0
    assert scores["ordered_tail_pvalue"][0] == 1.0
    assert scores["calibration_fallback_level"][0] == -1
    assert np.isfinite(scores["score"]).all()
    for control in scores["controls"].astype(str):
        assert f"{control}_conditional_score" in scores
        assert f"{control}_lineage" in scores
    assert "posthoc_same_token_conditional_score" in scores
    assert scores["routing_representation"].shape[1] == 3 * 17
    assert scores["dispersion_entropy_lower_conditional_score"][0] == 0.0
    assert scores["dispersion_entropy_upper_conditional_score"][0] == 0.0
    assert np.isfinite(scores["dispersion_entropy_lower_raw"]).all()
    assert np.isfinite(scores["dispersion_entropy_upper_raw"]).all()
    assert np.isfinite(scores["dispersion_role_js_raw"]).all()
    assert "dispersion_role_js_conditional_score" in scores
    np.testing.assert_array_equal(
        scores["absolute_position_score"], scores["token_index"]
    )
    np.testing.assert_allclose(
        scores["relative_position_offline_score"],
        scores["token_index"] / np.maximum(scores["response_length"] - 1, 1),
    )
    np.testing.assert_array_equal(
        scores["absolute_sequence_position_score"],
        scores["prompt_length"] + scores["token_index"],
    )
    np.testing.assert_array_equal(
        scores["response_length_offline_score"], scores["response_length"]
    )
    assert scores["drift_score"].tolist() == scores["score"].tolist()
    assert not bool(np.asarray(scores["mechanisms_combined"]).item())


def test_score_rejects_calibration_source_leakage(tmp_path):
    _, calibration = encoded("cal-a", "shared-source", embedding_value=1.0)
    _, test = encoded("test-a", "shared-source", embedding_value=2.0)
    calibration_index = bundle(
        tmp_path / "calibration", "train", "calibration", [calibration]
    )
    test_index = bundle(tmp_path / "test", "test", "all", [test])
    calibration_trace = tmp_path / "calibration-trace.npz"
    test_trace = tmp_path / "test-trace.npz"
    export_trace(calibration_index, calibration_trace)
    export_trace(test_index, test_trace)

    with pytest.raises(ValueError, match="share source groups"):
        score_traces(calibration_trace, test_trace, tmp_path / "scores.npz")


def test_evaluation_aligns_complete_labels_before_masking_boundary(tmp_path, monkeypatch):
    _, calibration = encoded("cal-a", "cal-source", embedding_value=1.0)
    _, test = encoded("test-a", "test-source", embedding_value=2.0)
    calibration_index = bundle(
        tmp_path / "calibration", "train", "calibration", [calibration]
    )
    test_index = bundle(tmp_path / "test", "test", "all", [test])
    calibration_trace = tmp_path / "calibration-trace.npz"
    test_trace = tmp_path / "test-trace.npz"
    score_path = tmp_path / "scores.npz"
    export_trace(calibration_index, calibration_trace)
    export_trace(test_index, test_trace)
    score_traces(calibration_trace, test_trace, score_path, position_bin_width=2)

    observed = {}

    def labels_after_freeze(frozen, arrays, test_root):
        path = frozen.artifact.path
        observed["score_exists"] = path.exists()
        observed["rows_seen"] = len(arrays["sample_id"])
        label = np.asarray([0, 0, 1, 0, 1], dtype=np.int8)
        labels = SimpleNamespace(
            token_label=label,
            source_id=arrays["source_id"].astype(str),
        )
        return labels

    monkeypatch.setattr(
        "experiments.directed_route_hypergraph.lineage_evaluation.load_frozen_labels",
        labels_after_freeze,
    )
    output = tmp_path / "evaluation.json"
    report = evaluate_scores(
        tmp_path / "unused-test-root",
        score_path,
        output,
        bootstrap_replicates=5,
    )

    assert observed == {"score_exists": True, "rows_seen": 5}
    assert report["tokens_complete"] == 5
    assert report["tokens_evaluated"] == 4
    assert report["unavailable_boundary_tokens"] == 1
    assert "ordered_minus_reverse" in report["paired_deltas"]
    assert "ordered_minus_posthoc_same_token" in report["paired_deltas"]
    assert "entropy_lower_bound" in report["dispersion_detection"]
    assert "entropy_upper_bound" in report["dispersion_detection"]
    assert "head_role_js" in report["dispersion_detection"]
    assert report["primary"]["combined_score"] is None
    assert report["observability"]["drift_observed"] is True
    assert report["observability"]["dispersion_observed"] is True
    assert report["observability"]["parametric_bias_observed"] is False
    assert report["dispersion_audit"]["mechanisms_combined"] is False
    assert "response_ordinal" in report["position_baselines"]
    assert "absolute_sequence_position" in report["position_baselines"]
    assert "response_length_offline" in report["position_baselines"]
    assert output.exists()

    malformed_arrays = load_npz(score_path)
    malformed_arrays.pop("ordered_conditional_score")
    malformed = tmp_path / "malformed-scores.npz"
    save_npz(malformed, **malformed_arrays)
    observed.clear()
    with pytest.raises(ValueError, match="missing ordered_conditional_score"):
        evaluate_scores(
            tmp_path / "unused-test-root",
            malformed,
            tmp_path / "unused-evaluation.json",
            bootstrap_replicates=5,
        )
    assert observed == {}


def test_token_identity_is_checked_before_labels_are_projected(monkeypatch):
    class FakeSample:
        def attention(self):
            return SimpleNamespace(
                token_ids=torch.tensor([10, 20, 21, 22]),
                response_idx=1,
            )

        def release_attention(self):
            pass

    class FakeDataset:
        def __getitem__(self, sample_id):
            assert sample_id == "sample"
            return FakeSample()

    projected = []
    frozen = SimpleNamespace(
        align_loaded=lambda dataset, arrays: projected.append(True)
    )
    arrays = {
        "sample_id": np.asarray(["sample"] * 3),
        "token_index": np.arange(3),
        "response_length": np.full(3, 3),
        "response_token_id": np.asarray([20, 999, 22]),
    }
    monkeypatch.setattr(
        "experiments.directed_route_hypergraph.lineage_evaluation.open_research_dataset",
        lambda *args, **kwargs: FakeDataset(),
    )

    with pytest.raises(ValueError, match="token IDs differ"):
        load_frozen_labels(frozen, arrays, "unused")
    assert projected == []


def test_onset_does_not_replace_an_unavailable_token_zero_span():
    result = onset_diagnostics(
        label=np.asarray([1, 0, 1, 0], dtype=np.int8),
        score=np.asarray([0.0, 0.1, 0.9, 0.2]),
        sample_id=np.asarray(["sample"] * 4),
        token_index=np.arange(4),
        available=np.asarray([False, True, True, True]),
    )

    assert result["responses_with_first_onset"] == 1
    assert result["first_onsets_unavailable_token_zero"] == 1
    assert result["first_onsets_available"] == 0
    assert result["matched_correct_predecessors"] == 0
