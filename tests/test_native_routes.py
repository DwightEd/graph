"""Scientific regression gates for restored routes, source splits and cache reuse."""

from collections import deque
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from state_audit.storage import read_arrays, read_json, write_json
from test_native_support import Tokenizer, response
from test_native_support import model as _model

from experiments.native_support.calibration import (
    PROTOCOLS,
    crossfit_collapse,
    design,
    source_partitions,
)
from experiments.native_support.comparison import (
    DIRECTORY,
    evaluate_existing,
    run_comparison,
)
from experiments.native_support.comparison_evaluation import group_metrics
from experiments.native_support.pipeline import capture_response
from experiments.native_support.routes import (
    carrier_geometry,
    focus_writes,
    measure_routes,
    route_geometry,
    route_scores,
)

model = _model


def test_restored_route_uses_message_norm_and_preserves_evidence_definition():
    attention = np.array([[[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]])
    magnitude = np.array([[[3., 4., 5.], [6., 8., 10.]]])
    groups = np.array([0, 0, 1])
    scores = route_scores(attention, magnitude, groups, 2, np.array([True, False]))
    assert scores["routing_imbalance"] == pytest.approx(1 / 6)
    assert scores["prompt_routing_imbalance"] == pytest.approx(-1 / 6)
    assert scores["attention_displacement"] == pytest.approx(0.2)
    assert scores["prompt_attention_displacement"] == pytest.approx(-0.1)
    assert scores["strict_history_attention"] == 0
    arrays = {"attention": attention, "edge_value_energy": magnitude ** 2,
              "edge_logit_write": -100 * magnitude, "group_ids": groups, "logit_gap": -3.}
    focus = {"focus_start": np.zeros((1, 2), dtype=int)}
    measured, _, _ = measure_routes(arrays, 2, np.array([True, False]), [], focus, 2)
    assert measured["routing_imbalance"] == scores["routing_imbalance"]
    changed = route_scores(attention, magnitude ** 2, groups, 2, None)
    assert changed["prompt_routing_imbalance"] != scores["prompt_routing_imbalance"]
    assert np.isnan(changed["routing_imbalance"])


def test_geometry_detects_shared_vs_distinct_sources_before_head_pooling():
    shared = np.array([[[1., 0., 0.], [1., 0., 0.]]])
    separate = np.array([[[1., 0., 0.], [0., 1., 0.]]])
    left = carrier_geometry(shared, np.ones(3, dtype=bool))
    right = carrier_geometry(separate, np.ones(3, dtype=bool))
    assert left["effective_sources"].item() == left["effective_rank"].item() == 1
    assert right["effective_sources"].item() == right["effective_rank"].item() == 2
    scaled = carrier_geometry(separate * np.array([[[2.], [7.]]]), np.ones(3, dtype=bool))
    np.testing.assert_allclose(right["effective_rank"], scaled["effective_rank"])


def test_temporal_geometry_uses_observed_anchors_and_special_scope_is_explicit():
    previous = deque(maxlen=3)
    masses = [np.array([[[5., 1., 0., 0.]]]), np.array([[[5., 0., 1., 0.]]])]
    ordinary = np.array([False, True, True])
    for mass in masses:
        geometry, anchors = route_geometry(mass, 3, ordinary, previous, "attention")
        previous.append(anchors)
    assert geometry["attention_legacy_anchor"].item() == 0
    assert geometry["attention_ordinary_anchor"].item() == 2
    assert geometry["attention_ordinary_log_volume"].item() == pytest.approx(np.log(2))


def test_read_and_signed_write_use_same_head_window_and_exclude_specials():
    edges = np.array([[[100., -2., 1., 100.], [100., 100., -3., 99.]]])
    groups = np.array([0, 0, 0, 2])
    result = focus_writes(edges, groups, {"focus_start": np.array([[1, 2]])}, 4, 2)
    np.testing.assert_allclose(result["focus_positive_write"], [[1, 0]])
    np.testing.assert_allclose(result["focus_negative_write"], [[2, 3]])
    bounded = focus_writes(edges, groups, {"focus_start": np.array([[1, 2]])}, 4, 2,
                           np.array([False, False, True, False]))
    np.testing.assert_allclose(bounded["focus_negative_write"], [[0, 3]])


def calibration_records():
    records = []
    for source in range(6):
        random = np.random.default_rng(source)
        record = {"id": str(source), "source_id": str(source), "prompt_length": 20 + source, "tokens": 25}
        for field, _ in PROTOCOLS.values():
            record[field] = random.normal(size=(25, 3))
        records.append(record)
    return records


def test_calibration_holds_entire_source_out_and_causal_score_ignores_its_future():
    records = calibration_records()
    partitions = source_partitions(records)
    for partition in partitions:
        assert not set(partition["test"]) & set(partition["fit"] + partition["calibration"])
        assert not set(partition["fit"]) & set(partition["calibration"])
    before, _ = crossfit_collapse(records)
    changed = deepcopy(records)
    for field, _ in PROTOCOLS.values():
        changed[0][field][5:] += 100
    after, _ = crossfit_collapse(changed)
    for name in ("functional_collapse", "attention_collapse"):
        np.testing.assert_array_equal(before["0"][name][:5], after["0"][name][:5])
    np.testing.assert_array_equal(design(20, 10, "causal")[:5], design(20, 30, "causal")[:5])
    assert not np.array_equal(design(20, 10, "historical_offline")[:5], design(20, 30, "historical_offline")[:5])


def test_too_few_sources_is_unavailable_not_a_zero_score():
    scores, protocol = crossfit_collapse(calibration_records()[:2])
    assert protocol["status"] == "unavailable"
    assert np.isnan(scores["0"]["functional_collapse"]).all()


def test_within_answer_and_source_weighting_are_reported_separately():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([10., 9., 1., 0.])
    measured = group_metrics(labels, scores, np.array(["a", "a", "b", "b"]), np.array(["x", "x", "y", "y"]))
    assert measured["auroc"] == 0.25
    assert measured["within_answer"]["pair_weighted_auroc"] == 0
    assert measured["within_answer"]["mixed_answers"] == 2


def test_comparison_reuses_cache_preserves_v1_and_scores_before_labels(model, tmp_path):
    item = response()
    directory = tmp_path / "responses/0000"
    capture_response(model, Tokenizer(), item, directory, 3)
    settings = {"model": "unused", "responses": [item]}
    write_json(tmp_path / "settings.json", settings)
    write_json(tmp_path / "summary.json", {"v1": "preserve"})
    write_json(tmp_path / "evaluation.json", {"v1": "preserve"})
    annotation = {"sample": {"token_ids": item["token_ids"][7:], "labels": [0, 1, 1, 0, 0, 0]}}
    write_json(tmp_path / "annotations.json", annotation)
    original = (directory / "token_000000.npz").read_bytes()
    with patch("state_audit.model.load_model", side_effect=AssertionError("model load forbidden")):
        result = run_comparison(tmp_path, settings)
    saved = read_arrays(tmp_path / DIRECTORY / "responses/0000/scores.npz")
    assert result["primary_method"] == "prompt_routing_imbalance"
    assert result["evaluation"]["methods"]["support_graph"]["all_error"]["positives"] == 2
    assert (directory / "token_000000.npz").read_bytes() == original
    assert read_json(tmp_path / "summary.json") == read_json(tmp_path / "evaluation.json") == {"v1": "preserve"}
    annotation["sample"]["labels"] = [0, 0, 0, 1, 1, 0]
    write_json(tmp_path / "annotations.json", annotation)
    run_comparison(tmp_path, settings)
    changed = read_arrays(tmp_path / DIRECTORY / "responses/0000/scores.npz")
    for name in ("prompt_routing_imbalance", "risk", "functional_ordinary_log_volume"):
        np.testing.assert_array_equal(saved[name], changed[name])
    refreshed = evaluate_existing(tmp_path)
    assert refreshed["status"] == "evaluated"
    result_path = tmp_path / DIRECTORY / "evaluation.json"
    valid_result = result_path.read_bytes()
    missing = evaluate_existing(tmp_path, tmp_path / "missing.json")
    assert missing["reason"] == "missing_token_annotations"
    assert result_path.read_bytes() == valid_result
    token_header = (tmp_path / DIRECTORY / "responses/0000/tokens.csv").read_text().splitlines()[0]
    assert "functional_collapse" in token_header
