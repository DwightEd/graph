from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow.routing_events import (
    RoutingEventAudit,
    causal_event_centers,
    matched_event_controls,
    position_adjusted_gap,
)
from experiments.reanchor_flow.scan_dataset import ScanSample


class Captures:
    def __init__(self, split="train", *, duplicate_source=False):
        self.split = split
        identifiers = [("a", "source-a"), ("b", "source-b")]
        if duplicate_source:
            identifiers.append(("c", "source-a"))
        self.records = [SimpleNamespace(sample_id=sample_id, source_id=source_id, task_type="QA") for sample_id, source_id in identifiers]
        self.samples = {}
        self.labels = {}
        self.requested_fields = []
        for sample_id, source_id in identifiers:
            labels = np.repeat([0, 1, 0, 1], 16)
            score = np.zeros((1, 2, 64))
            score[0, 0, [8, 20, 26, 40, 52, 58]] = .9
            transport = np.broadcast_to([1., 2., 1., 4.], (1, 2, 64, 4)).copy()
            transport[0, 0, score[0, 0] > 0] = [4., 1., 2., 1.]
            unit = np.zeros((1, 2, 64, 4), dtype=int)
            self.samples[sample_id] = ScanSample(
                sample_id, source_id, "QA", 10, np.arange(9, 73),
                {"reanchor_score": score, "reanchor_bucket_transport": transport,
                 "reanchor_bucket_source_unit_id": unit,
                 "reanchor_bucket_source_position": np.zeros_like(unit)},
                {"local_window": 4},
            )
            self.labels[sample_id] = labels

    def load(self, sample_id, *, fields):
        self.requested_fields.append(fields)
        return self.samples[sample_id]

    def sample_weights(self):
        return {record.sample_id: 1 / sum(other.source_id == record.source_id for other in self.records) for record in self.records}


class Labels:
    def __init__(self, captures):
        self.captures = captures

    def load(self, sample):
        return self.captures.labels[sample.sample_id]


def test_event_centers_are_causal_and_prefix_stable():
    score = np.zeros(30)
    score[[2, 3, 6, 8, 12, 18, 19, 25]] = [.6, .9, .7, .8, .7, .9, .7, .9]
    full = causal_event_centers(score, .6)
    np.testing.assert_array_equal(full, [2, 8, 18, 25])
    for end in range(1, len(score) + 1):
        np.testing.assert_array_equal(causal_event_centers(score[:end], .6), full[full < end])


def test_position_only_group_difference_has_no_paired_strata():
    position = np.arange(31)
    labels = (position >= 15).astype(int)
    score = position[None, None, :] / 31
    assert position_adjusted_gap(score, labels, position) is None


def test_matched_controls_are_nearby_same_label_and_unique():
    score = np.zeros(45)
    centers = np.array([10, 17, 34])
    score[centers] = .8
    labels = np.r_[np.zeros(25), np.ones(20)]
    pairs = matched_event_controls(score, labels, centers)
    assert len(pairs) == 3
    assert len({control for _, control in pairs}) == len(pairs)
    for event, control in pairs:
        assert labels[event] == labels[control]
        assert abs(event - control) <= 16
        assert min(abs(control - centers)) > 4
        assert score[control] == 0


def test_train_head_selection_is_source_balanced_and_test_frozen(tmp_path):
    train = Captures()
    audit = RoutingEventAudit(max_heads=1).discover(train, Labels(train))
    repeated = Captures(duplicate_source=True)
    duplicate_audit = RoutingEventAudit(max_heads=1).discover(repeated, Labels(repeated))
    assert audit.selection == duplicate_audit.selection
    selected = deepcopy(audit.selection)
    assert selected["QA"][0]["head"] == 0
    assert selected["QA"][0]["train_paired_sources"] == 2
    assert selected["QA"][0]["threshold"] == .9
    test = Captures("test")
    for sample in test.samples.values():
        sample["reanchor_score"][0, 1] = np.arange(64)
    report = audit.run(test, Labels(test), tmp_path, plot=True)
    assert audit.selection == selected
    assert report["status"] == "exploratory_structural_audit"
    entry = report["tasks"]["QA"][0]
    assert entry["head"] == 0
    assert entry["paired_hallucinated_minus_nonhallucinated_sources"] == 2
    for name in ("nonhallucinated", "hallucinated"):
        group = entry["groups"][name]
        assert group["source_count"] == 2
        assert group["matched_pair_count"] > 0
        assert group["mean_absolute_control_offset"] <= 16
        assert np.shape(group["event"]["mean"]) == (7, 21)
        assert np.isclose(group["event"]["mean"][5][8], .9)
        assert group["control"]["mean"][5][8] == 0
    assert (tmp_path / "event_audit.json").is_file()
    assert (tmp_path / entry["figure"]).stat().st_size > 1000
    event_index = [json.loads(line) for line in (tmp_path / report["event_index"]).read_text().splitlines()]
    assert len(event_index) == 12
    assert all(event["head"] == 0 for event in event_index)
    assert all(event["predicted_token_position"] == event["predictor_position"] + 1 for event in event_index)
    assert any(event["control_response_index"] is not None for event in event_index)


def test_missing_paired_labels_produces_explicit_empty_task(tmp_path):
    captures = Captures()
    for sample_id in captures.labels:
        captures.labels[sample_id][:] = 0
    audit = RoutingEventAudit().discover(captures, Labels(captures))
    report = audit.run(captures, Labels(captures), tmp_path, plot=False)
    assert report["selection"]["QA"] == []
    assert report["tasks"]["QA"] == []


def test_test_split_cannot_select_heads():
    test = Captures("test")
    with pytest.raises(ValueError, match="train"):
        RoutingEventAudit().discover(test, Labels(test))
