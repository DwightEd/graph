from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_protocol import file_sha256
from experiments.attention_operator_validation.evaluate import (
    frozen_evaluation_spec,
    grouped_probe_report,
    validate_frozen_evaluation_spec,
    validate_label_free_bindings,
)

from .helpers import FEATURE_NAMES, make_table


class _Sample:
    def __init__(self, source_id, task_type, response_length):
        self.source_id = source_id
        self.task_type = task_type
        self._attention = SimpleNamespace(num_response_tokens=response_length)
        self.released = False

    def attention(self):
        return self._attention

    def release_attention(self):
        self.released = True


class _Dataset:
    def __init__(self, root):
        self.root = root
        self.manifest = {"split": "test"}
        self.samples = {
            "sample-a": _Sample("source-a", "QA", 2),
            "sample-b": _Sample("source-b", "QA", 3),
        }
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]


def test_label_free_binding_checks_every_canonical_field(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"split":"test"}\n', encoding="utf-8")
    dataset = _Dataset(tmp_path)
    table = make_table(manifest_sha256=file_sha256(manifest))

    actual = validate_label_free_bindings(table, dataset)

    assert actual.tolist() == ["source-a", "source-b"]
    assert all(sample.released for sample in dataset.samples.values())

    invalid = replace(
        table,
        response_length=np.asarray((2, 4), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="response lengths"):
        validate_label_free_bindings(invalid, _Dataset(tmp_path))


def test_frozen_directions_and_groups_must_match_registered_spec():
    directions, groups = frozen_evaluation_spec(FEATURE_NAMES)
    actual = validate_frozen_evaluation_spec(FEATURE_NAMES, directions, groups)
    assert actual == (directions, groups)

    changed = {**directions, FEATURE_NAMES[0]: "high"}
    with pytest.raises(ValueError, match="directions"):
        validate_frozen_evaluation_spec(FEATURE_NAMES, changed, groups)


def test_grouped_probe_rejects_partially_valid_label_conditioned_folds():
    feature = np.arange(18, dtype=np.float64).reshape(3, 6)
    report = grouped_probe_report(
        feature,
        FEATURE_NAMES,
        np.asarray((1, 0, 0), dtype=np.int8),
        np.asarray(("positive-only", "negative-a", "negative-b"), dtype=str),
        folds=3,
        bootstrap_replicates=4,
        seed=17,
        response_length=np.asarray((2, 3, 4), dtype=np.int32),
    )

    assert report["available"] is False


def test_grouped_probe_scores_all_answers_and_controls_length():
    random = np.random.default_rng(23)
    feature = random.normal(size=(8, len(FEATURE_NAMES)))
    label = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int8)
    source = np.asarray([f"source-{index}" for index in range(8)], dtype=str)
    report = grouped_probe_report(
        feature,
        FEATURE_NAMES,
        label,
        source,
        folds=4,
        bootstrap_replicates=4,
        seed=29,
        response_length=np.arange(2, 10, dtype=np.int32),
    )

    assert report["available"] is True
    assert report["response_length_only"]["answers_scored"] == len(label)
    assert report["routing_only"]["answers_scored"] == len(label)
    assert report["routing_only"]["includes_response_length"] is True
