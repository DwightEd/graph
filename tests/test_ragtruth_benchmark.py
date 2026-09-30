"""Compact-native equivalence, full population coverage, and source-disjoint selection."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from state_audit.storage import write_arrays
from experiments.native_support.ragtruth_benchmark.data import TASKS
from experiments.native_support.ragtruth_benchmark.selection import development_sources, choose_readout, SELECTION_METHODS


def test_parameter_selection_rejects_overlapping_sources_and_test_only_cache():
    overlap = [dict(task="QA", split=split, source_id="same") for split in ("train", "test")]
    with pytest.raises(ValueError, match="overlap"):
        development_sources(overlap, "QA")
    with pytest.raises(ValueError, match="train sources"):
        development_sources([dict(task="QA", split="test", source_id="x")], "QA")


def test_development_can_select_a_stronger_route_baseline_instead_of_forcing_local_source(tmp_path):
    records = [dict(id=str(index), source_id=f"source{index}", task="Summary", split="train",
                    directory=f"responses/{index}", tokens=2) for index in range(2)]
    for record in records:
        values = {name: np.array([1., 0.]) for name in SELECTION_METHODS}
        values["raw_route"] = np.array([0., 1.])
        write_arrays(tmp_path / record["directory"] / "scores.npz", token_id=np.array([5, 6]), **values)
    truth = {record["id"]: dict(token_ids=[5, 6], labels=[0, 1]) for record in records}
    with patch("experiments.native_support.ragtruth_benchmark.selection.annotations", return_value=truth):
        selected = choose_readout(tmp_path, dict(records=records), "Summary")
    assert selected["method"] == "raw_route"
    assert selected["weight"] is None
