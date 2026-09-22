"""Separate readout shrinkage from segmentation, and freeze real-data selection."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from state_audit.storage import read_arrays, read_json, write_json
from test_joint_state import write_compact_fixture
from test_native_support_evaluation import official_fixture as _official_fixture

from experiments.native_support.run import main
from experiments.native_support.state_model import run_state_model
from experiments.native_support.state_readout import (
    DIRECTORY,
    observed_readout,
    run_readout,
)
from experiments.native_support.switching import switching_filter
from experiments.native_support.validation import choose_sources, select_cohorts

official_fixture = _official_fixture


def test_readout_uses_observed_values_without_direct_prior_shrinkage():
    route = np.full(12, -.4)
    state = switching_filter(route[:, None], np.array([.8]), np.eye(1))
    observed, _ = observed_readout(route, state["run_posterior"])
    np.testing.assert_allclose(observed, route)
    assert state["state_mean"][0, 0] == pytest.approx(.2)
    independent = np.zeros((12, 12))
    independent[:, 0] = 1
    values = np.linspace(-.6, .8, 12)
    np.testing.assert_allclose(observed_readout(values, independent)[0], values)
    prefix = np.eye(12)
    np.testing.assert_allclose(observed_readout(values, prefix)[0], np.cumsum(values) / np.arange(1, 13))


def test_fixed_posterior_readout_is_causal_and_matches_current_coefficient():
    random = np.random.default_rng(9)
    values = random.uniform(-1, 1, 20)
    posterior = np.zeros((20, 20))
    for target in range(20):
        posterior[target, :target + 1] = random.dirichlet(np.ones(target + 1))
    score, coefficient = observed_readout(values, posterior)
    changed = values.copy()
    changed[9:] += 2
    np.testing.assert_allclose(observed_readout(changed, posterior)[0][:9], score[:9])
    changed = values.copy()
    changed[9] += .1
    altered = observed_readout(changed, posterior)[0]
    assert altered[9] - score[9] == pytest.approx(.1 * coefficient[9])
    for target, value in enumerate(score):
        assert values[:target + 1].min() - 1e-12 <= value <= values[:target + 1].max() + 1e-12


def test_saved_state_readout_keeps_old_scores_and_never_recomputes_states(tmp_path, capsys):
    settings, annotations = write_compact_fixture(tmp_path)
    run_state_model(tmp_path, settings)
    original_file = tmp_path / "joint_state_v4/w16/responses/0000/scores.npz"
    original = original_file.read_bytes()
    with patch("experiments.native_support.state_model.switching_filter", side_effect=AssertionError("state recomputation forbidden")):
        main(["--stage", "readout", "--output", str(tmp_path)])
    first = read_arrays(tmp_path / DIRECTORY / "w16/responses/0000/scores.npz")
    assert original_file.read_bytes() == original
    for name, values in read_arrays(original_file).items():
        np.testing.assert_array_equal(first[name], values)
    annotations["0"]["labels"] = [1, 0, 0, 1, 0, 0]
    write_json(tmp_path / "annotations.json", annotations)
    run_readout(tmp_path)
    after = read_arrays(tmp_path / DIRECTORY / "w16/responses/0000/scores.npz")
    np.testing.assert_array_equal(first["joint_observed"], after["joint_observed"])
    main(["--stage", "evaluate", "--output", str(tmp_path)])
    assert "joint_observed" in capsys.readouterr().out


def test_source_selection_is_label_independent_and_disjoint(tmp_path):
    rows, sources = [], {}
    for split in ("train", "test"):
        for source in range(9):
            identity = str(source)
            sources[identity] = {"task_type": "QA"}
            rows.append({"id": f"{split}_{source}", "source_id": identity,
                         "split": split, "model": "generator", "labels": [source % 2]})
    write_json(tmp_path / "settings.json", {"responses": [{"id": "seen", "source_id": "0"}]})
    args = SimpleNamespace(exclude_output=tmp_path, task="QA", generator="generator",
                           reference_count=3, limit=3, selection_seed=37)
    groups, excluded = select_cohorts(args, rows, sources)
    selected_ids = lambda data: {name: [r["id"] for r in selected] for name, selected in data.items()}
    changed = deepcopy(rows)
    for row in changed:
        row["labels"] = []
    again, _ = select_cohorts(args, list(reversed(changed)), sources)
    assert selected_ids(groups) == selected_ids(again)
    reference = {r["source_id"] for r in groups["reference"]}
    target = {r["source_id"] for r in groups["test"]}
    assert not reference & target and not (reference | target) & set(excluded)
    with pytest.raises(ValueError, match="Requested 10"):
        choose_sources(rows, sources, "QA", "test", "generator", 10, set(), 37)


def test_validation_cli_prepares_and_runs_with_external_reference(official_fixture, tmp_path, capsys):
    import torch

    torch.set_num_threads(1)
    args, model, rows, sources = official_fixture
    expanded_rows, expanded_sources = [], []
    for split in ("train", "test"):
        for index in range(2):
            identity = f"{split}_{index}"
            expanded_sources.append(dict(sources["s1"], source_id=identity))
            expanded_rows.append(dict(rows[index], id=identity, source_id=identity, split=split))
    (args.dataset / "source_info.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in expanded_sources))
    (args.dataset / "response.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in expanded_rows))
    inspected = tmp_path / "inspected"
    write_json(inspected / "settings.json", {"responses": [{"id": "seen", "source_id": "s1"}]})
    output = tmp_path / "validation"
    command = ["--stage", "validate", "--dataset", str(args.dataset), "--model", str(model),
               "--output", str(output), "--exclude-output", str(inspected), "--reference-count", "2",
               "--limit", "2", "--device", "cpu", "--dtype", "float32", "--resume"]
    with patch("state_audit.model.load_model", side_effect=AssertionError("prepare must not load weights")):
        main(command + ["--prepare-only"])
    main(command)
    summary = read_json(output / "summary.json")
    assert summary["candidate_method"] == "joint_observed"
    assert summary["reference_mode"] == "external_reference_with_overlapping_sources_excluded"
    assert summary["cohort"]["labels_used_for_selection"] is False
    assert summary["evaluation"]["methods"]["joint_observed"]["all_error"]["positives"] == 1
    with patch("state_audit.model.load_model", side_effect=AssertionError("resume must reuse cache")):
        main(command)
    assert "joint_observed" in capsys.readouterr().out
