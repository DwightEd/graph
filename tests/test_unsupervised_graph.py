"""Scientific invariants: label isolation, graph boundaries and contrast pairing."""

import numpy as np
import torch
import json
from types import SimpleNamespace
import pytest

from experiments.unsupervised_graph.data import INPUTS, NodeFeatures, load_inputs, matched_donors, neighbor_mean, strata
from experiments.unsupervised_graph.model import PairNetwork, rank_scores


def test_graph_excludes_self_and_other_answers():
    matrix = np.array([[1.], [3.], [9.], [100.]], dtype=np.float32)
    answers = np.array([0, 0, 0, 1])
    actual = neighbor_mean(matrix, answers, radius=1)
    np.testing.assert_allclose(actual[:, 0], [3, 5, 3, 0])


def test_shuffled_graph_is_not_feature_shuffle():
    matrix = np.arange(20, dtype=np.float32)[:, None]
    answer = np.repeat([0, 1], 10)
    actual = neighbor_mean(matrix, answer, radius=1, shuffled=True)
    assert actual[:10].max() < 10
    assert actual[10:].min() >= 10
    assert not np.allclose(actual, neighbor_mean(matrix, answer, radius=1))
    np.testing.assert_array_equal(actual, neighbor_mean(matrix, answer, radius=1, shuffled=True))


def test_donors_match_nuisance_and_exclude_source():
    count = 320
    pack = dict(context=np.column_stack((np.zeros((count, 2)), np.tile(np.linspace(0, 1, 8), 40))),
                generator=np.repeat([0, 1], 160), source_index=np.repeat(np.arange(40), 8))
    donors = matched_donors(pack, np.random.default_rng(42))
    np.testing.assert_array_equal(strata(pack)[donors], strata(pack))
    assert np.all(pack["source_index"][donors] != pack["source_index"])


def test_fresh_features_are_123_and_finite_without_labels():
    rng = np.random.default_rng(42)
    context = rng.normal(size=(100, 6))
    observations = rng.normal(size=(100, 11))
    transform = NodeFeatures().fit(context, observations)
    matrix = transform.transform(context, observations)
    assert matrix.shape == (100, 123)
    assert np.isfinite(matrix).all()
    assert matrix.dtype == np.float32


def test_pair_network_has_finite_trainable_gradients():
    torch.manual_seed(42)
    model = PairNetwork()
    nodes, neighbors = torch.randn(12, 123), torch.randn(12, 123)
    loss = torch.nn.functional.softplus(-model(nodes, neighbors)).mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()) > 0


def test_midrank_ties_and_fixed_orientation():
    scores = {name: np.array([0., 1.]) for name in ("pair", "route")}
    reference = {name: np.array([0., 0., 1., 1.]) for name in scores}
    ranked = rank_scores(scores, reference)
    np.testing.assert_allclose(ranked["pair"], [.25, .75])
    np.testing.assert_allclose(ranked["fixed_unsupervised"], [.25, .75])


def test_input_loader_never_accesses_label_members(tmp_path, monkeypatch):
    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, name):
            assert name in INPUTS, f"Unexpected training input access: {name}"
            return np.zeros(3)

    monkeypatch.setattr(np, "load", lambda *args, **kwargs: Archive())
    metadata = dict(records=[dict(generator="model", packed_start=0, packed_stop=3)])
    (tmp_path / "QA_train.json").write_text(json.dumps(metadata))
    pack, _ = load_inputs(tmp_path, "QA", "train")
    assert not {"labels", "onsets", "firsts", "baselines"} & set(pack)


def test_selection_cannot_change_after_test_freeze(tmp_path):
    from experiments.unsupervised_graph.run import develop_task
    (tmp_path / "QA").mkdir()
    (tmp_path / "QA/test_scores.npz").touch()
    with pytest.raises(FileExistsError, match="frozen"):
        develop_task(SimpleNamespace(output=tmp_path), "QA", fuse=True)
