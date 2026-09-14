import copy
import json

import numpy as np
import pytest
from scipy import sparse

from experiments.unsupervised_token_graph.data import ResponseCache, ResponseRecord
from experiments.unsupervised_token_graph.information_flow import (
    RoutingResidual, FlowGraph, SourceCalibration, build_flow_graph, source_weights,
)
from experiments.unsupervised_token_graph.flow_run import read_roster, score_roster, verify_freeze
from experiments.unsupervised_token_graph.flow_evaluate import _metrics, evaluate_frozen, paired_source_difference


def record():
    a = np.zeros((2, 2, 7, 7))
    for channel, matrix in enumerate(a.reshape(-1, 7, 7)):
        for q in range(7):
            matrix[q, q] = 0.2 + channel * 0.02
            if q:
                matrix[q, :q] = (1 - matrix[q, q]) / q
    return ResponseRecord("answer", "source", a, 3, prompt_length=3)


def as_csr(r):
    ptr, col, val = [0], [], []
    for a in r.attention.reshape(-1, len(r.attention[0, 0]), len(r.attention[0, 0])):
        for q in range(r.response_idx, len(a)):
            c = np.flatnonzero(a[q])
            col.extend(c); val.extend(a[q, c]); ptr.append(len(col))
    result = copy.copy(r)
    result.attention = None
    result.sparse = dict(response_row_ptr=np.array(ptr), response_column_indices=np.array(col),
                         response_values=np.array(val), attention_diagonal=r.attention.diagonal(axis1=-2, axis2=-1))
    return result


def linear_graph(seed, n=16, source=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    row, col, val = [], [], []
    for q in range(3, n):
        j = int(rng.integers(0, 3))
        row.append(q); col.append(j); val.append(0.8)
        x[q] = 0.8 * x[j] + rng.normal(scale=0.01, size=3)
    return FlowGraph(str(seed), source or str(seed), x,
                     sparse.csr_matrix((val, (row, col)), shape=(n, n)), 3,
                     np.full(n, 0.2), "hidden", "synthetic-linear-v1")


def test_dense_csr_equivalence_and_retained_mass():
    r = record()
    dense = build_flow_graph(r, floor=0.15, attribute="diagonal")
    compressed = build_flow_graph(as_csr(r), floor=0.15, attribute="diagonal")
    np.testing.assert_allclose(dense.x, compressed.x)
    np.testing.assert_allclose(dense.adjacency[3:].toarray(), compressed.adjacency[3:].toarray())
    np.testing.assert_allclose(dense.routing()[3:], compressed.routing()[3:])
    assert dense.routing()[-1, -1] > 0  # omitted mass is not normalized away
    assert dense.adjacency.nnz < 7 * 7
    np.testing.assert_allclose(dense.routing().sum(axis=1), 1)


@pytest.mark.parametrize("bad", ["future", "pointer", "nan"])
def test_invalid_sparse_rejected(bad):
    r = as_csr(record())
    if bad == "future":
        r.sparse["response_column_indices"][0] = 6
    elif bad == "pointer":
        r.sparse["response_row_ptr"][0] = 1
    else:
        r.sparse["response_values"][0] = np.nan
    with pytest.raises(ValueError):
        build_flow_graph(r)


def test_rectangular_query_alignment_is_not_guessed():
    r = record()
    r.attention = r.attention[:, :, 2:6, :]
    with pytest.raises(ValueError, match="physical-token"):
        build_flow_graph(r)


def test_label_arrays_are_not_deserialized(tmp_path):
    r = record()
    path = tmp_path / "r.npz"
    # Reading this object array would fail with allow_pickle=False.
    np.savez(path, attention=r.attention, response_idx=3, labels=np.array([{}], dtype=object))
    result = ResponseCache().load(path)
    assert "labels" not in result.metadata
    assert build_flow_graph(result, attribute="diagonal").x.shape == (7, 4)


def test_conditional_innovation_finds_injected_target_and_uses_neighbors():
    reference = [linear_graph(i) for i in range(30)]
    model = RoutingResidual().fit(reference)
    no_neighbors = RoutingResidual(mode="no_neighbors").fit(reference)
    test = linear_graph(100)
    clean = model.score(test)["score"]
    assert clean.mean() < no_neighbors.score(test)["score"].mean() * 0.1
    changed = copy.deepcopy(test)
    changed.x[7] += 6
    scores = model.score(changed)["score"]
    assert np.argmax(scores) == 7 - test.start
    assert scores[4] > clean[4] + 100
    # The predictor for i never receives target i's own state.
    np.testing.assert_allclose(model._design(test)[1][4], model._design(changed)[1][4])


@pytest.mark.parametrize("mode", ["graph", "no_neighbors", "permuted_weights", "mass_matched_uniform"])
def test_causal_prefix_invariance(mode):
    model = RoutingResidual(mode=mode).fit([linear_graph(i) for i in range(8)])
    whole = linear_graph(91)
    prefix = copy.copy(whole)
    prefix.x = whole.x[:10]
    prefix.adjacency = whole.adjacency[:10, :10]
    prefix.diagonal = whole.diagonal[:10]
    for name, result in model.score(prefix).items():
        np.testing.assert_allclose(result, model.score(whole)[name][:7], atol=1e-10)


def test_source_answer_token_weighting():
    graphs = [linear_graph(1, source="a"), linear_graph(2, n=20, source="a"), linear_graph(3, source="b")]
    weights = source_weights(graphs)
    assert np.isclose(weights[:30].sum(), 0.5)
    assert np.isclose(weights[30:].sum(), 0.5)


def test_permutation_preserves_prompt_history_mass():
    g = build_flow_graph(record(), attribute="diagonal")
    # Make heterogeneous weights so the permutation control is observable.
    g.adjacency.data *= np.linspace(0.1, 1, len(g.adjacency.data))
    model = RoutingResidual(mode="permuted_weights")
    a = model._adjacency(g)
    np.testing.assert_array_equal(a.indices, g.adjacency.indices)
    for segment in (slice(None, g.start), slice(g.start, None)):
        np.testing.assert_allclose(a[:, segment].sum(axis=1), g.adjacency[:, segment].sum(axis=1))
    assert not np.array_equal(a.data, g.adjacency.data)


def test_calibration_small_sample_ties_and_source_blocks():
    cal = SourceCalibration([np.array([1., 3.]), np.array([2.]), np.array([4.])], ["a", "a", "b"])
    assert cal.summary()["threshold_is_infinite"]
    np.testing.assert_allclose(cal.p_values([3., 4., 5.]), [1., 2 / 3, 1 / 3])
    enough = SourceCalibration([np.array([float(i)]) for i in range(19)], list(map(str, range(19))))
    assert enough.summary()["threshold"] == 18
    assert enough.p_values([18.])[0] > 0.05
    assert enough.p_values([19.])[0] == 0.05


def test_model_roundtrip(tmp_path):
    model = RoutingResidual().fit([linear_graph(i) for i in range(8)])
    path = tmp_path / "model.npz"
    model.save(path)
    np.testing.assert_allclose(model.score(linear_graph(60))["score"],
                               RoutingResidual.load(path).score(linear_graph(60))["score"])


def make_roster(tmp_path):
    rows = []
    for i, role in enumerate(["reference", "reference", "calibration", "test"]):
        r = record()
        np.savez(tmp_path / f"{i}.npz", attention=r.attention, response_idx=3,
                 hidden=linear_graph(i, n=7).x, hidden_schema=json.dumps({'observer_model': 'synthetic', 'observer_revision': 'v1', 'tokenizer': 'integer', 'tokenizer_revision': 'v1', 'layer': '0', 'state_location': 'synthetic-node'}), token_ids=np.arange(7), labels=np.array([{}], dtype=object))
        rows.append(dict(path=f"{i}.npz", source_id=str(i), response_id=str(i), split=role))
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(rows))
    return path


def test_source_split_leakage_rejected(tmp_path):
    path = make_roster(tmp_path)
    rows = json.loads(path.read_text())
    rows[-1]["source_id"] = rows[0]["source_id"]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="leakage"):
        read_roster(path)


def test_frozen_end_to_end_and_tamper_rejection(tmp_path):
    roster = make_roster(tmp_path)
    out = tmp_path / "run"
    score_roster(roster, out, attribute="hidden")
    verify_freeze(out)
    annotation = tmp_path / "labels.jsonl"
    annotation.write_text(json.dumps(dict(id="3", token_ids=[3, 4, 5, 6], offsets=[[0, 1], [2, 3], [4, 5], [6, 7]],
                                          labels=[dict(start=2, end=3)])) + "\n")
    report = evaluate_frozen(out, annotation)
    assert report["methods"]["graph"]["all"]["pooled"]["tokens"] == 4
    assert report["methods"]["graph"]["through_first_error"]["pooled"]["tokens"] == 2
    assert report["methods"]["graph"]["strict_post_first"]["pooled"]["tokens"] == 2
    with pytest.raises(FileExistsError):
        score_roster(roster, out)
    (out / "scores.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_freeze(out)


def test_tied_metrics_are_permutation_invariant():
    for y in ([0, 1, 0, 1], [1, 1, 0, 0]):
        result = _metrics(np.array(y), np.ones(4))
        assert result["auroc"] == 0.5
        assert result["ap"] == 0.5


def test_hidden_is_required_by_default():
    with pytest.raises(ValueError, match="hidden attributes requested but absent"):
        build_flow_graph(record())


def test_uniform_control_is_same_dimension_and_different_message():
    refs = [linear_graph(i) for i in range(10)]
    actual = RoutingResidual().fit(refs)
    uniform = RoutingResidual(mode="mass_matched_uniform").fit(refs)
    g = linear_graph(90)
    assert actual.beta.shape == uniform.beta.shape
    np.testing.assert_allclose(actual._design(g)[1][:, :5], uniform._design(g)[1][:, :5])
    assert not np.allclose(actual._design(g)[1][:, 5:], uniform._design(g)[1][:, 5:])


def test_mismatched_observer_tokenizer_is_rejected(tmp_path):
    out = tmp_path / "run"
    score_roster(make_roster(tmp_path), out, attribute="hidden")
    annotation = tmp_path / "bad_labels.jsonl"
    annotation.write_text(json.dumps(dict(id="3", token_ids=[103, 104, 105, 106],
        offsets=[[0, 1], [2, 3], [4, 5], [6, 7]], labels=[])) + "\n")
    with pytest.raises(ValueError, match="observer response token_ids"):
        evaluate_frozen(out, annotation)


def test_large_sparse_prompt_does_not_require_dense_adjacency():
    n = 100000
    r = ResponseRecord("large", "s", None, n - 1, sparse=dict(
        attention_diagonal=np.full((1, 1, n), 0.2), response_row_ptr=np.array([0, 1]),
        response_column_indices=np.array([0]), response_values=np.array([0.8])))
    graph = build_flow_graph(r, attribute="diagonal")
    assert sparse.isspmatrix_csr(graph.adjacency)
    assert graph.adjacency.nnz == 1
    assert graph.adjacency.data.nbytes + graph.adjacency.indptr.nbytes < 1000000


def test_zero_length_offsets_report_unavailable_instead_of_normal(tmp_path):
    out = tmp_path / "run"
    score_roster(make_roster(tmp_path), out, attribute="hidden")
    annotation = tmp_path / "partial_labels.jsonl"
    annotation.write_text(json.dumps(dict(id="3", token_ids=[3, 4, 5, 6],
        offsets=[[0, 1], [2, 3], [4, 5], [0, 0]], labels=[dict(start=2, end=3)])) + "\n")
    report = evaluate_frozen(out, annotation)
    assert report["coverage"] == 0.75
    assert report["unavailable_tokens"] == {"3": 1}
    assert report["methods"]["graph"]["all"]["pooled"]["tokens"] == 3


def test_same_dimension_different_hidden_schema_is_rejected():
    refs = [linear_graph(i) for i in range(4)]
    refs[1].schema = "different-model-layer-tokenizer"
    with pytest.raises(ValueError, match="mixed node attribute schemas"):
        RoutingResidual().fit(refs)


def test_missing_hidden_provenance_rejected():
    r = record()
    r.hidden = np.ones((7, 4))
    with pytest.raises(ValueError, match="hidden_schema"):
        build_flow_graph(r)


def test_paired_bootstrap_resamples_the_same_sources():
    result = paired_source_difference({"a": 0.1, "b": 0.3, "c": 0.6},
                                      {"a": 0.3, "b": 0.5, "c": 0.8})
    assert result["eligible_sources"] == 3
    np.testing.assert_allclose(result["bootstrap95"], [-0.2, -0.2])
    assert np.isclose(result["graph_minus_uniform"], -0.2)


def test_small_rounding_overflow_does_not_create_negative_entropy():
    r = record()
    r.attention[:] = 0
    for q in range(7):
        r.attention[:, :, q, q] = 0.5
        if q:
            r.attention[:, :, q, q - 1] = 0.501
    graph = build_flow_graph(r, attribute="diagonal")
    np.testing.assert_allclose(graph.routing().sum(axis=1), 1)
    assert np.all(graph.routing_entropy() >= 0)
