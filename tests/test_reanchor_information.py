import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

from experiments.unsupervised_token_graph.channels import ChannelGraph, iter_channels
from experiments.unsupervised_token_graph.data import ResponseCache, ResponseRecord
from experiments.unsupervised_token_graph.evaluate import Ranking, label_views
from experiments.unsupervised_token_graph.information import SourceFlow, entropy_bits, future_influence, js_bits
from experiments.unsupervised_token_graph.reanchor import InformationDiagnostics, ReanchorAnalyzer
from experiments.unsupervised_token_graph.reanchor_run import analyze_record


def graph(rows, p=2, queries=None):
    rows = np.asarray(rows, float)
    queries = p + np.arange(len(rows)) if queries is None else np.asarray(queries)
    return ChannelGraph(0, 0, queries, sparse.csr_matrix(rows), p)


def canonical(path, attention, p):
    layers, heads, tokens, _ = attention.shape
    ptr, columns, values = [0], [], []
    for l in range(layers):
        for h in range(heads):
            for q in range(p, tokens):
                keys = np.flatnonzero(attention[l, h, q, :q])
                columns.extend(keys); values.extend(attention[l, h, q, keys])
                ptr.append(len(values))
    np.savez_compressed(path, token_ids=np.arange(tokens), response_idx=p,
                        attention_diagonal=np.diagonal(attention, axis1=2, axis2=3),
                        response_row_ptr=np.asarray(ptr), response_column_indices=np.asarray(columns, int),
                        response_values=np.asarray(values), labels=np.array([{"must_not_load": 1}], dtype=object))


def test_sparse_and_dense_channels_have_identical_measurements(tmp_path):
    rng = np.random.default_rng(4)
    attention = np.tril(rng.random((2, 2, 8, 8)))
    attention /= attention.sum(-1, keepdims=True)
    # A common sparse retained-edge view, not a comparison against lost values.
    off_diagonal = ~np.eye(8, dtype=bool)
    attention[(attention < .08) & off_diagonal] = 0
    canonical(tmp_path / "sparse.npz", attention, 3)
    np.savez_compressed(tmp_path / "dense.npz", attention=attention[:, :, 3:],
                        query_positions=np.arange(3, 8), prompt_length=3, token_ids=np.arange(8))
    a, b = (ResponseCache().load(tmp_path / name) for name in ("sparse.npz", "dense.npz"))
    for ca, cb in zip(iter_channels(a), iter_channels(b)):
        np.testing.assert_allclose(ca.attention.toarray(), cb.attention.toarray())
        np.testing.assert_array_equal(ca.prediction_positions, np.arange(4, 9))
    aa, bb = analyze_record(a, root_bins=2), analyze_record(b, root_bins=2)
    for key in aa:
        if key != "cache_format":
            np.testing.assert_allclose(aa[key], bb[key], equal_nan=True, atol=1e-12)


def test_query_alignment_keeps_canonical_self_attention(tmp_path):
    attention = np.zeros((1, 1, 5, 5))
    for q in range(2, 5):
        attention[0, 0, q, 0], attention[0, 0, q, q] = .2, .8
    canonical(tmp_path / "a.npz", attention, 2)
    channel = next(iter_channels(ResponseCache().load(tmp_path / "a.npz")))
    result = ReanchorAnalyzer().run(channel)
    np.testing.assert_allclose(result.local_mass, .8)
    np.testing.assert_allclose(result.reanchor_ratio, .2)
    np.testing.assert_array_equal(channel.queries, [2, 3, 4])
    np.testing.assert_array_equal(channel.prediction_positions, [3, 4, 5])


def test_true_multi_hop_ancestry_changes_source_mismatch():
    rows = [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [.5, 0, 0, .5, 0]]
    connected = SourceFlow().run(graph(rows), keep_roots=True)
    rows[0] = [0, 1, 0, 0, 0]
    switched = SourceFlow().run(graph(rows))
    assert connected["source_mismatch_bits"][-1] == pytest.approx(0)
    assert switched["source_mismatch_bits"][-1] == pytest.approx(1)
    assert connected["prompt_path_length"][-1] == pytest.approx(2)
    assert connected["prompt_reach"][-1] == pytest.approx(1)
    no_ancestry = SourceFlow().run(graph(rows), control="no_ancestry")
    assert no_ancestry["prompt_reach"][-1] == pytest.approx(.5)
    assert np.isnan(no_ancestry["source_mismatch_bits"][-1])


def test_entropy_chain_rule_and_missing_mass():
    rows = [[.3, .1, .2, 0, 0], [.1, 0, .5, .1, 0], [0, .2, .2, .3, .1]]
    values = SourceFlow().run(graph(rows), keep_roots=True)
    np.testing.assert_allclose(values["root_distribution"].sum(-1), 1)
    np.testing.assert_allclose(values["prompt_reach"] + values["self_terminal_mass"] + values["unknown_mass"], 1)
    np.testing.assert_allclose(values["terminal_entropy_bits"],
                               values["carrier_conditional_entropy_bits"] + values["carrier_information_bits"])
    assert values["unknown_mass"][0] == pytest.approx(.4)
    assert values["prompt_reach"][0] == pytest.approx(.4)
    assert np.all(values["carrier_information_bits"] >= -1e-12)


def test_unobserved_carrier_is_unknown_not_a_fabricated_zero_root():
    rows = [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]]
    values = SourceFlow().run(graph(rows, queries=[2, 4]))
    assert values["unknown_mass"][1] == pytest.approx(1)
    assert np.isnan(values["prompt_root_entropy_bits"][1])


def test_prompt_coarsening_does_not_increase_js():
    rows = [[0, 1, 0, 0, 0, 0], [.5, 0, 0, 0, .5, 0]]
    exact = SourceFlow().run(graph(rows, p=4))
    coarse = SourceFlow(root_bins=2).run(graph(rows, p=4))
    assert exact["source_mismatch_bits"][-1] == pytest.approx(1)
    assert coarse["source_mismatch_bits"][-1] == pytest.approx(0)


def test_prefix_invariance_for_events_and_information():
    rng = np.random.default_rng(8)
    rows = np.tril(rng.random((18, 18)))
    rows /= rows.sum(-1, keepdims=True)
    full = graph(rows[2:], p=2)
    prefix = graph(rows[2:10, :10], p=2)
    analyzer = ReanchorAnalyzer(min_history=2)
    a, b = analyzer.run(full), analyzer.run(prefix)
    for name in a.__dataclass_fields__:
        np.testing.assert_allclose(getattr(a, name)[:8], getattr(b, name), equal_nan=True)
    for control in ("real", "history_permuted", "no_ancestry"):
        a, b = SourceFlow().run(full, control), SourceFlow().run(prefix, control)
        for name in a:
            np.testing.assert_allclose(a[name][:8], b[name], equal_nan=True)


def test_window_aging_alone_never_becomes_an_event():
    rows = np.zeros((15, 17)); rows[:, 0] = .2; rows[:, 2] = .8
    result = ReanchorAnalyzer(window=3, min_history=1).run(graph(rows))
    assert np.any(np.diff(result.reanchor_ratio) > .5)
    assert np.all(result.event_strength == 0)
    assert not result.event.any()


def test_head_swap_is_visible_without_averaging_and_input_unchanged():
    attention = np.zeros((1, 2, 2, 5))
    attention[0, 0, 0, 0] = attention[0, 0, 1, 1] = 1
    attention[0, 1, 0, 1] = attention[0, 1, 1, 0] = 1
    before = attention.copy()
    record = ResponseRecord("x", "s", attention, 2, 2, query_positions=np.array([2, 3]))
    values = analyze_record(record)
    np.testing.assert_array_equal(values["prompt_entropy"], 0)
    np.testing.assert_allclose(values["distribution_shift"][:, 1], 1)
    np.testing.assert_array_equal(attention, before)


def test_empty_prompt_conditional_is_nan():
    rows = [[1, 0, 0, 0], [0, 0, .5, .5]]
    result = ReanchorAnalyzer().run(graph(rows))
    assert np.isnan(result.prompt_entropy[1])
    assert np.isnan(result.distribution_shift[1])
    assert np.isnan(js_bits([0, 0], [1, 0]))


def test_fai_is_offline_and_tail_is_unobserved():
    rows = np.zeros((8, 10)); rows[:, 0] = 1
    a = graph(rows)
    rows[3:, 0], rows[3:, 3] = .4, .6
    b = graph(rows)
    x, _ = future_influence(a, 2, 5)
    y, count = future_influence(b, 2, 5)
    assert y[0] > x[0]
    assert count[-1] == 0 and np.isnan(y[-1])
    np.testing.assert_allclose(SourceFlow().run(a)["prompt_reach"][:3], SourceFlow().run(b)["prompt_reach"][:3])


def test_bad_future_attention_is_rejected():
    record = ResponseRecord("x", "s", np.ones((1, 1, 1, 4)) / 4, 2, 2,
                            query_positions=np.array([2]))
    with pytest.raises(ValueError, match="causal"):
        list(iter_channels(record))


def test_small_rounding_overflow_only():
    record = ResponseRecord("x", "s", np.array([[[[.5001, .5001, 0]]]]), 2, 2,
                            query_positions=np.array([2]))
    channel = next(iter_channels(record))
    assert float(channel.attention.sum()) == pytest.approx(1)
    record.attention *= .5
    channel = next(iter_channels(record))
    assert float(channel.attention.sum()) == pytest.approx(.5001)


def test_binary_distribution_does_not_collapse_to_one_histogram_bin():
    value = InformationDiagnostics.histogram_kl(np.zeros(94), np.ones(6))
    assert value["js"] == pytest.approx(1)
    assert value["overlap"] == pytest.approx(0)
    assert np.isinf(value["kl"]) and value["bins"] == 2


def test_logloss_refuses_broadcast_and_nonprobabilities():
    assert InformationDiagnostics.empirical_logloss([.1, .9], [0, 1]) == pytest.approx(-np.log(.9))
    with pytest.raises(ValueError):
        InformationDiagnostics.empirical_logloss([[.1], [.9]], [0, 1])
    with pytest.raises(ValueError):
        InformationDiagnostics.empirical_logloss([2, 3], [0, 1])


@pytest.mark.parametrize("labels", [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]])
def test_tied_scores_are_order_invariant(labels):
    measured = Ranking(labels, [0, 0, 0, 0]).measure()
    assert measured["auroc"] == pytest.approx(.5)
    assert measured["ap"] == pytest.approx(.5)


def test_weighted_ranking_matches_sklearn():
    rng = np.random.default_rng(3)
    labels, scores, weights = rng.integers(2, size=100), rng.integers(4, size=100), rng.random(100)
    measured = Ranking(labels, scores).measure(weights)
    assert measured["auroc"] == pytest.approx(roc_auc_score(labels, scores, sample_weight=weights))
    assert measured["ap"] == pytest.approx(average_precision_score(labels, scores, sample_weight=weights))


def test_zero_length_token_does_not_acquire_a_label():
    views = label_views([[0, 0], [0, 2]], [{"start": 0, "end": 1}])
    np.testing.assert_array_equal(views["all_error"][0], [False, True])


def test_entropy_units():
    assert entropy_bits([.5, .5]) == pytest.approx(1)


def test_weighted_js_uses_branch_mass_not_equal_weights():
    rows = [[0, 1, 0, 0], [.1, 0, .9, 0]]
    result = SourceFlow().run(graph(rows))
    assert result["source_mismatch_bits"][-1] == pytest.approx(entropy_bits([.1, .9]))


def test_rewiring_is_a_real_topology_control():
    rows = np.zeros((5, 7))
    rows[0, 0] = 1
    rows[1, 1] = 1
    rows[2, 0] = rows[3, 0] = 1
    rows[4, 0], rows[4, 2], rows[4, 3] = .2, .7, .1
    channel = graph(rows)
    real = SourceFlow().run(channel)
    changed = [SourceFlow(seed=seed).run(channel, control="history_permuted") for seed in range(10)]
    assert any(not np.isclose(x["source_mismatch_bits"][-1], real["source_mismatch_bits"][-1]) for x in changed)
    for result in changed:
        np.testing.assert_allclose(result["direct_prompt_mass"], real["direct_prompt_mass"])
        np.testing.assert_allclose(result["unknown_mass"], real["unknown_mass"])
