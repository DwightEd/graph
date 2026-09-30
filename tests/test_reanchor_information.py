import numpy as np
import pytest
from scipy import sparse

from experiments.unsupervised_token_graph.channels import ChannelGraph
from experiments.unsupervised_token_graph.information import SourceFlow, entropy_bits, future_influence


def graph(rows, p=2, queries=None):
    rows = np.asarray(rows, float)
    queries = p + np.arange(len(rows)) if queries is None else np.asarray(queries)
    return ChannelGraph(0, 0, queries, sparse.csr_matrix(rows), p)


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
