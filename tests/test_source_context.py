import numpy as np

from route_graph.context import source_context_readout


def example():
    a = np.broadcast_to(np.eye(8), (2, 1, 8, 8)).copy()
    for receiver, source in [(2, 0), (4, 3), (6, 0)]:
        a[0, 0, receiver] = 0
        a[0, 0, receiver, source] = 1
    a[1, 0, 7] = 0
    a[1, 0, 7, [2, 4, 6]] = [0.3, 0.3, 0.4]
    return (
        a,
        np.array([10, 11, 20, 12, 21, 13, 14, 15]),
        np.array([0, 0, 0, 0, 0, 1, 2, 2]),
    )


def test_same_direct_mass_can_have_different_contextual_ownership():
    a, ids, roles = example()
    result = source_context_readout(a, ids, roles, 7, [20, 21])
    assert result["valid"]
    assert result["context_support"] == [1, 0]
    assert result["direct_support"] == [0.5, 0.5]
    assert (
        result["endpoint_permutation"]
        == np.random.default_rng(20260912).permutation(2).tolist()
    )
    # Swap the source attribution of the history relay without changing q's row.
    a[0, 0, 6] = 0
    a[0, 0, 6, 3] = 1
    swapped = source_context_readout(a, ids, roles, 7, [20, 21])
    assert swapped["context_support"] == [0, 1]
    assert swapped["direct_support"] == result["direct_support"]


def test_candidate_permutation_and_future_suffix_do_not_change_support():
    a, ids, roles = example()
    first = source_context_readout(a, ids, roles, 7, [20, 21])
    permuted = source_context_readout(a, ids, roles, 7, [21, 20])
    assert permuted["context_support"] == first["context_support"][::-1]
    extended = np.zeros((2, 1, 9, 9))
    extended[:, :, :8, :8] = a
    extended[:, :, 8, 8] = 1
    result = source_context_readout(
        extended, np.r_[ids, 999], np.r_[roles, 2], 7, [20, 21]
    )
    assert result == first


def test_missing_source_candidates_or_history_are_explicitly_unscorable():
    a, ids, roles = example()
    for candidates in ([20, 99], [98, 99]):
        result = source_context_readout(a, ids, roles, 7, candidates)
        assert not result["valid"]
        assert result["context_risk"] is None
    roles[6] = 3
    result = source_context_readout(a, ids, roles, 7, [20, 21])
    assert not result["valid"]
    assert result["context_risk"] is None

    assert result["direct_valid"]
    assert result["direct_risk"] == 0.5
    assert result["count_valid"]


def test_unknown_candidate_context_cannot_become_negative_evidence():
    a, ids, roles = example()
    a[0, 0, 4] = 0
    a[0, 0, 4, 4] = 1
    result = source_context_readout(a, ids, roles, 7, [20, 21])
    assert result["candidate_context_coverage"] == [1, 0]
    assert not result["context_valid"]
    assert result["context_risk"] is None
    assert result["direct_valid"]


def test_random_control_is_reproducible_with_multiple_candidate_occurrences():
    a, ids, roles = example()
    ids[1] = 20
    a[0, 0, 1] = 0
    a[0, 0, 1, 0] = 1
    first = source_context_readout(a, ids, roles, 7, [20, 21], seed=11)
    repeated = source_context_readout(a, ids, roles, 7, [20, 21], seed=11)
    assert first == repeated
    assert sorted(first["endpoint_permutation"]) == [0, 1, 2]
