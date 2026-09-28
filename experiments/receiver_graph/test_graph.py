import numpy as np
from .measure import conditional_mean, history_joint
from .score import restart_weights, pool


def test_actual_token_key_alignment_and_future_mask():
    attention = np.zeros((1, 4, 6))
    attention[0, 1, 3] = .7  # prompt=3: y0 input is readable when predicting y1.
    attention[0, 2, 3] = .4
    derivative = attention*2
    reading, joint = history_joint(attention, derivative, 3)
    assert reading[0,1,0]==.7
    assert reading[0,2,0]==.4
    assert np.triu(joint[0]).sum()==0


def test_matched_null_preserves_groups_and_singleton_edges():
    values = np.zeros((2, 8, 8))
    values[:,7,:7] = np.arange(1,8)
    ids = np.arange(8)
    matched = conditional_mean(values, ids)
    np.testing.assert_allclose(matched[:,7,5:7], values[:,7,5:7])
    np.testing.assert_allclose(matched[:,7,3:5], [[4.5,4.5],[4.5,4.5]])
    np.testing.assert_allclose(matched.sum(-1), values.sum(-1))


def test_isolated_node_unchanged_and_correlated_copies_do_not_multiply_evidence():
    z = np.ones(5)
    weights = restart_weights(np.zeros((5,5)))
    np.testing.assert_allclose(pool(z, weights, .9), z)
    weights = np.full((5,5), .2)
    np.testing.assert_allclose(pool(z, weights, 0), np.sqrt(5))
    assert np.max(pool(z, weights, .99))<1.02


def test_tail_preserves_out_of_reference_order_and_missingness():
    from .balance import smooth_tail
    reference = np.array([-1., 0., 1.])
    values = smooth_tail(np.array([2., 3., 1e4, np.nan]), reference, np.ones(3))
    assert np.isfinite(values[:3]).all()
    assert np.all(np.diff(values[:3])>0)
    assert np.isnan(values[3])
