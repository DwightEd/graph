import numpy as np
from .model import solve_correction,overlap_parts,combine_overlap,score_observations
from .edges import scatter_self


def test_zero_innovation_preserves_baseline():
    base = np.array([.2,.4,.9,.1])
    scores,_ = score_observations(base,base,np.ones(3),np.ones(3))
    for value in scores.values():
        np.testing.assert_array_equal(value,base)


def test_zero_edges_do_not_propagate_and_corrections_are_bounded():
    raw = np.array([-.4,.07,.8,-.01])
    answer,gap = solve_correction(raw,np.zeros(3))
    np.testing.assert_allclose(answer,np.clip(raw,-.1,.1))
    assert gap==0


def test_two_point_solution_and_dual_certificate():
    answer,gap = solve_correction(np.array([-.08,.08]),np.ones(1))
    np.testing.assert_allclose(answer,[-.03,.03],atol=1e-8)
    assert gap<1e-8


def test_native_identity_overlap_and_opposite_sign_separation():
    attention = np.array([[[.4,.6],[.4,.6]],[[.1,.9],[.1,.9]]])
    effect = np.array([[[1.,-2.],[1.,-2.]],[[.5,3.],[.5,3.]]])
    energy,parts = overlap_parts(attention,effect)
    np.testing.assert_allclose(combine_overlap(energy,parts),1.)
    effect[:,1] *= -1
    energy,parts = overlap_parts(attention,effect)
    np.testing.assert_array_equal(combine_overlap(energy,parts),[0.])


def test_zero_message_energy_has_no_artificial_edge():
    attention = np.ones((2,3,4))/4
    energy,parts = overlap_parts(attention,np.zeros_like(attention))
    np.testing.assert_array_equal(combine_overlap(energy,parts),[0.,0.])


def test_self_key_scatter_uses_previous_token_absolute_address():
    value = np.zeros((1,3,6))
    value[0,:,-1] = [1,2,3]
    expected = np.zeros((1,3,5))
    expected[0,[0,1,2],[2,3,4]] = [1,2,3]
    np.testing.assert_array_equal(scatter_self(value,3),expected)


def test_large_innovation_does_not_exceed_bound_under_tv():
    rng = np.random.default_rng(42)
    answer,gap = solve_correction(rng.normal(size=30),rng.uniform(size=29))
    assert max(abs(answer))<=.1
    assert gap<1e-6
