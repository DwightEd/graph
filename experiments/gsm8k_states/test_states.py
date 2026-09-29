import numpy as np
from .readout import fit_lda,apply_lda,average_steps


def test_dual_lda_matches_primal_full_coordinate_solution():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(12,7))
    y = np.array([0,1]*6)
    weight = np.arange(1,13,dtype=float)
    model = fit_lda(x,y,weight)
    weight /= weight.sum()
    z = (x-model['center'])/model['scale']
    means = np.stack([np.average(z[y==k],axis=0,weights=weight[y==k]) for k in (0,1)])
    r = z-means[y]
    covariance = .5*np.eye(7)+.5*(r.T*weight)@r
    expected = np.linalg.solve(covariance,means[1]-means[0])
    np.testing.assert_allclose(model['direction'],expected,atol=1e-10)
    assert np.isfinite(apply_lda(x,model)).all()


def test_step_reduction_keeps_layers_and_different_lengths():
    values = np.array([[1.,3.,9.],[2.,4.,8.]])
    result = average_steps(values,np.array([[0,2],[2,3]]))
    np.testing.assert_allclose(result,[[2,3],[9,8]])


def test_step_states_entry_is_before_step_end_is_last_observed_token():
    import torch
    from .capture import step_states
    state = torch.arange(24).reshape(8,3).float()
    pooled = step_states([state],np.array([[0,2],[2,5]]),3)
    np.testing.assert_array_equal(pooled['entry'][0],state[[2,4]].numpy())
    np.testing.assert_array_equal(pooled['end'][0],state[[4,7]].numpy())
    np.testing.assert_allclose(pooled['mean'][0],torch.stack((state[3:5].mean(0),state[5:8].mean(0))).numpy())


def test_top_fraction_never_crosses_step_boundaries():
    from .aggregation import top_fraction
    values = np.array([100.,1.,2.,3.,4.,-1.,-2.])
    pooled = top_fraction(values,np.array([[0,5],[5,7]]))
    np.testing.assert_array_equal(pooled,[100.,-1.])
