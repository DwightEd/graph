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


def test_energy_rank_centering_scaling_and_known_spectrum():
    from .readout import energy_rank
    matrix = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    result = energy_rank(matrix)
    np.testing.assert_allclose(result['rank'], 2.)
    np.testing.assert_allclose(result['normalized'], 1.)
    np.testing.assert_allclose(energy_rank(matrix*3+20)['rank'], result['rank'])
    assert energy_rank(np.ones((3,2)))['rank']==0
    assert energy_rank(np.ones((1,2)))['rank'] is None


def test_attention_geometry_uses_prediction_query_and_visible_normalization():
    from .readout import attention_geometry
    attention = np.zeros((2,5,5))
    for query in range(5):
        attention[:,query,:query+1] = 1/(query+1)
    measured = attention_geometry(attention, 2, 4)
    np.testing.assert_allclose(measured['attention_entropy'], [1.,1.])
    np.testing.assert_allclose(measured['prior_mass'], [1.,2/3])
    np.testing.assert_allclose(measured['head_disagreement'], 0., atol=1e-14)


def test_first_error_geometry_pairs_exclude_unknown_and_match_same_answer():
    from .evaluate import adjacent_geometry_pairs
    rows = [dict(id='x',problem='p',role='evaluation',step=i,label=y,length=5,value=v)
            for i,y,v in [(0,0,1.),(1,1,3.),(2,-1,100.)]]
    pairs = adjacent_geometry_pairs(rows,'value',1)
    assert len(pairs)==1
    assert pairs[0]['difference']==2.


def test_equal_count_prefix_does_not_mix_later_state_changes(tmp_path):
    from .evaluate import equal_length_hidden_pairs
    base = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    states = np.vstack((base,base,base*5,np.zeros((1,2))))
    path = tmp_path/'hidden.npz'
    np.savez(path, activation=states, step_ranges=np.array([[1,5],[5,13]]))
    rows = [dict(id='x',problem='p',role='evaluation',step=step,label=step,
                 hidden_cache=str(path)) for step in (0,1)]
    pairs = equal_length_hidden_pairs(rows)
    prefix = next(row for row in pairs if row['method']=='state_prefix')
    assert prefix['matched_count']==4
    np.testing.assert_allclose(prefix['difference'],0.,atol=1e-12)
