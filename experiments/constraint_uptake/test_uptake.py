import numpy as np
from .score import key_effects,head_features


def test_self_key_source_assignment_and_no_negative_conflict():
    native = np.zeros((1,2,1,5))
    native[0,0,0,-1] = -2
    native[0,1,0,0] = -1
    native[0,1,0,-1] = 1
    attention = np.zeros((1,2,5))
    attention[:,:,-1] = 1
    feature,_ = head_features(native,native,attention,3,np.array([0,1]))
    np.testing.assert_allclose(feature[0,:,0,0],[0,1])
    np.testing.assert_allclose(feature[0,:,0,4],[1,0])


def test_batched_factor_product_equals_explicit_tensor_contraction():
    rng = np.random.default_rng(18)
    gradient = rng.normal(size=(4,5,3,8))
    values = rng.normal(size=(2,7,8))
    own = rng.normal(size=(4,5,8))
    attention = rng.uniform(size=(4,5,8))
    expected = np.einsum('hqcd,hkd->hqck',gradient,np.repeat(values,2,axis=0))*attention[:,:,None,:-1]
    actual = key_effects(gradient,values,own,attention)
    np.testing.assert_allclose(actual[...,:-1],expected,atol=1e-12)
