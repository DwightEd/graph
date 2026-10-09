"""Scientific checks of the coupling object; all inputs are synthetic."""
import unittest

import numpy as np
from scipy.stats import multivariate_normal, norm

from .source_coupling import (answer_coordinates, coupling_rank, dependence_ratio,
    fit_normal_model, fit_risk_reference, normal_score)
from .unlabeled import (equal_source_weights, fit_weighted_cdf, graph_fields,
                        local_weights, reference_rank)


def analytic_model(independent=False):
    A=np.array([[2.,.3],[.3,1.5]])
    b=np.array([.4,-.2]) if not independent else np.zeros(2)
    d=1.1
    beta=np.linalg.solve(A,b)
    v=d-b@beta
    return dict(mean=np.array([.1,-.2,.3,.4,-.5]),A=A,b=b,d=d,beta=beta,v=v,
                conditioner=np.array([[.2,0],[0,.3],[.1,.1]]))


def synthetic_fit():
    source=np.array([0,0,0,1,1,1,2,2,2,2])
    data=np.array([[-2.,-1.,-.8,-2.,-1.],[-1.,-.4,-.3,-1.,-.8],
        [0.,.2,.1,-.2,-.5],[.7,.1,.3,-.4,-.2],[1.2,.8,.7,-.7,-.9],
        [-.3,-.7,-.2,-1.2,-1.1],[.1,-.8,.1,-.1,-.4],[1.,.9,.5,-.9,-.7],
        [1.4,1.1,.9,-.6,-.3],[-.7,.5,-.6,-1.5,-1.8]])
    names=('source_local','source_full','raw_route')
    reference={}
    weights=equal_source_weights(source)
    for name,values in zip(names,data[:,:3].T):
        distinct,cumulative=fit_weighted_cdf(values,weights)
        reference[name+'_values']=distinct
        reference[name+'_cumulative']=cumulative
    ranks={name:reference_rank(reference,name,data[:,index]) for index,name in enumerate(names)}
    ranks.update(local_native_logp=data[:,3],full_native_logp=data[:,4])
    return source,data,ranks,reference


class ConditionalCouplingTests(unittest.TestCase):
    def test_independent_blocks_exact_zero(self):
        z=np.random.default_rng(42).normal(size=(20,5))
        D,_=dependence_ratio(z,analytic_model(independent=True))
        np.testing.assert_array_equal(D,np.zeros(20))

    def test_ratio_matches_independent_generic_logpdf(self):
        model=analytic_model()
        z=np.random.default_rng(43).normal(size=(19,5))
        D,details=dependence_ratio(z,model)
        residual=np.column_stack([details['source_local_residual'],details['source_full_residual'],details['route_residual']])
        covariance=np.block([[model['A'],model['b'][:,None]],[model['b'][None,:],np.array([[model['d']]])]])
        expected=(multivariate_normal.logpdf(residual[:,:2],mean=np.zeros(2),cov=model['A'])
            +norm.logpdf(residual[:,2],scale=np.sqrt(model['d']))
            -multivariate_normal.logpdf(residual,mean=np.zeros(3),cov=covariance))
        np.testing.assert_allclose(D,expected,atol=2e-14,rtol=0)

    def test_residual_sign_symmetry(self):
        model=analytic_model()
        z=np.random.default_rng(1).normal(size=(13,5))
        baseline,_=dependence_ratio(z,model)
        native_prediction=(z[:,3:]-model['mean'][3:])@model['conditioner'].T+model['mean'][:3]
        flipped=z.copy()
        flipped[:,:3]=2*native_prediction-z[:,:3]
        alternate,_=dependence_ratio(flipped,model)
        np.testing.assert_allclose(baseline,alternate,atol=2e-14,rtol=0)

    def test_source_pair_covariance_permutation_equivariance(self):
        model=analytic_model()
        z=np.random.default_rng(2).normal(size=(13,5))
        baseline,_=dependence_ratio(z,model)
        swapped=dict(model)
        swapped['mean']=model['mean'][[1,0,2,3,4]]
        swapped['conditioner']=model['conditioner'][[1,0,2]]
        swapped['A']=model['A'][[1,0]][:,[1,0]]
        swapped['b']=model['b'][[1,0]]
        swapped['beta']=np.linalg.solve(swapped['A'],swapped['b'])
        alternate,_=dependence_ratio(z[:,[1,0,2,3,4]],swapped)
        np.testing.assert_allclose(baseline,alternate,atol=2e-14,rtol=0)

    def test_native_condition_shift_cancels(self):
        model=analytic_model()
        z=np.random.default_rng(3).normal(size=(13,5))
        baseline,_=dependence_ratio(z,model)
        shift=np.array([.7,-.3])
        shifted=z.copy()
        shifted[:,3:]+=shift
        shifted[:,:3]+=model['conditioner']@shift
        alternate,_=dependence_ratio(shifted,model)
        np.testing.assert_allclose(baseline,alternate,atol=2e-14,rtol=0)

    def test_support_endpoints_and_ties_without_epsilon(self):
        reference=dict(test_values=np.array([0.,1.]),test_cumulative=np.array([0.,.5,1.]))
        values=normal_score(np.array([0.,.25,.25,.5,.75,1.]),reference,'test')
        np.testing.assert_array_equal(values[[0,1,2]],np.repeat(norm.ppf(.25),3))
        np.testing.assert_array_equal(values[[4,5]],np.repeat(norm.ppf(.75),2))
        self.assertEqual(values[3],0)

    def test_fit_source_replication_preserves_model(self):
        source,data,ranks,reference=synthetic_fit()
        original,_,_,_=fit_normal_model(ranks,source,reference)
        order=np.r_[np.arange(3),np.arange(3),np.arange(3,len(source))]
        repeated={name:values[order] for name,values in ranks.items()}
        replicated,_,_,_=fit_normal_model(repeated,source[order],reference)
        for name in original:
            np.testing.assert_allclose(original[name],replicated[name],atol=2e-13,rtol=0)

    def test_full_native_coordinates_match_fit_maps(self):
        source,data,ranks,reference=synthetic_fit()
        _,native,z,_=fit_normal_model(ranks,source,reference)
        raw={name:data[:,index] for index,name in enumerate(('source_local','source_full','raw_route'))}
        reconstructed,_=answer_coordinates(reference,native,raw,dict(local=data[:,3],full=data[:,4]))
        np.testing.assert_array_equal(z,reconstructed)

    def test_risk_reference_midrank_constant_and_ties(self):
        source=np.array([0,0,1,1])
        reference=fit_risk_reference(np.zeros(4),source)
        np.testing.assert_array_equal(coupling_rank(reference,np.zeros(3)),np.full(3,.5))
        reference=fit_risk_reference(np.array([0.,0.,1.,2.]),source)
        np.testing.assert_array_equal(coupling_rank(reference,np.array([0.,1.,2.])),[.25,.625,.875])

    def test_constant_and_nonfinite_coordinates_rejected(self):
        source,_,ranks,reference=synthetic_fit()
        constant=dict(ranks,local_native_logp=np.zeros(len(source)))
        with self.assertRaises(ValueError):
            fit_normal_model(constant,source,reference)
        with self.assertRaises(ValueError):
            dependence_ratio(np.full((2,5),np.nan),analytic_model())

    def test_inherited_graph_matches_manual_two_node_system(self):
        attention=np.zeros((2,32,8))
        attention[1,:,0]=.8
        fields,diagnostics=graph_fields(np.array([.2,.8]),dict(native=local_weights(attention)))
        # The inherited optimizer stops at max gradient 1e-8. The positive
        # graph resolvent has row sum one, so its max solution error is bounded
        # by that same gradient residual; this is not a new detector tolerance.
        self.assertLessEqual(diagnostics['native']['max_gradient'],1e-8)
        np.testing.assert_allclose(fields['native_huber'],[1/3,2/3],atol=1e-8,rtol=0)
        fields,_=graph_fields(np.full(2,.5),dict(native=local_weights(attention)))
        np.testing.assert_array_equal(fields['native_huber'],np.full(2,.5))

    def test_valid_token_filter_occurs_after_full_graph(self):
        attention=np.zeros((3,32,8))
        attention[1:,:,0]=.8
        fields,_=graph_fields(np.array([0.,1.,0.]),dict(native=local_weights(attention)))
        # Middle raw degree is 1.6, so each edge becomes .5 before lambda .5.
        # The exact 3-node linear system therefore has endpoint value 1/7.
        np.testing.assert_allclose(fields['native_huber'][[0,2]],[1/7,1/7],atol=1e-8,rtol=0)


if __name__=='__main__':
    unittest.main()
