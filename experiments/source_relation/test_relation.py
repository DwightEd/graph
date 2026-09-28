import unittest
import torch
from .measure import compare_paths, js, source_relation, normalize
from .kernel import message_kernel, mmd


class RelationTests(unittest.TestCase):
    def test_source_causality(self):
        q, k = torch.ones(2, 3, 4), torch.ones(1, 3, 4)
        result = source_relation(q, k, torch.tensor([2, 5, 9]))
        torch.testing.assert_close(result[0], torch.tensor([[0.,0.,0.],[1.,0.,0.],[.5,.5,0.]]))

    def test_reverse_graph_exposes_future_source_values_only(self):
        native = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[.5,.5,0.]]])
        reverse = normalize(native.transpose(-1,-2))
        expected = torch.tensor([[[0.,2/3,1/3],[0.,0.,1.],[0.,0.,0.]]])
        torch.testing.assert_close(reverse,expected)
        self.assertEqual(reverse.tril().abs().sum().item(),0.)

    def test_relation_discriminates_with_identical_current_reading(self):
        # Current choice reads source 2. Its prompt context is source 0.
        relation = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]]])
        good = torch.tensor([[[1.,0.,0.],[0.,0.,1.]]])
        bad = torch.tensor([[[0.,1.,0.],[0.,0.,1.]]])
        history = torch.tensor([[[0.],[1.]]])
        gd, gr, _ = compare_paths(good, history, relation)
        bd, br, _ = compare_paths(bad, history, relation)
        torch.testing.assert_close(gd[:,1], bd[:,1])
        self.assertAlmostEqual(gr[0,1].item(), 0.)
        self.assertGreater(br[0,1].item(), .69)

    def test_prediction_mapping_does_not_use_current_choice(self):
        direct = torch.eye(3)[None]
        history = torch.tensor([[[0.,0.],[1.,0.],[0.,1.]]])
        relation = torch.eye(3)[None]
        direct_js, _, state_js = compare_paths(direct, history, relation)
        self.assertGreater(direct_js[0,2].item(), .69)
        self.assertEqual(state_js[0,2].item(), 0.)
        self.assertTrue(torch.isnan(direct_js[0,0]))

    def test_js_symmetry_and_undefined(self):
        p, q = torch.tensor([[1.,0.]]), torch.tensor([[.25,.75]])
        torch.testing.assert_close(js(p,q), js(q,p))
        self.assertTrue(torch.isnan(js(p*0,q)).all())

    def test_native_gram_equals_full_output_geometry(self):
        torch.manual_seed(42)
        values = torch.randn(2,5,3,dtype=torch.float64)
        projection = torch.randn(2,7,3,dtype=torch.float64)
        gram = projection.transpose(-1,-2)@projection
        kernel,bandwidth = message_kernel(values,gram)
        projected = values@projection.transpose(-1,-2)
        distance = ((projected[:,:,None,:]-projected[:,None,:,:])**2).sum(-1)
        expected = torch.exp(-distance/(2*bandwidth[:,None,None]))
        torch.testing.assert_close(kernel,expected)

    def test_kernel_distinguishes_near_from_far_messages(self):
        values = torch.tensor([[[0.],[.01],[5.]]],dtype=torch.float64)
        kernel,_ = message_kernel(values,torch.ones(1,1,1,dtype=torch.float64))
        p,q,r = torch.eye(3,dtype=torch.float64).reshape(1,3,3).split(1,dim=1)
        self.assertLess(mmd(p,q,kernel).item(),mmd(p,r,kernel).item())
        self.assertAlmostEqual(mmd(p,p,kernel).item(),0.)
        torch.testing.assert_close(mmd(p,q,kernel),mmd(q,p,kernel))


if __name__ == '__main__':
    unittest.main()
