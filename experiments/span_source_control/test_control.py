"""Mathematical and native causal checks needed for interpreting the experiment."""
import unittest
import numpy as np
import torch

from .measure import centered_effect


class SourceControlTest(unittest.TestCase):
    def test_softmax_gradient(self):
        scores = torch.tensor([.3, -.7, .9], dtype=torch.float64, requires_grad=True)
        values = torch.tensor([2., -3., 1.], dtype=torch.float64)
        attention = scores.softmax(-1)
        output = (attention * values).sum()
        gradient = torch.autograd.grad(output, scores)[0]
        result = centered_effect(attention.detach().numpy(), (attention * values).detach().numpy())
        np.testing.assert_allclose(result, gradient.numpy(), atol=1e-12)
        self.assertAlmostEqual(result.sum(), 0., places=12)


if __name__ == '__main__':
    unittest.main()
