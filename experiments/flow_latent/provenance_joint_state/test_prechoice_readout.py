"""Scientific contracts for the candidate-dependent complete-state readout."""
import unittest

import torch

from .prechoice_readout import CandidateCompatibility, standardize_fit
from .source_prechoice_pilot import make_controls


class CandidateReadoutTests(unittest.TestCase):
    def test_all_coordinates_and_sites_have_exact_full_response(self):
        torch.manual_seed(8)
        model = CandidateCompatibility(2, 3, 4).double()
        states = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        candidates = torch.randn(2, 4, dtype=torch.float64)
        score = model(states, candidates)[0, 0]
        gradient = torch.autograd.grad(score, states)[0]
        expected = model.site_weight[None, :, :, None] * model.coordinate * candidates[0] / 2
        torch.testing.assert_close(gradient, expected)
        self.assertTrue(bool(torch.all(gradient != 0)))
        reordered = model(states, candidates.flip(0))
        torch.testing.assert_close(reordered, model(states, candidates).flip(1))

    def test_leave_out_state_cannot_change_fit_normalization(self):
        states = torch.randn(4, 2, 3, 4)
        mask = torch.tensor([True, True, False, False])
        normalized, mean, scale = standardize_fit(states, mask)
        states[~mask] += 1000
        changed, second_mean, second_scale = standardize_fit(states, mask)
        torch.testing.assert_close(mean, second_mean)
        torch.testing.assert_close(scale, second_scale)
        torch.testing.assert_close(normalized[mask], changed[mask])

    def test_source_swap_reverses_same_query_candidate_truth(self):
        row = dict(source_id='1', family='boolean_attribute', left=['attributes', 'a'],
                   right=['attributes', 'b'], value_left=True, value_right=False,
                   original_prompt="Structured data:\n{'attributes': {'a': True, 'b': False}}\nOverview:")
        controls = make_controls([row])
        self.assertEqual([case['correct'] for case in controls], [0, 1, 1, 0])
        self.assertEqual(controls[0]['path'], controls[2]['path'])
        self.assertEqual(controls[1]['path'], controls[3]['path'])


if __name__ == '__main__':
    unittest.main()
