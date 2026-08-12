import unittest

import torch

from behavior import (
    align_error_onsets,
    centered_window,
    positive_mask,
    summarize_run_windows,
    validate_positive_runs,
)


class BehaviorTests(unittest.TestCase):
    def test_positive_runs_and_mask(self):
        self.assertEqual(
            validate_positive_runs(7, [[1, 3], [5, 7]]),
            ((1, 3), (5, 7)),
        )
        mask = positive_mask(7, [[1, 3], [5, 7]])
        torch.testing.assert_close(
            mask,
            torch.tensor([False, True, True, False, False, True, True]),
        )
        with self.assertRaises(ValueError):
            validate_positive_runs(7, [[2, 4], [3, 5]])

    def test_centered_window_nan_pads_boundaries(self):
        features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        window, valid = centered_window(features, center=0, radius=2)
        self.assertEqual(window.shape, (5, 2))
        torch.testing.assert_close(
            valid, torch.tensor([False, False, True, True, True])
        )
        self.assertTrue(torch.isnan(window[:2]).all())
        torch.testing.assert_close(window[2:], features[:3])

    def test_align_error_onsets_first_and_all(self):
        features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        first, first_valid = align_error_onsets(
            features, [[1, 2], [4, 5]], radius=1, policy="first"
        )
        all_windows, all_valid = align_error_onsets(
            features, [[1, 2], [4, 5]], radius=1, policy="all"
        )
        self.assertEqual(first.shape, (1, 3, 4))
        self.assertEqual(first_valid.shape, (1, 3))
        self.assertEqual(all_windows.shape, (2, 3, 4))
        self.assertEqual(all_valid.shape, (2, 3))

    def test_summarize_pre_error_post(self):
        features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        summary = summarize_run_windows(
            features, [[2, 4]], pre_window=2, post_window=2
        )
        expected = torch.stack(
            (features[:2].mean(0), features[2:4].mean(0), features[4:].mean(0))
        )
        torch.testing.assert_close(summary[0], expected)


if __name__ == "__main__":
    unittest.main()
