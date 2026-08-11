import unittest

import numpy as np

from transition_visualization import (
    STATIC_BLOCK_NAMES,
    block_balanced_state,
    run_windows,
    static_feature_blocks,
)
from rich_visualization import RICH_RESPONSE_FEATURE_NAMES


class TransitionVisualizationTests(unittest.TestCase):
    def _empty_rich(self, rows=3):
        return np.zeros((rows, len(RICH_RESPONSE_FEATURE_NAMES)), dtype=np.float32)

    def test_locality_bins_are_mutually_exclusive(self):
        features = self._empty_rich(rows=2)
        index = {name: i for i, name in enumerate(RICH_RESPONSE_FEATURE_NAMES)}
        features[0, index["history_mass"]] = 1.0
        features[0, index["history_near1_share"]] = 0.2
        features[0, index["history_near4_share"]] = 0.5
        features[0, index["history_near8_share"]] = 0.7
        features[0, index["history_far16_share"]] = 0.1

        blocks = static_feature_blocks(features)
        locality = blocks["locality"][0]
        np.testing.assert_allclose(locality[2:], [0.2, 0.3, 0.2, 0.2, 0.1], atol=1e-6)
        self.assertAlmostEqual(float(locality[2:].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(blocks["locality"][1, 2:].sum()), 0.0, places=6)

    def test_nearby_hallucination_runs_are_analyzed_independently(self):
        windows = run_windows(
            134,
            [(81, 84), (85, 87)],
            pre_window=10,
            post_window=10,
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].clean_pre_start, 71)
        self.assertEqual(windows[0].clean_pre_end, 81)
        self.assertEqual(windows[0].clean_post_start, 84)
        self.assertEqual(windows[0].clean_post_end, 85)
        self.assertEqual(windows[1].clean_pre_start, 84)
        self.assertEqual(windows[1].clean_pre_end, 85)
        self.assertEqual(windows[1].pre_length, 1)

    def test_block_balancing_gives_each_block_equal_budget_for_unit_vectors(self):
        blocks = {}
        for block_index, name in enumerate(STATIC_BLOCK_NAMES):
            width = 2 + block_index
            block = np.zeros((5, width), dtype=np.float32)
            block[-1] = 1.0
            blocks[name] = block
        balanced = block_balanced_state(blocks)
        offset = 0
        norms = []
        for name in STATIC_BLOCK_NAMES:
            width = blocks[name].shape[1]
            norms.append(float(np.linalg.norm(balanced[-1, offset : offset + width])))
            offset += width
        self.assertLess(max(norms) - min(norms), 1e-5)


if __name__ == "__main__":
    unittest.main()
