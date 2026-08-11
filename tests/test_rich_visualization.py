import unittest

import numpy as np
import torch

from rich_visualization import (
    RICH_RESPONSE_FEATURE_NAMES,
    SOURCE_ROLE_FEATURE_NAMES,
    response_phase_labels,
    rich_response_features,
    source_role_features,
)


class _Attention:
    response_idx = 2
    num_response_tokens = 2
    num_tokens = 4
    num_layers = 1
    num_heads = 2
    num_channels = 2


class _Sample:
    def __init__(self):
        self._attention = _Attention()

    def attention(self):
        return self._attention

    def relation_edges(self):
        return {
            "source": torch.tensor([0, 1, 0, 2], dtype=torch.long),
            "target": torch.tensor([2, 2, 3, 3], dtype=torch.long),
            "weight": torch.tensor([0.4, 0.2, 0.05, 0.4], dtype=torch.float32),
            "channel_count": torch.tensor([2, 1, 1, 2], dtype=torch.long),
            "edge_type": torch.tensor([0, 0, 0, 1], dtype=torch.long),
        }

    def attention_edges(self):
        return {
            "layer": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
            "head": torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
            "source": torch.tensor([0, 1, 0, 2, 0, 2], dtype=torch.long),
            "target": torch.tensor([2, 2, 3, 3, 2, 3], dtype=torch.long),
            "weight": torch.tensor([0.2, 0.4, 0.1, 0.3, 0.6, 0.5], dtype=torch.float32),
        }


class RichVisualizationTests(unittest.TestCase):
    def test_rich_response_features_capture_grounding_history_and_concentration(self):
        features = rich_response_features(_Sample())
        self.assertEqual(features.shape, (2, len(RICH_RESPONSE_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())

        index = {name: i for i, name in enumerate(RICH_RESPONSE_FEATURE_NAMES)}
        self.assertAlmostEqual(float(features[0, index["prompt_mass"]]), 0.6, places=5)
        self.assertAlmostEqual(float(features[1, index["history_mass"]]), 0.4, places=5)
        self.assertAlmostEqual(
            float(features[1, index["prompt_mass_share"]]), 1.0 / 9.0, places=5
        )
        self.assertAlmostEqual(
            float(features[1, index["history_near1_share"]]), 1.0, places=5
        )
        self.assertGreater(float(features[1, index["hhi"]]), 0.7)

    def test_source_role_features_cover_prompt_and_response_nodes(self):
        features = source_role_features(_Sample())
        self.assertEqual(features.shape, (4, len(SOURCE_ROLE_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        out_degree = SOURCE_ROLE_FEATURE_NAMES.index("out_degree")
        self.assertEqual(features[:, out_degree].tolist(), [2.0, 1.0, 1.0, 0.0])

    def test_phase_labels_separate_far_pre_error_and_post(self):
        phases = response_phase_labels(12, [(5, 7)], pre_window=3, post_window=2)
        self.assertEqual(
            phases.tolist(),
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0, 0],
        )

    def test_overlapping_windows_never_overwrite_hallucination(self):
        phases = response_phase_labels(10, [(3, 4), (6, 7)], pre_window=4, post_window=4)
        self.assertEqual(int(phases[3]), 2)
        self.assertEqual(int(phases[6]), 2)


if __name__ == "__main__":
    unittest.main()
