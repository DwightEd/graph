import unittest

import numpy as np
import torch

from graph_features import (
    BASIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_NAMES,
    RESPONSE_FEATURE_NAMES,
    STATIC_BLOCK_NAMES,
    basic_structural_features,
    block_slices,
    dynamic_state,
    response_graph_features,
    source_transition_features,
    static_feature_blocks,
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
            "source": torch.tensor([0, 1, 0, 2]),
            "target": torch.tensor([2, 2, 3, 3]),
            "weight": torch.tensor([0.4, 0.2, 0.05, 0.4]),
            "channel_count": torch.tensor([2, 1, 1, 2]),
            "edge_type": torch.tensor([0, 0, 0, 1]),
        }

    def attention_edges(self):
        return {
            "layer": torch.zeros(6, dtype=torch.long),
            "head": torch.tensor([0, 0, 0, 0, 1, 1]),
            "source": torch.tensor([0, 1, 0, 2, 0, 2]),
            "target": torch.tensor([2, 2, 3, 3, 2, 3]),
            "weight": torch.tensor([0.2, 0.4, 0.1, 0.3, 0.6, 0.5]),
        }


class GraphFeatureTests(unittest.TestCase):
    def test_basic_and_full_response_dimensions(self):
        sample = _Sample()
        basic = basic_structural_features(sample.attention(), sample.relation_edges())
        full = response_graph_features(sample)
        self.assertEqual(basic.shape, (2, len(BASIC_FEATURE_NAMES)))
        self.assertEqual(full.shape, (2, len(RESPONSE_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(full).all())

    def test_static_blocks_have_one_33d_state(self):
        full = response_graph_features(_Sample())
        blocks = static_feature_blocks(full)
        self.assertEqual(tuple(blocks), STATIC_BLOCK_NAMES)
        self.assertEqual(sum(value.shape[1] for value in blocks.values()), 33)

    def test_dynamic_state_has_19_features(self):
        sample = _Sample()
        full = response_graph_features(sample)
        blocks = static_feature_blocks(full)
        static = np.concatenate(
            [blocks[name] / np.sqrt(blocks[name].shape[1]) for name in STATIC_BLOCK_NAMES],
            axis=1,
        )
        changes = source_transition_features(sample, full)
        dynamic = dynamic_state(static, block_slices(blocks), changes)
        self.assertEqual(dynamic.shape, (2, len(DYNAMIC_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(dynamic).all())


if __name__ == "__main__":
    unittest.main()
