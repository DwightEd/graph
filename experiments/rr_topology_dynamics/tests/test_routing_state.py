import unittest

import torch

from experiments.rr_topology_dynamics.attractor import PRIMARY_FEATURE_NAMES
from experiments.rr_topology_dynamics.extractor import TopologyDynamicsExtractor
from experiments.rr_topology_dynamics.routing_state import RoutingStateExtractor
from research_dataset import SparseAttentionBlock


class _AttentionGeometry:
    num_response_tokens = 3
    response_idx = 2


class _SparseSample:
    def attention(self):
        return _AttentionGeometry()

    def iter_sparse_attention_blocks(self, block_rows=4096):
        del block_rows
        yield SparseAttentionBlock(
            row=torch.arange(6),
            layer=torch.zeros(6, dtype=torch.long),
            head=torch.zeros(6, dtype=torch.long),
            query=torch.tensor([0, 0, 1, 1, 2, 2]),
            target=torch.tensor([2, 2, 3, 3, 4, 4]),
            source=torch.tensor([0, 1, 0, 2, 1, 3]),
            weight=torch.tensor([0.1, 0.3, 0.2, 0.4, 0.05, 0.5]),
        )


class RoutingStateExtractorTest(unittest.TestCase):
    def test_preserves_exact_prompt_and_response_source_identity(self):
        state = RoutingStateExtractor().extract(_SparseSample())

        torch.testing.assert_close(
            state.prompt_source_mass,
            torch.tensor(
                [
                    [0.1, 0.3],
                    [0.2, 0.0],
                    [0.0, 0.05],
                ]
            ),
        )
        torch.testing.assert_close(
            state.response_source_mass,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            state.retained_edge_count, torch.tensor([2.0, 2.0, 2.0])
        )

    def test_topology_extractor_exposes_only_predeclared_primary_features(self):
        extracted = TopologyDynamicsExtractor().extract(_SparseSample())

        self.assertEqual(extracted.feature_names, PRIMARY_FEATURE_NAMES)
        self.assertEqual(extracted.features.shape, (3, len(PRIMARY_FEATURE_NAMES)))
        self.assertEqual(extracted.controls.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
