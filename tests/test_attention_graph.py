import unittest

import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.model import (
    MaskedAttentionAutoencoder,
    random_target_view,
    reconstruction_losses,
    target_masked_view,
)
from attention_graph.score import RobustResidualCalibrator, score_graph_raw
from attention_graph.statistics import TOKEN_FEATURES, token_statistics


class Sample:
    sample_id = "r1"
    source_id = "s1"
    response_idx = 2
    attention_floor = 0.01
    token_ids = torch.tensor([10, 11, 12, 13])
    attention_diagonal = torch.tensor([
        [[0.8, 0.7, 0.6, 0.5], [0.7, 0.6, 0.5, 0.4]]
    ], dtype=torch.float16)
    response_row_ptr = torch.tensor([0, 2, 4, 5, 6])
    response_column_indices = torch.tensor([0, 1, 0, 2, 0, 2], dtype=torch.int32)
    response_values = torch.tensor([0.2, 0.4, 0.1, 0.3, 0.6, 0.5], dtype=torch.float16)
    num_layers = 1
    num_heads = 2
    num_tokens = 4
    num_response_tokens = 2
    num_channels = 2


class AttentionGraphTests(unittest.TestCase):
    def test_graph_preserves_pair_topology_and_channel_traces(self):
        graph = build_attention_graph(Sample(), GraphBuildConfig())
        self.assertEqual(graph.edge_index.tolist(), [[0, 1, 0, 2], [2, 2, 3, 3]])
        self.assertEqual(graph.edge_type.tolist(), [0, 0, 0, 1])
        self.assertEqual(graph.trace_edge_id.numel(), 6)
        self.assertEqual(graph.trace_channel.numel(), 6)
        self.assertEqual(graph.node_attr.shape, (4, 2))

    def test_encoder_and_reconstruction_are_label_free_and_finite(self):
        graph = build_attention_graph(Sample())
        model = MaskedAttentionAutoencoder(
            num_channels=2, embedding_dim=8, message_steps=2, dropout=0.0
        )
        generator = torch.Generator().manual_seed(3)
        view = random_target_view(
            graph, target_mask_rate=0.5, channel_drop_rate=0.0, generator=generator
        )
        hidden = model.encode(graph, view)
        self.assertEqual(hidden.shape, (4, 8))
        losses = reconstruction_losses(model, graph, view, generator=generator)
        self.assertTrue(torch.isfinite(losses.total))

    def test_leave_one_out_scoring_returns_learned_embeddings(self):
        graph = build_attention_graph(Sample())
        model = MaskedAttentionAutoencoder(
            num_channels=2, embedding_dim=8, message_steps=1, dropout=0.0
        )
        embedding, residual = score_graph_raw(model, graph, target_block_size=1, seed=1)
        self.assertEqual(embedding.shape, (2, 8))
        self.assertEqual(residual.shape, (2, 6))
        calibrator = RobustResidualCalibrator.fit(residual + 1e-3)
        z, score = calibrator.transform(residual)
        self.assertEqual(z.shape, residual.shape)
        self.assertEqual(score.shape, (2,))

    def test_statistics_are_diagnostics_not_model_input(self):
        graph = build_attention_graph(Sample())
        values = token_statistics(graph)
        self.assertEqual(values.shape, (2, len(TOKEN_FEATURES)))
        self.assertTrue(torch.isfinite(values).all())

    def test_target_mask_hides_every_incoming_edge_of_target(self):
        graph = build_attention_graph(Sample())
        view = target_masked_view(graph, torch.tensor([3]))
        self.assertTrue(view.node_mask[3])
        self.assertFalse(view.visible_edge_mask[graph.edge_index[1] == 3].any())
        self.assertTrue(view.visible_edge_mask[graph.edge_index[1] == 2].all())


if __name__ == "__main__":
    unittest.main()
