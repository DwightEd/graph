from dataclasses import replace
import unittest

import torch

from attention_gnn import (
    RelationChannelEncoder,
    RelationChannelAutoencoder,
    build_attention_graph,
    masked_view,
    reconstruction_loss,
)
from cache import AttentionSample


RESPONSE_IDX = 2
ROWS = {
    (0, 2): ([0, 1], [0.40, 0.20]),
    (0, 3): ([0, 2], [0.30, 0.50]),
    (0, 4): ([1, 3], [0.25, 0.45]),
    (1, 2): ([1], [0.35]),
    (1, 3): ([0, 2], [0.15, 0.55]),
    (1, 4): ([0, 3], [0.10, 0.50]),
    (2, 2): ([0], [0.60]),
    (2, 3): ([1], [0.40]),
    (2, 4): ([2, 3], [0.20, 0.70]),
    (3, 2): ([0, 1], [0.30, 0.10]),
    (3, 3): ([2], [0.65]),
    (3, 4): ([1], [0.33]),
}


def attention_sample(num_tokens=5):
    row_ptr = [0]
    columns = []
    values = []
    for channel in range(4):
        for target in range(RESPONSE_IDX, num_tokens):
            source, weight = ROWS[(channel, target)]
            columns.extend(source)
            values.extend(weight)
            row_ptr.append(len(values))

    channel = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1)
    position = torch.arange(num_tokens, dtype=torch.float32).reshape(1, 1, -1)
    diagonal = 0.01 + channel / 10 + position / 100
    sample = AttentionSample(
        sample_id="response-1",
        source_id="source-1",
        response_idx=RESPONSE_IDX,
        token_ids=torch.arange(num_tokens, dtype=torch.int32),
        attention_diagonal=diagonal.to(torch.float16),
        response_row_ptr=torch.tensor(row_ptr, dtype=torch.int32),
        response_column_indices=torch.tensor(columns, dtype=torch.int32),
        response_values=torch.tensor(values, dtype=torch.float16),
        attention_floor=0.01,
    )
    sample.validate()
    sample.y_token = torch.ones(num_tokens, dtype=torch.int64)
    return sample


class AttentionGraphConstructionTests(unittest.TestCase):
    def test_canonical_csr_becomes_exact_sparse_edge_channel_traces_without_labels(self):
        graph = build_attention_graph(attention_sample())

        self.assertEqual(
            graph.edge_index.tolist(),
            [
                [0, 1, 0, 1, 2, 0, 1, 2, 3],
                [2, 2, 3, 3, 3, 4, 4, 4, 4],
            ],
        )
        self.assertEqual(graph.edge_type.tolist(), [0, 0, 0, 0, 1, 0, 0, 1, 1])
        self.assertEqual(graph.edge_ptr.tolist(), [0, 3, 6, 8, 9, 12, 13, 15, 16, 19])
        self.assertEqual(
            graph.edge_channel.tolist(),
            [0, 2, 3, 0, 1, 3, 0, 1, 2, 0, 1, 3, 1, 0, 3, 2, 0, 1, 2],
        )
        torch.testing.assert_close(
            graph.edge_value.float(),
            torch.tensor(
                [
                    0.40, 0.60, 0.30, 0.20, 0.35, 0.10, 0.30, 0.15, 0.40,
                    0.50, 0.55, 0.65, 0.10, 0.25, 0.33, 0.20, 0.45, 0.50, 0.70,
                ]
            ),
            atol=5e-4,
            rtol=0,
        )
        self.assertFalse(
            any("label" in name.casefold() or name in {"y", "y_token"} for name in vars(graph))
        )


class RelationChannelEncoderTests(unittest.TestCase):
    def test_edge_embedding_preserves_total_attention_mass(self):
        graph = build_attention_graph(attention_sample())
        encoder = RelationChannelEncoder(
            num_channels=4, embedding_dim=8, message_passing_steps=1, dropout=0.0
        ).eval()
        index, original = encoder._edge_embedding(graph, masked_view(graph))
        doubled = replace(graph, edge_value=graph.edge_value * 2)
        _, changed = encoder._edge_embedding(doubled, masked_view(doubled))

        self.assertTrue(torch.equal(index, doubled.edge_index))
        self.assertFalse(torch.allclose(original, changed))

    def test_forward_returns_one_embedding_per_node(self):
        graph = build_attention_graph(attention_sample())
        encoder = RelationChannelEncoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=1,
            dropout=0.0,
        ).eval()

        embedding = encoder(graph, masked_view(graph))

        self.assertEqual(embedding.shape, (graph.num_nodes, 8))

    def test_masked_edge_and_channel_payload_cannot_enter_encoder(self):
        graph = build_attention_graph(attention_sample())
        masked_edges = torch.tensor([0])
        masked_channels = torch.tensor([1])
        trace_edge = torch.repeat_interleave(
            torch.arange(graph.edge_index.shape[1]), graph.edge_ptr.diff()
        )
        hidden_payload = (trace_edge == 0) | (graph.edge_channel == 1)
        changed_value = graph.edge_value.clone()
        changed_value[hidden_payload] += 7
        changed = replace(graph, edge_value=changed_value)

        torch.manual_seed(7)
        encoder = RelationChannelEncoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=2,
            dropout=0.0,
        ).eval()
        original_embedding = encoder.encode(
            graph,
            masked_view(
                graph,
                masked_edges=masked_edges,
                masked_channels=masked_channels,
            ),
        )
        changed_embedding = encoder.encode(
            changed,
            masked_view(
                changed,
                masked_edges=masked_edges,
                masked_channels=masked_channels,
            ),
        )

        torch.testing.assert_close(original_embedding, changed_embedding)

    def test_full_graph_and_causal_prefix_have_identical_shared_node_embeddings(self):
        full = build_attention_graph(attention_sample(5))
        prefix = build_attention_graph(attention_sample(4))
        torch.manual_seed(11)
        encoder = RelationChannelEncoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=2,
            dropout=0.0,
        ).eval()

        full_embedding = encoder.encode(full, masked_view(full))
        prefix_embedding = encoder.encode(prefix, masked_view(prefix))

        torch.testing.assert_close(
            full_embedding[: prefix.num_nodes], prefix_embedding, atol=1e-6, rtol=1e-6
        )

    def test_zero_message_encoder_is_rewire_invariant_but_message_encoder_is_not(self):
        graph = build_attention_graph(attention_sample())
        rewired_source = graph.edge_index[0].clone()
        rewired_source[[0, 1, 2, 3, 5, 6, 7, 8]] = rewired_source[
            [1, 0, 3, 2, 6, 5, 8, 7]
        ]
        rewired = replace(
            graph,
            edge_index=torch.stack((rewired_source, graph.edge_index[1])),
        )

        torch.manual_seed(13)
        feature_only = RelationChannelEncoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=0,
            dropout=0.0,
        ).eval()
        torch.testing.assert_close(
            feature_only.encode(graph, masked_view(graph)),
            feature_only.encode(rewired, masked_view(rewired)),
        )

        torch.manual_seed(13)
        message_encoder = RelationChannelEncoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=1,
            dropout=0.0,
        ).eval()
        original = message_encoder.encode(graph, masked_view(graph))
        changed = message_encoder.encode(rewired, masked_view(rewired))
        self.assertFalse(torch.allclose(original[graph.response_idx :], changed[graph.response_idx :]))


class RelationChannelAutoencoderTests(unittest.TestCase):
    def test_reconstructs_only_masked_edge_support_and_channel_values(self):
        graph = build_attention_graph(attention_sample())
        view = masked_view(
            graph,
            masked_edges=torch.tensor([0, 4]),
            masked_channels=torch.tensor([1]),
        )
        torch.manual_seed(17)
        model = RelationChannelAutoencoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=2,
            dropout=0.0,
        )

        losses = reconstruction_loss(model, graph, view)

        self.assertEqual(set(losses), {"support", "weight", "distribution", "total"})
        self.assertTrue(all(loss.ndim == 0 for loss in losses.values()))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        self.assertGreater(float(losses["total"].detach()), 0.0)
        losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_masked_distribution_reconstruction_includes_censored_other_mass(self):
        graph = build_attention_graph(attention_sample())
        view = masked_view(graph, masked_channels=torch.tensor([0]))
        model = RelationChannelAutoencoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=1,
            dropout=0.0,
        )

        losses = reconstruction_loss(model, graph, view)

        self.assertIn("distribution", losses)
        self.assertTrue(torch.isfinite(losses["distribution"]))
        self.assertGreaterEqual(float(losses["distribution"].detach()), 0.0)

    def test_distribution_target_excludes_unrelated_row_payload(self):
        graph = build_attention_graph(attention_sample())
        view = masked_view(graph, masked_edges=torch.tensor([0]))
        changed_values = graph.edge_value.clone()
        trace_edge = torch.repeat_interleave(
            torch.arange(graph.edge_index.shape[1]), graph.edge_ptr.diff()
        )
        masked_target = graph.edge_index[1, 0]
        changed_values[graph.edge_index[1, trace_edge] != masked_target] = 0.99
        changed = replace(graph, edge_value=changed_values)
        torch.manual_seed(23)
        model = RelationChannelAutoencoder(
            num_channels=4,
            embedding_dim=8,
            message_passing_steps=1,
            dropout=0.0,
        )

        original = reconstruction_loss(model, graph, view)["distribution"]
        modified = reconstruction_loss(model, changed, view)["distribution"]

        torch.testing.assert_close(original, modified)


if __name__ == "__main__":
    unittest.main()
