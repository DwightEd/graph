import unittest

import torch

from evidence_graph import EvidenceGraphConfig, build_evidence_graph


class FakeAttention:
    def __init__(self):
        self.num_layers = 2
        self.num_heads = 1
        self.num_channels = 2
        self.num_tokens = 6
        self.response_idx = 3
        self.num_response_tokens = 3
        self.attention_floor = 0.01
        self.attention_diagonal = torch.full((2, 1, 6), 0.1, dtype=torch.float16)

        rows = [
            ([0, 1], [0.40, 0.10]),
            ([0, 3], [0.10, 0.50]),
            ([1, 4], [0.10, 0.60]),
            ([0, 2], [0.30, 0.20]),
            ([0, 3], [0.10, 0.40]),
            ([1, 4], [0.10, 0.50]),
        ]
        ptr = [0]
        columns, values = [], []
        for source, weight in rows:
            columns.extend(source)
            values.extend(weight)
            ptr.append(len(values))
        self.response_row_ptr = torch.tensor(ptr, dtype=torch.int32)
        self.response_column_indices = torch.tensor(columns, dtype=torch.int32)
        self.response_values = torch.tensor(values, dtype=torch.float16)


class EvidenceGraphTests(unittest.TestCase):
    def test_mass_cover_keeps_minimum_typed_support(self):
        graph = build_evidence_graph(
            FakeAttention(), EvidenceGraphConfig(mass_cover=0.75, relay_discount=0.85)
        )
        self.assertEqual(
            graph.edge_index.tolist(),
            [[0, 2, 0, 3, 1, 4], [3, 3, 4, 4, 5, 5]],
        )
        self.assertEqual(graph.edge_type.tolist(), [0, 0, 0, 1, 0, 1])
        self.assertEqual(graph.response_state.shape, (3, 22))
        self.assertEqual(graph.edge_attr.shape[1], 9)

    def test_provenance_flows_through_history_nodes(self):
        graph = build_evidence_graph(
            FakeAttention(), EvidenceGraphConfig(mass_cover=1.0, relay_discount=0.85)
        )
        names = {name: index for index, name in enumerate(graph.response_state_names)}
        ancestry = graph.response_state[:, names["prompt_ancestry"]]
        relay = graph.response_state[:, names["grounded_history_relay"]]
        self.assertGreater(float(ancestry[0]), 0.0)
        self.assertGreater(float(relay[1]), 0.0)
        self.assertGreater(float(relay[2]), 0.0)

    def test_no_labels_are_stored(self):
        graph = build_evidence_graph(FakeAttention())
        self.assertFalse(
            any("label" in name or name.startswith("y") for name in vars(graph))
        )


if __name__ == "__main__":
    unittest.main()
