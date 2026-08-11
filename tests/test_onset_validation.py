import unittest

import torch

from graphs import TokenGraph
from onset_validation import causal_rewire, merge_positive_runs
from research_dataset import relations_from_graph, structural_features_from_relations


class _Attention:
    num_tokens = 9
    response_idx = 3
    num_response_tokens = 6
    num_channels = 4


def _nondegenerate_graph() -> TokenGraph:
    """A causal graph with swappable prompt and history relations."""
    source = torch.tensor(
        [0, 1, 2, 3, 0, 4, 1, 5, 2],
        dtype=torch.long,
    )
    target = torch.tensor(
        [3, 4, 5, 6, 6, 7, 7, 8, 8],
        dtype=torch.long,
    )
    edge_type = (source >= _Attention.response_idx).to(torch.int8)
    counts = torch.tensor([1 + index % 3 for index in range(len(source))])
    edge_ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    edge_channel = torch.cat(
        [torch.arange(count, dtype=torch.int32) % _Attention.num_channels for count in counts]
    )
    edge_value = torch.linspace(0.05, 0.45, edge_channel.numel())
    return TokenGraph(
        num_nodes=_Attention.num_tokens,
        response_idx=_Attention.response_idx,
        edge_index=torch.stack((source, target)),
        edge_type=edge_type,
        edge_ptr=edge_ptr,
        edge_channel=edge_channel,
        edge_value=edge_value,
    )


class OnsetValidationTests(unittest.TestCase):
    def test_merge_positive_runs_merges_gaps_of_zero_or_one_only(self):
        self.assertEqual(
            merge_positive_runs([[81, 84], [84, 87]]),
            [(81, 87)],
        )
        self.assertEqual(
            merge_positive_runs([[81, 84], [85, 87]]),
            [(81, 87)],
        )
        self.assertEqual(
            merge_positive_runs([[81, 84], [86, 87]]),
            [(81, 84), (86, 87)],
        )

    def test_causal_rewire_preserves_payload_degrees_and_node_features(self):
        attention = _Attention()
        original = _nondegenerate_graph()

        rewired, accepted = causal_rewire(
            original, seed=1, sweeps=1
        )

        original_source, original_target = original.edge_index
        rewired_source, rewired_target = rewired.edge_index
        self.assertGreater(accepted, 0)
        self.assertTrue(torch.equal(rewired_target, original_target))
        self.assertTrue(torch.equal(rewired.edge_type, original.edge_type))
        self.assertTrue(torch.equal(rewired.edge_ptr, original.edge_ptr))
        self.assertTrue(torch.equal(rewired.edge_channel, original.edge_channel))
        self.assertTrue(torch.equal(rewired.edge_value, original.edge_value))
        self.assertTrue(
            torch.equal(
                torch.bincount(rewired_source, minlength=attention.num_tokens),
                torch.bincount(original_source, minlength=attention.num_tokens),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.bincount(rewired_target, minlength=attention.num_tokens),
                torch.bincount(original_target, minlength=attention.num_tokens),
            )
        )
        self.assertTrue(bool((rewired_source < rewired_target).all()))
        self.assertTrue(
            torch.equal(
                rewired.edge_type,
                (rewired_source >= attention.response_idx).to(torch.int8),
            )
        )
        pairs = rewired_target * attention.num_tokens + rewired_source
        self.assertEqual(torch.unique(pairs).numel(), pairs.numel())
        self.assertTrue(bool((rewired_source != original_source).any()))

        original_features = structural_features_from_relations(
            attention, relations_from_graph(attention, original)
        )
        rewired_features = structural_features_from_relations(
            attention, relations_from_graph(attention, rewired)
        )
        invariant_columns = [
            index for index in range(original_features.shape[1]) if index != 3
        ]
        torch.testing.assert_close(
            rewired_features[:, invariant_columns],
            original_features[:, invariant_columns],
            rtol=0,
            atol=0,
        )
        self.assertTrue(bool((rewired_features[:, 3] != original_features[:, 3]).any()))

    def test_lazy_rewire_reaches_both_permutation_parities(self):
        source = torch.tensor([0, 1, 2], dtype=torch.long)
        target = torch.tensor([6, 7, 8], dtype=torch.long)
        graph = TokenGraph(
            num_nodes=9,
            response_idx=3,
            edge_index=torch.stack((source, target)),
            edge_type=torch.zeros(3, dtype=torch.int8),
        )

        states = {
            tuple(causal_rewire(graph, seed=seed, sweeps=10)[0].edge_index[0].tolist())
            for seed in range(64)
        }
        parities = {
            sum(state[left] > state[right] for left in range(3) for right in range(left + 1, 3)) % 2
            for state in states
        }

        self.assertGreater(len(states), 3)
        self.assertEqual(parities, {0, 1})


if __name__ == "__main__":
    unittest.main()
