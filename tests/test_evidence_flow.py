import unittest

import numpy as np
import torch

from attention_graph.evidence_flow import (
    _lag_bin_rewired_sources,
    anomaly_components,
    anomaly_components_from_attention,
    direct_field_names,
    evidence_flow_from_attention,
    evidence_flow_fields,
    lookback_evidence_from_attention,
    propagation_field_names,
)
from cache import AttentionSample


def _route(*, source):
    return {
        "channel": torch.tensor([0, 1, 0, 1]),
        "layer": torch.tensor([0, 0, 0, 0]),
        "head": torch.tensor([0, 1, 0, 1]),
        "source": torch.tensor(source),
        "target": torch.tensor([2, 2, 2, 2]),
        "weight": torch.tensor([.25, .4, .5, .6]),
        "attention_floor": .1,
    }


class EvidenceFlowTests(unittest.TestCase):
    def test_fused_lookback_and_evidence_matches_two_pass_reference(self):
        attention = AttentionSample(
            "fused", "source", 2, torch.arange(6, dtype=torch.int32),
            torch.tensor([[[.0, .0, .1, .2, .1, .3],
                           [.0, .0, .2, .1, .2, .1]]], dtype=torch.float16),
            torch.tensor([0, 1, 2, 4, 5, 6, 7, 9, 10], dtype=torch.int32),
            torch.tensor([0, 1, 0, 2, 3, 1, 2, 0, 2, 4], dtype=torch.int32),
            torch.tensor([.2, .3, .1, .4, .2, .5, .3, .2, .1, .6], dtype=torch.float16),
            .01,
        )
        attention.validate()
        from attention_graph.token_representation import direct_lookback_channels

        expected_lookback = direct_lookback_channels(attention, csr_row_block=3)
        expected = evidence_flow_from_attention(
            expected_lookback, attention, csr_row_block=3,
            sample_id="fused", seed=7,
        )
        actual = lookback_evidence_from_attention(
            attention, csr_row_block=3, sample_id="fused", seed=7,
        )
        fragmented = lookback_evidence_from_attention(
            attention, csr_row_block=1, sample_id="fused", seed=7,
        )

        torch.testing.assert_close(actual[0], expected_lookback)
        for left, right in zip(actual[1:4], expected[:3]):
            torch.testing.assert_close(left, right)
        self.assertEqual(actual[4], expected[3])
        for left, right in zip(actual[:4], fragmented[:4]):
            torch.testing.assert_close(left, right)
        self.assertEqual(actual[4], fragmented[4])

    def test_streaming_attention_flow_is_block_size_invariant(self):
        """Production flow reads CSR blocks, not a materialized edge route."""
        attention = AttentionSample(
            "stream", "source", 2, torch.arange(6, dtype=torch.int32),
            torch.zeros((1, 2, 6), dtype=torch.float16),
            torch.tensor([0, 1, 2, 4, 5, 6, 7, 9, 10], dtype=torch.int32),
            torch.tensor([0, 1, 0, 2, 3, 1, 4, 0, 2, 4], dtype=torch.int32),
            torch.tensor([.2, .3, .1, .4, .2, .5, .3, .2, .1, .6], dtype=torch.float16),
            .01,
        )
        lookback = torch.tensor([
            [[.2, .3]], [[.4, .5]], [[.6, .7]], [[.8, .9]],
        ])

        single = evidence_flow_from_attention(
            lookback, attention, csr_row_block=4, sample_id="stream", seed=7,
        )
        fragmented = evidence_flow_from_attention(
            lookback, attention, csr_row_block=1, sample_id="stream", seed=7,
        )

        for left, right in zip(single[:3], fragmented[:3]):
            torch.testing.assert_close(left, right)
        self.assertEqual(single[3], fragmented[3])

    def test_lag_bucket_rewire_changes_every_edge_with_an_alternative(self):
        source_relative = torch.tensor([3, 2, 1, 0, 1])
        target_relative = torch.tensor([7, 7, 7, 7, 2])

        first = _lag_bin_rewired_sources(
            source_relative,
            target_relative,
            prompt_count=10,
            seed=23,
            sample_id="vectorized-rewire",
        )
        second = _lag_bin_rewired_sources(
            source_relative,
            target_relative,
            prompt_count=10,
            seed=23,
            sample_id="vectorized-rewire",
        )

        original = source_relative + 10
        torch.testing.assert_close(first, second)
        self.assertTrue(bool((first[:4] != original[:4]).all()))
        self.assertEqual(int(first[4]), int(original[4]))
        self.assertTrue(bool((first >= 10).all()))
        self.assertTrue(bool((first < target_relative + 10).all()))
        original_lag = target_relative - source_relative
        rewired_lag = target_relative - (first - 10)
        torch.testing.assert_close(
            torch.floor(torch.log2(rewired_lag.float())),
            torch.floor(torch.log2(original_lag.float())),
        )

    def test_one_hop_channel_fields_are_exact_direct_mass_and_residual_flows(self):
        lookback = torch.tensor([[[.2, .3]], [[.8, .9]]])
        direct, propagation, _ = evidence_flow_fields(
            lookback, _route(source=[0, 0, 1, 1]),
            prompt_count=1,
        )

        self.assertEqual(direct.shape, (2, 4))
        self.assertEqual(propagation.shape, (2, 4))
        torch.testing.assert_close(direct[0], torch.zeros(4))
        torch.testing.assert_close(
            direct[1], torch.tensor([.25, .4, .5, .6]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(propagation[0], torch.zeros(4))
        torch.testing.assert_close(
            propagation[1], torch.tensor([.175, .32, .3, .36]),
            atol=1e-6, rtol=1e-6,
        )

    def test_changing_one_head_edge_changes_only_its_channel(self):
        lookback = torch.tensor([[[.2, .3]], [[.4, .5]], [[.8, .9]]])
        route = {
            "channel": torch.tensor([0, 1, 0, 1]),
            "layer": torch.zeros(4, dtype=torch.long),
            "head": torch.tensor([0, 1, 0, 1]),
            "source": torch.tensor([0, 0, 1, 2]),
            "target": torch.full((4,), 3, dtype=torch.long),
            "weight": torch.tensor([.25, .4, .5, .6]),
            "attention_floor": .1,
        }
        first = evidence_flow_fields(
            lookback, route,
            prompt_count=1,
        )[1]
        second = evidence_flow_fields(
            lookback, {**route, "source": torch.tensor([0, 0, 1, 1])},
            prompt_count=1,
        )[1]

        torch.testing.assert_close(first[:, :2], second[:, :2])
        torch.testing.assert_close(first[:, 2], second[:, 2])
        self.assertNotEqual(float(first[2, 3]), float(second[2, 3]))

    def test_field_names_match_layer_head_width(self):
        self.assertEqual(len(direct_field_names(2, 3)), 12)
        self.assertEqual(len(propagation_field_names(2, 3)), 12)

    def test_causal_rewired_control_preserves_metadata_and_lag_bucket(self):
        lookback = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2) / 20
        route = {
            "channel": torch.tensor([0, 1, 2, 3, 0, 1]),
            "layer": torch.tensor([0, 0, 1, 1, 0, 0]),
            "head": torch.tensor([0, 1, 0, 1, 0, 1]),
            "source": torch.tensor([0, 1, 2, 3, 4, 5]),
            "target": torch.tensor([6, 6, 6, 6, 6, 6]),
            "weight": torch.tensor([.2, .3, .4, .5, .6, .7]),
            "attention_floor": .01,
        }

        _, _, diagnostics = evidence_flow_fields(
            lookback, route, prompt_count=2,
            sample_id="rewired", seed=4, randomize_rr=True,
        )
        control = diagnostics["rewired_route"]
        original_rr = route["source"] >= 2
        control_rr = control["source"] >= 2

        for name in ("channel", "layer", "head", "target", "weight"):
            torch.testing.assert_close(control[name], route[name])
        torch.testing.assert_close(control_rr, original_rr)
        torch.testing.assert_close(
            control["source"][~control_rr], route["source"][~original_rr]
        )
        torch.testing.assert_close(
            torch.floor(torch.log2(route["target"][original_rr] - route["source"][original_rr])),
            torch.floor(torch.log2(control["target"][control_rr] - control["source"][control_rr])),
        )
        self.assertTrue(bool((control["source"][control_rr] < control["target"][control_rr]).all()))

    def test_randomized_topology_is_deterministic_and_leaves_direct_masses_unchanged(self):
        lookback = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1) / 4
        route = {
            "channel": torch.zeros(8, dtype=torch.long),
            "layer": torch.zeros(8, dtype=torch.long),
            "head": torch.zeros(8, dtype=torch.long),
            "source": torch.tensor([0, 0, 0, 0, 1, 2, 2, 4]),
            "target": torch.tensor([1, 2, 4, 5, 2, 4, 5, 5]),
            "weight": torch.tensor([.4, .3, .2, .1, .8, .6, .5, .7]),
            "attention_floor": .01,
        }
        direct, true_field, _ = evidence_flow_fields(
            lookback, route, prompt_count=1,
        )
        randomized_direct, first, _ = evidence_flow_fields(
            lookback, route, prompt_count=1,
            sample_id="random-null", seed=11, randomize_rr=True,
        )
        _, second, _ = evidence_flow_fields(
            lookback, route, prompt_count=1,
            sample_id="random-null", seed=11, randomize_rr=True,
        )
        torch.testing.assert_close(direct, randomized_direct)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(true_field, first))

    def test_anomaly_components_use_rr_connectivity_not_token_adjacency(self):
        route = {
            "source": np.asarray([2, 4, 3]),
            "target": np.asarray([4, 5, 5]),
        }
        active, component = anomaly_components(
            route, prompt_count=2,
            scores=np.asarray([.99, .98, .10, .97]), threshold=.95,
        )
        np.testing.assert_array_equal(active, [True, True, False, True])
        self.assertNotEqual(int(component[0]), int(component[1]))
        self.assertEqual(int(component[1]), int(component[3]))
        self.assertEqual(int(component[2]), -1)

    def test_streaming_anomaly_components_match_explicit_route(self):
        attention = AttentionSample(
            "components", "source", 2, torch.arange(6, dtype=torch.int32),
            torch.zeros((1, 1, 6), dtype=torch.float16),
            torch.tensor([0, 1, 3, 5, 7], dtype=torch.int32),
            torch.tensor([0, 0, 2, 0, 3, 0, 4], dtype=torch.int32),
            torch.full((7,), .2, dtype=torch.float16), .01,
        )
        scores = np.asarray([.99, .98, .10, .97], dtype=np.float32)
        route = {
            "source": np.asarray([0, 0, 2, 0, 3, 0, 4]),
            "target": np.asarray([2, 3, 3, 4, 4, 5, 5]),
        }

        expected = anomaly_components(
            route, prompt_count=2, scores=scores, threshold=.95
        )
        actual = anomaly_components_from_attention(
            attention, scores=scores, threshold=.95, csr_row_block=1
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
