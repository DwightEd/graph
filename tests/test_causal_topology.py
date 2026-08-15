import unittest

import torch

from attention_graph.causal_topology import CausalTopologyConfig, CausalTopologyEncoder
from cache import AttentionSample


def _sample(*, last_prompt_source=0, last_prompt_weight=.20):
    """One channel, three prompt tokens, and four causal response tokens."""
    columns = torch.tensor([
        0,       # t=0: prompt mass
        1, 3,    # t=1: a prompt edge and one RR edge
        2, 3, 4, # t=2: a prompt edge and two RR edges
        last_prompt_source, 3, 5,  # t=3: prompt edge and two RR edges
    ], dtype=torch.int32)
    values = torch.tensor([
        .30,
        .20, .50,
        .20, .25, .25,
        last_prompt_weight, .10, .40,
    ], dtype=torch.float16)
    return AttentionSample(
        "topology", "source", 3, torch.arange(7, dtype=torch.int32),
        torch.tensor([[[.05, .05, .05, .10, .10, .10, .10]]], dtype=torch.float16),
        torch.tensor([0, 1, 3, 6, 9], dtype=torch.int32),
        columns, values, .01,
    )


class CausalTopologyEncoderTests(unittest.TestCase):
    def setUp(self):
        self.encoder = CausalTopologyEncoder(
            CausalTopologyConfig(fourier_frequencies=3, rewire_seed=17)
        )

    def test_public_encoding_keeps_per_channel_balance_and_scale(self):
        encoding = self.encoder.encode(_sample())

        self.assertEqual(encoding.balance_log_scale.shape, (4, 1, 1, 2))
        self.assertEqual(encoding.attention_marginals.shape, (4, 1, 1, 3))
        self.assertEqual(encoding.retained_support.shape, (4, 1, 1, 2))
        self.assertEqual(encoding.prompt_provenance.shape, (4, 1, 1, 6))
        self.assertEqual(encoding.rr_one_hop.shape, (4, 1, 1, 6))
        self.assertEqual(encoding.rr_two_hop.shape, (4, 1, 1, 2))
        for value in vars(encoding).values():
            self.assertEqual(value.device, _sample().response_values.device)

        # Missing eligible off-diagonal edges are imputed at the .01 floor.
        # t=1 therefore has prompt mass .2 + 2*.01 and history mass .5.
        prompt_weight = float(torch.tensor(.2, dtype=torch.float16))
        response_weight = float(torch.tensor(.5, dtype=torch.float16))
        diagonal = float(torch.tensor(.1, dtype=torch.float16))
        floor = float(torch.tensor(.01, dtype=torch.float16))
        prompt_mass = prompt_weight + 2 * floor
        expected_total = (prompt_mass / 3) + (response_weight + diagonal) / 2
        expected_balance = (prompt_mass / 3) / expected_total
        self.assertAlmostEqual(float(encoding.balance_log_scale[1, 0, 0, 0]), expected_balance, places=6)
        self.assertAlmostEqual(
            float(encoding.balance_log_scale[1, 0, 0, 1]),
            float(torch.log(torch.tensor(expected_total))),
            places=6,
        )
        torch.testing.assert_close(
            encoding.attention_marginals[1, 0, 0],
            torch.tensor([prompt_mass, response_weight, diagonal]),
        )

    def test_prompt_provenance_distinguishes_equal_mass_at_different_sources(self):
        first = self.encoder.encode(_sample(last_prompt_source=0))
        second = self.encoder.encode(_sample(last_prompt_source=1))

        torch.testing.assert_close(first.balance_log_scale, second.balance_log_scale)
        self.assertFalse(torch.allclose(
            first.prompt_provenance[-1], second.prompt_provenance[-1]
        ))

    def test_prompt_provenance_includes_the_imputed_uniform_background(self):
        first = self.encoder.encode(_sample(last_prompt_source=1, last_prompt_weight=.20))
        second = self.encoder.encode(_sample(last_prompt_source=1, last_prompt_weight=.40))

        self.assertFalse(torch.allclose(
            first.prompt_provenance[-1], second.prompt_provenance[-1]
        ))
        self.assertNotEqual(
            float(first.attention_marginals[-1, 0, 0, 0]),
            float(second.attention_marginals[-1, 0, 0, 0]),
        )

    def test_absent_csr_edges_are_imputed_at_the_attention_floor(self):
        attention = AttentionSample(
            "censored", "source", 2, torch.arange(3, dtype=torch.int32),
            torch.tensor([[[.0, .0, .20]]], dtype=torch.float16),
            torch.tensor([0, 0], dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float16),
            .01,
        )

        encoding = self.encoder.encode(attention)

        torch.testing.assert_close(
            encoding.attention_marginals[0, 0, 0],
            torch.tensor([
                2 * float(torch.tensor(.01, dtype=torch.float16)),
                0.0,
                float(torch.tensor(.20, dtype=torch.float16)),
            ]),
        )
        self.assertGreater(float(encoding.balance_log_scale[0, 0, 0, 0]), 0.0)

    def test_floor_weight_edges_do_not_create_topology_messages(self):
        attention = AttentionSample(
            "floor-edge", "source", 1, torch.arange(3, dtype=torch.int32),
            torch.zeros((1, 1, 3), dtype=torch.float16),
            torch.tensor([0, 0, 1], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([.01], dtype=torch.float16), .01,
        )

        encoding = self.encoder.encode(attention)

        self.assertEqual(float(encoding.rr_one_hop.abs().sum()), 0.0)
        self.assertEqual(float(encoding.rr_two_hop.abs().sum()), 0.0)

    def test_retained_support_reports_prompt_and_history_coverage(self):
        encoding = self.encoder.encode(_sample())

        # t=2 retains one of three prompt endpoints and both causal response endpoints.
        torch.testing.assert_close(
            encoding.retained_support[2, 0, 0], torch.tensor([1 / 3, 1.0])
        )
        torch.testing.assert_close(
            encoding.retained_support[0, 0, 0], torch.tensor([1 / 3, 0.0])
        )

    def test_layer_head_channels_remain_separate(self):
        attention = AttentionSample(
            "channels", "source", 1, torch.arange(3, dtype=torch.int32),
            torch.zeros((1, 2, 3), dtype=torch.float16),
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            torch.tensor([0, 0, 0, 1], dtype=torch.int32),
            torch.tensor([.2, .4, .6, .7], dtype=torch.float16),
            .01,
        )

        encoding = self.encoder.encode(attention)

        self.assertEqual(encoding.attention_marginals.shape, (2, 1, 2, 3))
        torch.testing.assert_close(
            encoding.attention_marginals[:, 0, :, :2],
            torch.tensor(
                [[[.2, 0.0], [.6, 0.0]], [[.4, .01], [.01, .7]]],
                dtype=torch.float16,
            ).float(),
        )

    def test_rr_features_keep_absolute_difference_and_neighborhood_variance(self):
        encoding = self.encoder.encode(_sample())
        one_hop = encoding.rr_one_hop[:, 0, 0]

        # t=2 attends to two different response states; dispersion must survive.
        self.assertGreater(float(one_hop[2, 2:4].abs().sum()), 0.0)
        self.assertGreater(float(one_hop[2, 4:6].sum()), 0.0)
        self.assertEqual(float(one_hop[0].abs().sum()), 0.0)

    def test_two_hop_paths_and_lag_rewired_null_are_distinct_and_deterministic(self):
        first = self.encoder.encode(_sample())
        second = self.encoder.encode(_sample())

        torch.testing.assert_close(first.rewired_rr_one_hop, second.rewired_rr_one_hop)
        torch.testing.assert_close(first.rewired_rr_two_hop, second.rewired_rr_two_hop)
        self.assertGreater(float(first.rr_two_hop[3].abs().sum()), 0.0)
        self.assertFalse(torch.allclose(first.rr_one_hop, first.rewired_rr_one_hop))


if __name__ == "__main__":
    unittest.main()
