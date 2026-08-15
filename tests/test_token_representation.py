import unittest

import numpy as np
import torch

from attention_graph.token_representation import (
    compact_layer_structure,
    direct_lookback_channels,
    exact_channel_route,
    structure_names,
    _cluster_bootstrap_difference,
    _read_dataset_labels,
    _route_edges_by_relation,
)
from cache import AttentionSample
from main import _require_llama31_geometry, parse_args


def _sample(sample_id="sample", source_id="source"):
    # R0: P0 .2 + P1 .2 + self .1
    # R1: P0 .2 + R0 .6 + self .2
    diagonal = torch.zeros((1, 1, 4), dtype=torch.float16)
    diagonal[:, :, 2] = .1
    diagonal[:, :, 3] = .2
    return AttentionSample(
        sample_id, source_id, 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([.2, .2, .2, .6], dtype=torch.float16), .01,
    )


def _multi_channel_sample():
    # Row order is channel-major: C0R0,C0R1,C1R0,C1R1,...
    masses = torch.tensor(
        [.1, .2, .3, .4, .5, .6, .7, .8], dtype=torch.float16
    )
    diagonal = torch.zeros((2, 2, 4), dtype=torch.float16)
    diagonal[:, :, 2:] = .1
    return AttentionSample(
        "multi", "multi-source", 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.arange(9, dtype=torch.int32), torch.zeros(8, dtype=torch.int32),
        masses, .01,
    )


class TokenGraphRepresentationTests(unittest.TestCase):
    def test_undefined_lookback_channel_uses_attention_floor(self):
        attention = AttentionSample(
            "undefined-lookback", "source", 1,
            torch.arange(3, dtype=torch.int32),
            torch.zeros((2, 2, 3), dtype=torch.float16),
            torch.zeros(9, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float16), .01,
        )

        lookback = direct_lookback_channels(attention)

        self.assertEqual(tuple(lookback.shape), (2, 2, 2))
        torch.testing.assert_close(
            lookback, torch.full((2, 2, 2), attention.attention_floor)
        )

    def test_lookback_is_preserved_before_any_channel_average(self):
        values = direct_lookback_channels(_sample())
        self.assertEqual(tuple(values.shape), (2, 1, 1))
        torch.testing.assert_close(
            values[:, 0, 0], torch.tensor([2 / 3, .2]), atol=2e-3, rtol=2e-3
        )

    def test_multi_channel_order_matches_layer_head_geometry(self):
        sample = _multi_channel_sample()
        values = direct_lookback_channels(sample)
        masses = torch.tensor([
            [[.1, .3], [.5, .7]],
            [[.2, .4], [.6, .8]],
        ])
        expected = (masses / 2) / (masses / 2 + torch.tensor([.1, .05])[:, None, None])
        self.assertEqual(tuple(values.shape), (2, 2, 2))
        torch.testing.assert_close(values, expected, atol=2e-3, rtol=2e-3)

    def test_prompt_range_history_change_and_provenance_are_explicit(self):
        sample = _sample()
        names = structure_names(2)
        matrix = compact_layer_structure(sample, provenance_hops=2)
        self.assertEqual(tuple(matrix.shape), (2, len(names), 1))
        feature = {name: matrix[:, index, 0] for index, name in enumerate(names)}
        torch.testing.assert_close(
            feature["retained_prompt_coverage"], torch.tensor([1., .5]),
            atol=2e-3, rtol=2e-3,
        )
        torch.testing.assert_close(
            feature["retained_prompt_span"], torch.tensor([1., .5]),
            atol=2e-3, rtol=2e-3,
        )
        self.assertAlmostEqual(float(feature["history_lag"][1]), 1.0, places=3)
        self.assertAlmostEqual(
            float(feature["prompt_provenance_log_mass_hop1"][1]),
            float(np.log10(.6 * .4)), places=3,
        )
        self.assertEqual(float(feature["prompt_provenance_log_mass_hop2"][1]), -12.0)

    def test_top_head_route_is_compact_per_layer_not_all_head_average(self):
        names = structure_names(1)
        matrix = compact_layer_structure(
            _multi_channel_sample(), provenance_hops=1
        )
        prompt_mass = matrix[:, names.index("retained_prompt_mass")]
        torch.testing.assert_close(
            prompt_mass,
            torch.tensor([[.3, .7], [.4, .8]]),
            atol=2e-3, rtol=2e-3,
        )

    def test_strong_minority_head_is_not_diluted(self):
        diagonal = torch.zeros((1, 4, 2), dtype=torch.float16)
        sample = AttentionSample(
            "sparse-head", "source", 1, torch.arange(2, dtype=torch.int32),
            diagonal, torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([.8], dtype=torch.float16), .01,
        )
        names = structure_names(1)
        matrix, route = compact_layer_structure(
            sample, provenance_hops=1, return_route=True
        )
        self.assertAlmostEqual(
            float(matrix[0, names.index("retained_prompt_mass"), 0]), .8, places=3
        )
        self.assertAlmostEqual(float(route["weight"][0]), .8, places=3)

    def test_exact_channel_route_keeps_each_csr_edge_in_its_layer_head_channel(self):
        route = exact_channel_route(_multi_channel_sample())
        torch.testing.assert_close(
            route["channel"], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        )
        torch.testing.assert_close(
            route["layer"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        )
        torch.testing.assert_close(
            route["head"], torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        )
        torch.testing.assert_close(
            route["source"], torch.zeros(8, dtype=torch.long)
        )
        torch.testing.assert_close(
            route["target"], torch.tensor([2, 3, 2, 3, 2, 3, 2, 3])
        )
        torch.testing.assert_close(
            route["weight"], torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8]),
            atol=2e-3, rtol=2e-3,
        )

    def test_sparse_plot_route_preserves_source_target_distance(self):
        route = {
            "layer": np.asarray([0, 1, 0, 0]),
            "source": np.asarray([0, 0, 2, 3]),
            "target": np.asarray([3, 3, 4, 4]),
            "weight": np.asarray([.2, .7, .5, .4], dtype=np.float32),
        }
        edges = _route_edges_by_relation(route, response_idx=2, response_count=3)
        self.assertEqual(len(edges["selected"]), 3)
        np.testing.assert_array_equal(edges["rp_source"], [0])
        np.testing.assert_array_equal(edges["rp_target"], [1])
        np.testing.assert_allclose(edges["rp_weight"], [.7])
        np.testing.assert_array_equal(edges["rr_source"], [0, 1])
        np.testing.assert_array_equal(edges["rr_target"], [2, 2])
        np.testing.assert_allclose(edges["rr_weight"], [.5, .4])


class PipelineContractTests(unittest.TestCase):
    def test_frozen_cli_requires_llama31_8b_geometry_and_point01_floor(self):
        valid = type("Dataset", (), {
            "manifest": {"num_layers": 32, "num_heads": 32, "attention_floor": .01}
        })()
        invalid = type("Dataset", (), {
            "manifest": {"num_layers": 32, "num_heads": 32, "attention_floor": 0.0}
        })()

        _require_llama31_geometry(valid)
        with self.assertRaisesRegex(ValueError, "Llama-3.1-8B geometry"):
            _require_llama31_geometry(invalid)

    def test_paired_bootstrap_keeps_response_clusters_and_detects_gain(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        sample_ids = np.asarray(["a", "a", "b", "b", "c", "c"])
        better = np.asarray([.1, .9, .2, .8, .3, .7])
        worse = np.asarray([.9, .1, .8, .2, .7, .3])
        result = _cluster_bootstrap_difference(
            labels, better, worse, sample_ids, seed=4, replicates=20,
        )
        self.assertGreater(result["auroc_difference"]["ci95"][0], 0)
        self.assertGreater(result["auprc_difference"]["ci95"][0], 0)

    def test_saved_render_unlocks_a_formal_style_label_seal(self):
        class Sample:
            def __init__(self, dataset, sample_id):
                self.dataset, self.sample_id = dataset, sample_id

            def attention(self):
                self.dataset.processed.add(self.sample_id)

            def release_attention(self):
                pass

        class Store:
            def response_labels(self, sample):
                return torch.tensor([int(sample.sample_id == "b")])

        class Dataset:
            sample_ids = ["a", "b"]

            def __init__(self):
                self.processed = set()

            def __getitem__(self, sample_id):
                return Sample(self, sample_id)

            def labels(self):
                if len(self.processed) != len(self.sample_ids):
                    raise RuntimeError(
                        "formal labels become available only after every attention "
                        "sample has been processed"
                    )
                return Store()

        dataset = Dataset()
        labels = _read_dataset_labels(dataset, "test label seal")
        np.testing.assert_array_equal(labels, np.asarray([0, 1], dtype=np.int8))
        self.assertEqual(dataset.processed, {"a", "b"})

    def test_cli_exposes_only_causal_topology_experiment_controls(self):
        args = parse_args([
            "represent-tokens", "--train-split", "train", "--test-split", "test",
            "--output-dir", "output", "--fourier-frequencies", "6",
        ])
        self.assertEqual(args.position_bins, 10)
        self.assertEqual(args.bootstrap_replicates, 200)
        self.assertEqual(args.reference_size, 12_000)
        self.assertEqual(args.checkpoint_interval, 250)
        self.assertEqual(args.subspace_components, 32)
        self.assertEqual(args.tail_fraction, .05)
        self.assertEqual(args.fourier_frequencies, 6)
        self.assertEqual(args.row_block_size, 4096)
        self.assertEqual(args.seed, 42)
        for obsolete in (
            "provenance_hops", "csr_row_block", "sample_id",
            "display_mass_cover", "display_edges_per_type", "display_max_edges",
            "display_layer", "anomaly_quantile",
        ):
            self.assertFalse(hasattr(args, obsolete))


if __name__ == "__main__":
    unittest.main()
