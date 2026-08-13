import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, RP, RR, build_attention_graph
from attention_graph.statistics import (
    TOKEN_FEATURES,
    direct_lookback,
    direct_lookback_from_graph,
    token_statistics,
)
from attention_graph.token_representation import (
    EXACT_FEATURES,
    SCORE_FEATURES,
    TokenRepresentationConfig,
    _PositivePathReliability,
    _build_scores,
    _directed_score,
    _feature_separation,
    _without_relation,
    exact_token_features,
    structure_preserving_messages,
    discover_token_representations,
)
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from main import parse_args
from research_dataset import ResearchDataset


def _sample(sample_id="sample", source_id="source"):
    # R0: prompt .2+.2, diagonal .1
    # R1: prompt .2, R0 .6, diagonal .2
    diagonal = torch.zeros((1, 1, 4), dtype=torch.float16)
    diagonal[:, :, 2] = .1
    diagonal[:, :, 3] = .2
    return AttentionSample(
        sample_id, source_id, 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([.2, .2, .2, .6], dtype=torch.float16), .01,
    )


def _chain_sample():
    # P0 -> R0 (.6), R0 -> R1 (.5), R1 -> R2 (.4).
    return AttentionSample(
        "chain", "chain-source", 1, torch.arange(4, dtype=torch.int32),
        torch.zeros((1, 1, 4), dtype=torch.float16),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([.6, .5, .4], dtype=torch.float16), .01,
    )


def _multi_channel_sample():
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


def _varied_sample(sample_id, source_id, variant):
    prompt_source = variant % 3
    prompt = .14 + .015 * variant
    history = .22 + .012 * variant
    diagonal = torch.zeros((1, 1, 6), dtype=torch.float16)
    diagonal[:, :, 3:] = torch.tensor(
        [.08 + .005 * variant, .10 + .004 * variant, .12 + .003 * variant],
        dtype=torch.float16,
    )
    return AttentionSample(
        sample_id, source_id, 3, torch.arange(6, dtype=torch.int32), diagonal,
        torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        torch.tensor([prompt_source, prompt_source, 3, prompt_source, 3, 4], dtype=torch.int32),
        torch.tensor([prompt, prompt + .03, history, prompt + .05,
                      history - .04, history + .02], dtype=torch.float16), .01,
    )


def _write_split(root, split, count, source_prefix):
    (root / "attention").mkdir(parents=True)
    rows, labels = [], []
    for index in range(count):
        sample = _varied_sample(
            f"{split}-{index}", f"{source_prefix}-{index}",
            index + (0 if split == "train" else 7),
        )
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata={
            "split": split, "task_type": "QA" if index % 2 else "Summary",
            "data_source": "MARCO" if index % 2 else "CNN/DM",
            "generator_model": "generator",
        }))
        labels.append({
            "sample_id": sample.sample_id,
            "positive_runs": [[1, 2]] if index % 2 else [],
        })
    label_path = root / "labels.jsonl"
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    write_split_index(
        root, rows, attention_floor=.01, num_layers=1, num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={
            "schema": "ragtruth-attention-split-v1", "split": split,
            "observer_model": "observer", "generator_model": "generator",
            "labels_sha256": sha256(label_path),
        },
    )


class ExactScalarContractTests(unittest.TestCase):
    def test_exact_scalar_fixture_and_mean_edge_strength(self):
        graph = build_attention_graph(_sample(), GraphBuildConfig())
        values = token_statistics(graph)
        index = {name: TOKEN_FEATURES.index(name) for name in TOKEN_FEATURES}
        self.assertEqual(values.shape, (2, len(TOKEN_FEATURES)))
        expected_first = {
            "retained_mass": .4, "prompt_mass_fraction": 1.0,
            "history_mass_fraction": 0.0, "normalized_entropy": 1.0,
            "top1_share": .5, "retained_concentration": .5,
            "in_degree": 2.0, "edge_density": 1.0,
            "history_edge_fraction": 0.0, "history_lag": 0.0,
            "channel_coverage": 1.0, "mean_edge_strength": .2,
        }
        for name, expected in expected_first.items():
            self.assertAlmostEqual(float(values[0, index[name]]), expected, places=3)
        self.assertAlmostEqual(float(values[1, index["retained_mass"]]), .8, places=3)
        self.assertAlmostEqual(float(values[1, index["prompt_mass_fraction"]]), .25, places=3)
        self.assertAlmostEqual(float(values[1, index["top1_share"]]), .75, places=3)
        self.assertAlmostEqual(float(values[1, index["retained_concentration"]]), .625, places=3)
        self.assertAlmostEqual(float(values[1, index["edge_density"]]), 2 / 3, places=3)
        self.assertAlmostEqual(float(values[1, index["mean_edge_strength"]]), .4, places=3)

    def test_direct_lookback_parity_is_not_hidden_in_projection(self):
        lookback = direct_lookback(_sample())
        torch.testing.assert_close(
            lookback, torch.tensor([1 / 3, .8]), atol=2e-3, rtol=2e-3
        )
        exact = exact_token_features(_sample())
        self.assertEqual(exact.shape, (2, len(EXACT_FEATURES)))
        torch.testing.assert_close(
            exact[:, EXACT_FEATURES.index("direct_lookback_anomaly")], lookback
        )

    def test_direct_lookback_preserves_layer_head_row_order(self):
        sample = _multi_channel_sample()
        observed = direct_lookback(sample)
        masses = torch.tensor([
            [[.1, .3], [.5, .7]],
            [[.2, .4], [.6, .8]],
        ])
        prompt_mean = masses / 2
        generated_mean = torch.tensor([.1, .05])[:, None, None]
        expected = 1.0 - (
            prompt_mean / (prompt_mean + generated_mean)
        ).mean((1, 2))
        torch.testing.assert_close(observed, expected, atol=2e-3, rtol=2e-3)

    def test_raw_auc_and_reverse_separability_are_distinct(self):
        report = _feature_separation(
            np.asarray([0, 0, 1, 1]), np.asarray([4., 3., 2., 1.])
        )
        self.assertEqual(report["raw_auroc_higher_is_anomaly"], 0.0)
        self.assertEqual(report["separability"], 1.0)
        self.assertEqual(report["post_hoc_direction"], "lower_for_hallucination")


class PropagationContractTests(unittest.TestCase):
    class _CenterScaler:
        def __init__(self, center):
            self.center = float(center)

        def transform(self, values, _position):
            return np.asarray(values, dtype=np.float32) - self.center

    class _RatioReliability:
        def __init__(self, reference):
            self.reference = float(reference)

        def transform(self, mass, _position, eligible):
            mass = np.asarray(mass, dtype=np.float32)
            return np.where(
                eligible, mass / (mass + self.reference), 0.0
            ).astype(np.float32)

    @staticmethod
    def _counterfactual_exact(sample, graph, relation):
        counterfactual = _without_relation(graph, relation)
        return torch.cat((
            token_statistics(counterfactual),
            direct_lookback_from_graph(counterfactual)[:, None],
        ), dim=1).cpu().numpy()

    def _scores(self, sample, graph, *, rr_center=np.log1p(.6)):
        exact = exact_token_features(sample, graph).cpu().numpy()
        base_z = exact.astype(np.float32)
        no_rp_exact = self._counterfactual_exact(sample, graph, RP)
        no_rr_exact = self._counterfactual_exact(sample, graph, RR)
        messages = {
            name: value.cpu().numpy()
            for name, value in structure_preserving_messages(
                graph, base_z, diffusion_hops=2
            ).items()
        }
        no_rp_graph = _without_relation(graph, RP)
        messages["no_rp_self_neighbor_residual"] = (
            structure_preserving_messages(
                no_rp_graph, no_rp_exact, diffusion_hops=2
            )["self_neighbor_residual"].cpu().numpy()
        )
        rp_eligible = np.asarray([[True, False, False], [True, True, False]])
        rr_eligible = rp_eligible[:, 1:]
        return _build_scores(
            base_z, exact,
            no_rp_exact, no_rp_exact,
            no_rr_exact, no_rr_exact,
            np.asarray([0., 1.], dtype=np.float32),
            rp_eligible, rr_eligible, messages,
            self._CenterScaler(0.0), self._CenterScaler(rr_center),
            self._RatioReliability(.6),
        )

    def test_mass_difference_survives_when_conditional_mean_is_equal(self):
        graph = build_attention_graph(_sample())
        base = torch.arange(2 * len(EXACT_FEATURES), dtype=torch.float32).reshape(2, -1)
        full = structure_preserving_messages(graph, base, diffusion_hops=2)
        scores = graph.edge_score.clone()
        scores[graph.edge_type == RR] *= .1
        weak = structure_preserving_messages(
            replace(graph, edge_score=scores), base, diffusion_hops=2
        )
        torch.testing.assert_close(
            full["conditional_neighbor"][1, 0], weak["conditional_neighbor"][1, 0]
        )
        self.assertAlmostEqual(
            float(full["rr_path_mass"][1, 0]),
            10 * float(weak["rr_path_mass"][1, 0]), places=4,
        )
        self.assertFalse(torch.equal(full["raw_message"], weak["raw_message"]))

    def test_two_hop_rr_and_prompt_inheritance_are_exact(self):
        graph = build_attention_graph(_chain_sample())
        base = torch.ones((3, len(EXACT_FEATURES)))
        messages = structure_preserving_messages(graph, base, diffusion_hops=2)
        self.assertAlmostEqual(float(messages["rr_path_mass"][2, 1]), .2, places=3)
        self.assertAlmostEqual(float(messages["rp_path_mass"][2, 2]), .12, places=3)
        self.assertEqual(messages["rr_hop_reach_count"].tolist(), [[0, 0], [1, 0], [1, 1]])

    def test_no_history_does_not_turn_zero_lag_into_anomaly(self):
        z = np.zeros((2, len(EXACT_FEATURES)), dtype=np.float32)
        z[:, EXACT_FEATURES.index("history_lag")] = -100
        score = _directed_score(z, np.asarray([False, True]))
        self.assertEqual(float(score[0]), 0.0)
        self.assertGreater(float(score[1]), 0.0)

    def test_relation_counterfactual_recomputes_exact_graph_features(self):
        graph = build_attention_graph(_sample())
        no_rp = _without_relation(graph, RP)
        no_rr = _without_relation(graph, RR)
        self.assertTrue(bool((no_rp.edge_type == RR).all()))
        self.assertTrue(bool((no_rr.edge_type == RP).all()))
        density = TOKEN_FEATURES.index("edge_density")
        self.assertEqual(float(token_statistics(no_rp)[0, density]), 0.0)
        self.assertLess(
            float(token_statistics(no_rr)[1, density]),
            float(token_statistics(graph)[1, density]),
        )
        self.assertTrue(bool((direct_lookback_from_graph(no_rp) == 1).all()))
        torch.testing.assert_close(
            direct_lookback_from_graph(graph), direct_lookback(_sample()),
            atol=2e-3, rtol=2e-3,
        )
        for counterfactual in (no_rp, no_rr):
            if counterfactual.trace_edge_id.numel():
                self.assertLess(
                    int(counterfactual.trace_edge_id.max()), counterfactual.num_edges
                )

    def test_rr_mass_changes_reliability_and_ineligible_hops_are_zero(self):
        sample = _sample()
        graph = build_attention_graph(sample)
        full = self._scores(sample, graph)
        edge_score = graph.edge_score.clone()
        edge_score[graph.edge_type == RR] *= .1
        weak_graph = replace(graph, edge_score=edge_score)
        weak = self._scores(sample, weak_graph)
        self.assertEqual(float(full["rr_path_deficit"][0]), 0.0)
        self.assertEqual(float(weak["rr_path_deficit"][0]), 0.0)
        self.assertGreater(
            float(weak["rr_path_deficit"][1]),
            float(full["rr_path_deficit"][1]),
        )
        self.assertLess(
            float(weak["innovation_reliability"][1, 0]),
            float(full["innovation_reliability"][1, 0]),
        )
        self.assertAlmostEqual(
            float(weak["no_rr"][1]), float(full["no_rr"][1]), places=6
        )

    def test_missing_but_eligible_path_is_deficit_not_structural_mask(self):
        sample = _sample()
        graph = _without_relation(build_attention_graph(sample), RR)
        scores = self._scores(sample, graph)
        self.assertEqual(float(scores["rr_path_deficit"][0]), 0.0)
        self.assertGreater(float(scores["rr_path_deficit"][1]), 0.0)

    def test_path_reliability_is_continuous_at_zero_and_monotone(self):
        mass = np.asarray([[0.], [1e-12], [1.], [100.]], dtype=np.float64)
        position = np.asarray([0., .25, .5, .75])
        eligible = np.ones_like(mass, dtype=bool)
        model = _PositivePathReliability(1).fit(
            np.asarray([[1.], [1.], [1.]]), np.asarray([0., .5, 1.]),
            np.ones((3, 1), dtype=bool),
        )
        observed = model.transform(mass, position, eligible)[:, 0]
        self.assertEqual(float(observed[0]), 0.0)
        self.assertLess(float(observed[1]), 1e-8)
        self.assertTrue(bool(np.all(np.diff(observed) > 0)))
        self.assertAlmostEqual(float(observed[2]), .5, places=6)
        self.assertGreater(float(observed[3]), .99)

    def test_relation_ablations_are_invariant_to_the_deleted_relation(self):
        sample = _sample()
        graph = build_attention_graph(sample)
        reference = self._scores(sample, graph)
        edge_score = graph.edge_score.clone()
        edge_score[graph.edge_type == RP] *= .05
        changed_rp = self._scores(sample, replace(graph, edge_score=edge_score))
        np.testing.assert_allclose(reference["no_rp"], changed_rp["no_rp"], atol=1e-6)


class PipelineContractTests(unittest.TestCase):
    def test_cli_has_only_current_structure_preserving_arguments(self):
        args = parse_args([
            "represent-tokens", "--train-split", "train", "--test-split", "test",
            "--output-dir", "output", "--sample-id", "42",
        ])
        self.assertEqual(args.command, "represent-tokens")
        self.assertEqual(args.position_bins, 10)
        self.assertEqual(args.diffusion_hops, 2)
        self.assertEqual(args.visual_reference_size, 30_000)
        self.assertFalse(hasattr(args, "prototypes"))

    def test_end_to_end_saves_exact_base_and_mass_preserving_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _write_split(train_root, "train", 5, "train-source")
            _write_split(test_root, "test", 4, "test-source")
            output = root / "output"
            result = discover_token_representations(
                ResearchDataset(train_root), ResearchDataset(test_root),
                ResearchDataset(test_root), output_dir=output,
                config=TokenRepresentationConfig(
                    position_bins=2, diffusion_hops=2,
                    sample_ids=("test-1",), seed=7,
                ),
            )
            self.assertEqual(result["test_nodes"], 12)
            self.assertEqual(len(list((output / "sample_graphs").glob("*.npz"))), 4)
            with np.load(output / "sample_graphs" / "sample_test-1.npz", allow_pickle=False) as sample:
                self.assertFalse(bool(sample["labels_included"]))
                self.assertEqual(sample["exact_token_features"].shape, (3, len(EXACT_FEATURES)))
                self.assertEqual(sample["rr_path_mass"].shape, (3, 2))
                self.assertEqual(sample["rp_path_mass"].shape, (3, 3))
                self.assertIn("self_neighbor_residual", sample.files)
                self.assertIn("no_rp_exact_token_features", sample.files)
                self.assertIn("mechanism_coordinates", sample.files)
            with np.load(output / "token_representations_label_free.npz", allow_pickle=False) as artifact:
                self.assertFalse(bool(artifact["labels_included"]))
                self.assertEqual(artifact["exact_token_features"].shape, (12, len(EXACT_FEATURES)))
                self.assertIn("token_graph_representation", artifact.files)
                self.assertNotIn("label", artifact.files)
            report = json.loads((output / "token_representation_report.json").read_text())
            self.assertTrue(report["exact_scalar_block_recoverable_without_projection"])
            self.assertFalse(report["graph_propagation"]["rr_matrix_row_normalized"])
            self.assertEqual(report["labels_read_during"], "evaluation_and_plot_coloring_only")
            self.assertEqual(set(report["views"]), set(("token_only", "token_graph", "no_rp", "no_rr")))


if __name__ == "__main__":
    unittest.main()
