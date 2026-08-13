import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.token_representation import (
    MECHANISMS,
    TokenRepresentationConfig,
    _PrototypeDetector,
    _RobustProjector,
    _view_matrix,
    fixed_graph_messages,
    mechanism_tensor,
    discover_token_representations,
)
from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from main import parse_args
from research_dataset import ResearchDataset


def _sample():
    # R0: P0=.2, P1=.2, diagonal=.1
    # R1: P0=.2, R0=.6, diagonal=.2
    diagonal = torch.zeros((1, 1, 4), dtype=torch.float16)
    diagonal[:, :, 2] = .1
    diagonal[:, :, 3] = .2
    return AttentionSample(
        "sample",
        "source",
        2,
        torch.arange(4, dtype=torch.int32),
        diagonal,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([.2, .2, .2, .6], dtype=torch.float16),
        .01,
    )


def _multi_channel_sample():
    # CSR row order is layer, head, response.  Every row uses prompt source 0
    # and a fixed diagonal so routing values must remain channel-specific.
    masses = torch.tensor(
        [.1, .2, .3, .4, .5, .6, .7, .8], dtype=torch.float16
    )
    diagonal = torch.zeros((2, 2, 4), dtype=torch.float16)
    diagonal[:, :, 2:] = .1
    return AttentionSample(
        "multi",
        "multi-source",
        2,
        torch.arange(4, dtype=torch.int32),
        diagonal,
        torch.arange(9, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
        masses,
        .01,
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
    columns = [prompt_source, prompt_source, 3, prompt_source, 3, 4]
    values = [prompt, prompt + .03, history, prompt + .05, history - .04, history + .02]
    return AttentionSample(
        sample_id, source_id, 3, torch.arange(6, dtype=torch.int32), diagonal,
        torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        torch.tensor(columns, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float16), .01,
    )


def _write_split(root, split, count, source_prefix):
    (root / "attention").mkdir(parents=True)
    rows, labels = [], []
    for index in range(count):
        sample = _varied_sample(
            f"{split}-{index}", f"{source_prefix}-{index}", index + (0 if split == "train" else 7)
        )
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata={
            "split": split,
            "task_type": "QA" if index % 2 else "Summary",
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
            "schema": "ragtruth-attention-split-v1",
            "split": split,
            "observer_model": "observer",
            "generator_model": "generator",
            "labels_sha256": sha256(label_path),
        },
    )


class MechanismTensorTests(unittest.TestCase):
    def test_exact_four_mechanisms_and_unresolved_control(self):
        values, unresolved = mechanism_tensor(_sample(), csr_row_block=1)
        self.assertEqual(MECHANISMS, (
            "routing_balance", "effective_support_fraction",
            "dominant_edge_strength", "response_locality",
        ))
        self.assertEqual(tuple(values.shape), (2, 1, 1, 4))
        expected = torch.tensor([
            [2 / 3, (1 / .36) / 3, .2, 1.0],
            [.2, (1 / .44) / 4, .6, 1.0],
        ])
        torch.testing.assert_close(
            values[:, 0, 0], expected, atol=2e-3, rtol=2e-3
        )
        torch.testing.assert_close(
            unresolved[:, 0, 0], torch.tensor([.5, 0.0]),
            atol=2e-3, rtol=2e-3,
        )

    def test_layer_head_order_is_preserved_without_averaging(self):
        values, _ = mechanism_tensor(_multi_channel_sample())
        self.assertEqual(tuple(values.shape), (2, 2, 2, 4))
        observed = values[..., 0]
        masses = torch.tensor([
            [[.1, .3], [.5, .7]],
            [[.2, .4], [.6, .8]],
        ])
        prompt_mean = masses / 2
        response_mean = torch.tensor([.1, .05])[:, None, None]
        expected = prompt_mean / (prompt_mean + response_mean)
        torch.testing.assert_close(observed, expected, atol=2e-3, rtol=2e-3)


class FixedGraphRepresentationTests(unittest.TestCase):
    def test_graph_messages_keep_token_embedding_and_add_exact_endpoints(self):
        graph = build_attention_graph(_sample(), GraphBuildConfig())
        token = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        fused, slices, structure = fixed_graph_messages(
            graph, token, source_sketch_dim=4, diffusion_hops=2
        )
        self.assertEqual(tuple(fused.shape), (2, 26))
        start, end = slices["token"]
        torch.testing.assert_close(fused[:, start:end], token)
        rr_start, rr_end = slices["rr_token_hop_1"]
        torch.testing.assert_close(
            fused[1, rr_start:rr_end], token[0], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            fused[0, rr_start:rr_end], torch.zeros(2), atol=1e-6, rtol=1e-6
        )
        self.assertEqual(structure["hop_reach_count"].tolist(), [[0, 0], [1, 0]])

    def test_no_rp_and_no_rr_are_block_ablation_not_new_features(self):
        graph = build_attention_graph(_sample(), GraphBuildConfig())
        fused, slices, _ = fixed_graph_messages(
            graph, torch.eye(2), source_sketch_dim=4, diffusion_hops=2
        )
        values = fused.numpy()
        no_rp = _view_matrix(values, slices, "no_rp")
        no_rr = _view_matrix(values, slices, "no_rr")
        rp = slice(*slices["rp_direct"])
        self.assertTrue(np.all(no_rp[:, rp] == 0))
        self.assertTrue(np.any(values[:, rp] != 0))
        for name, bounds in slices.items():
            if name.startswith("rp_diffusion_hop_"):
                self.assertTrue(np.all(no_rp[:, slice(*bounds)] == 0))
        for name, bounds in slices.items():
            if name.startswith("rr_") or name.startswith("rp_diffusion_hop_"):
                block = slice(*bounds)
                self.assertTrue(np.all(no_rr[:, block] == 0))
        token = slice(*slices["token"])
        np.testing.assert_array_equal(no_rp[:, token], values[:, token])
        np.testing.assert_array_equal(no_rr[:, token], values[:, token])


class UnsupervisedProjectionTests(unittest.TestCase):
    def test_projector_uses_float64_and_drops_constant_dimensions(self):
        rng = np.random.default_rng(3)
        reference = np.column_stack((
            rng.normal(size=100), rng.normal(size=100), np.ones(100)
        )).astype(np.float32)
        projector = _RobustProjector(2, seed=3).fit(reference)
        output = projector.transform(reference)
        self.assertEqual(output.shape, (100, 2))
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(projector.report()["active_dimensions"], 2)
        shaped = np.column_stack((reference[:, :2], reference[:, :2]))
        structured = _RobustProjector(2, seed=3).fit(shaped)
        loading = structured.structured_loading_report(1, 2, 2)
        self.assertAlmostEqual(sum(loading["mechanism_fraction"]), 1.0)
        self.assertAlmostEqual(sum(loading["head_fraction"]), 1.0)

    def test_prototype_detector_survives_duplicate_rows(self):
        train = np.repeat(
            np.asarray([[0., 0.], [1., 1.]], dtype=np.float32), 20, axis=0
        )
        detector = _PrototypeDetector(256, 100, seed=4).fit(train)
        score = detector.score(np.asarray([[0., 0.], [2., 2.]], dtype=np.float32))
        self.assertEqual(score.shape, (2,))
        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(detector.model.n_clusters, 2)

    def test_cli_has_one_current_representation_entrypoint(self):
        args = parse_args([
            "represent-tokens", "--train-split", "train",
            "--test-split", "test", "--output-dir", "output",
            "--sample-id", "42",
        ])
        self.assertEqual(args.command, "represent-tokens")
        self.assertEqual(args.base_dim, 32)
        self.assertEqual(args.diffusion_hops, 3)
        self.assertEqual(args.sample_id, ["42"])
        with self.assertRaises(SystemExit):
            parse_args([
                "discover-patterns", "--train-split", "train",
                "--test-split", "test", "--output-dir", "output",
            ])

    def test_config_rejects_odd_source_sketch(self):
        with self.assertRaises(ValueError):
            TokenRepresentationConfig(source_sketch_dim=3).validate()

    def test_end_to_end_writes_every_sample_before_label_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_root, test_root = root / "train", root / "test"
            _write_split(train_root, "train", 5, "train-source")
            _write_split(test_root, "test", 4, "test-source")
            output = root / "output"
            result = discover_token_representations(
                ResearchDataset(train_root),
                ResearchDataset(test_root),
                ResearchDataset(test_root),
                output_dir=output,
                config=TokenRepresentationConfig(
                    base_dim=2,
                    embedding_dim=2,
                    source_sketch_dim=4,
                    fit_reference_size=20,
                    detector_reference_size=20,
                    prototypes=2,
                    diffusion_hops=2,
                    sample_ids=("test-1",),
                    seed=7,
                ),
            )
            self.assertEqual(result["test_nodes"], 12)
            self.assertEqual(len(list((output / "sample_graphs").glob("*.npz"))), 4)
            with np.load(output / "sample_graphs" / "sample_test-1.npz", allow_pickle=False) as sample_artifact:
                self.assertEqual(sample_artifact["rr_hop_reach_count"].shape, (3, 2))
                self.assertIn("rr_token_hop_2", sample_artifact["feature_block_name"].tolist())
                self.assertEqual(sample_artifact["multiscale_graph_features"].shape[0], 3)
            with np.load(output / "token_representations_label_free.npz", allow_pickle=False) as artifact:
                self.assertFalse(bool(artifact["labels_included"]))
                self.assertEqual(artifact["token_graph_embedding"].shape, (12, 2))
                self.assertNotIn("label", artifact.files)
            report = json.loads((output / "token_representation_report.json").read_text())
            self.assertEqual(report["labels_read_during"], "evaluation_and_plot_coloring_only")
            self.assertEqual(set(report["views"]), set(("token_only", "token_graph", "no_rp", "no_rr")))
            self.assertIn("full_minus_no_rr", report["relation_ablation"])
            self.assertEqual(report["sample_visualizations"][0]["selection_rule"], "user_requested_before_labels")


if __name__ == "__main__":
    unittest.main()
