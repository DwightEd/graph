import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.model import MaskedAttentionAutoencoder
from attention_graph.score import RobustResidualCalibrator
from attention_graph.visualize import EmbeddingShiftVisualizer
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from main import parse_args
from research_dataset import ResearchDataset


def _sample(sample_id, source_id):
    return AttentionSample(
        sample_id, source_id, 2, torch.tensor([10, 11, 12, 13, 14]),
        torch.tensor([[[0.1, 0.1, 0.2, 0.2, 0.2]]], dtype=torch.float16),
        torch.tensor([0, 2, 5, 9]),
        torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 3]),
        torch.tensor([0.3, 0.2, 0.2, 0.4, 0.1, 0.1, 0.2, 0.3, 0.2], dtype=torch.float16),
        0.01,
    )


def _split(root):
    attention = root / "attention"
    attention.mkdir(parents=True)
    rows, label_rows = [], []
    for domain in ("source", "target"):
        for index in range(2):
            sample = _sample(f"{domain}-{index}", f"{domain}-doc-{index}")
            path = attention / f"{sample.sample_id}.npz"
            save_attention_sample(sample, path)
            rows.append(index_row(root, sample, path, metadata={
                "split": "test", "data_source": domain, "task_type": "QA",
            }))
            label_rows.append({
                "sample_id": sample.sample_id,
                "positive_runs": [[1, 2]] if index == 1 else [],
            })
    labels = root / "labels.jsonl"
    labels.write_text("".join(f"{__import__('json').dumps(row)}\n" for row in label_rows), encoding="utf-8")
    write_split_index(
        root, rows, attention_floor=0.01, num_layers=1, num_heads=1,
        alignment="post_token_query_at_same_position",
        extra={"split": "test", "labels_sha256": sha256(labels)},
    )


def _checkpoint(path):
    torch.manual_seed(4)
    model = MaskedAttentionAutoencoder(
        num_channels=1, embedding_dim=6, message_steps=1, dropout=0.0,
    )
    checkpoint = {
        "schema": "attention-graph-unsupervised-v1",
        "model_config": {"num_channels": 1, "embedding_dim": 6, "message_steps": 1, "dropout": 0.0},
        "graph_config": asdict(GraphBuildConfig(selection="threshold")),
        "attention_geometry": {
            "num_layers": 1,
            "num_heads": 1,
            "alignment": "post_token_query_at_same_position",
            "attention_floor": 0.01,
            "observer_model": None,
            "generator_model": None,
        },
        "calibrator": RobustResidualCalibrator.fit(np.ones((2, 6), dtype=np.float32)).to_dict(),
        "state_dict": model.state_dict(),
        "best_epoch": 1,
    }
    torch.save(checkpoint, path)


class EmbeddingShiftTests(unittest.TestCase):
    def test_source_and_target_domains_must_differ(self):
        with self.assertRaisesRegex(ValueError, "different"):
            EmbeddingShiftVisualizer(
                mock.Mock(), checkpoint="model.pt", domain_field="data_source",
                source_domain="CNN/DM", target_domain="CNN/DM", output_dir="out",
            )

    def test_encoder_exposes_before_and_after_and_encode_returns_after(self):
        graph = build_attention_graph(_sample("case", "doc"))
        model = MaskedAttentionAutoencoder(
            num_channels=1, embedding_dim=6, message_steps=1, dropout=0.0,
        )
        before, after = model.encode_stages(graph)
        self.assertEqual(before.shape, (5, 6))
        self.assertEqual(after.shape, (5, 6))
        self.assertFalse(torch.equal(before, after))
        torch.testing.assert_close(model.encode(graph), after)

    def test_visualization_projects_paired_balanced_domains_before_opening_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            _split(root)
            checkpoint = Path(directory) / "model.pt"
            _checkpoint(checkpoint)
            dataset = ResearchDataset(root)
            projected = {"done": False}

            import attention_graph.visualize as visual

            original_projection = visual._joint_projection
            original_labels = dataset.labels

            def projection(*args, **kwargs):
                result = original_projection(*args, **kwargs)
                projected["done"] = True
                return result

            def labels():
                if not projected["done"]:
                    raise AssertionError("labels were opened before projection")
                return original_labels()

            with mock.patch.object(visual, "_joint_projection", side_effect=projection):
                with mock.patch.object(dataset, "labels", side_effect=labels):
                    result = EmbeddingShiftVisualizer(
                        dataset,
                        checkpoint=checkpoint,
                        domain_field="data_source",
                        source_domain="source",
                        target_domain="target",
                        output_dir=Path(directory) / "out",
                        device="cpu",
                        max_nodes_per_domain=4,
                        perplexity=2,
                        seed=3,
                    ).run()

            self.assertTrue(projected["done"])
            self.assertTrue(Path(result["figure"]).is_file())
            self.assertEqual(result["source_nodes"], result["target_nodes"])
            self.assertEqual(
                result["claim_scope"],
                "message-passing representation shift, not domain alignment",
            )
            self.assertEqual(
                result["claim_scope"], "message-passing representation shift, not domain alignment"
            )
            with np.load(result["data"], allow_pickle=False) as values:
                self.assertEqual(values["coordinates_before"].shape, (8, 2))
                self.assertEqual(values["coordinates_after"].shape, (8, 2))
                self.assertEqual(values["embedding_before"].shape, values["embedding_after"].shape)
                self.assertEqual(values["labels_read_during"].item(), "coloring_only")
                self.assertEqual(
                    values["claim_scope"].item(),
                    "message-passing representation shift, not domain alignment",
                )
                self.assertEqual(values["domain"].tolist().count(0), 4)
                self.assertEqual(values["domain"].tolist().count(1), 4)
                self.assertEqual(len(set(zip(values["sample_id"], values["token_index"]))), 8)
                self.assertEqual(values["claim_scope"].item(), result["claim_scope"])

            self.assertEqual(result["source_samples"], 2)
            self.assertEqual(result["target_samples"], 2)

    def test_visualization_rejects_checkpoint_without_exact_attention_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "split"
            _split(root)
            checkpoint = Path(directory) / "model.pt"
            _checkpoint(checkpoint)
            payload = torch.load(checkpoint, weights_only=True)
            del payload["attention_geometry"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "retrain"):
                EmbeddingShiftVisualizer(
                    ResearchDataset(root), checkpoint=checkpoint,
                    domain_field="data_source", source_domain="source",
                    target_domain="target", output_dir=Path(directory) / "out",
                    device="cpu", max_nodes_per_domain=4,
                ).run()

    def test_visualize_cli_uses_checkpoint_domains_and_has_no_scores_or_graph_view(self):
        args = parse_args([
            "visualize", "--canonical-split", "test", "--checkpoint", "model.pt",
            "--domain-field", "data_source", "--source-domain", "CNN/DM",
            "--target-domain", "Yelp", "--output-dir", "out",
        ])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_nodes_per_domain, 5000)
        self.assertFalse(hasattr(args, "scores"))
        with self.assertRaises(SystemExit):
            parse_args(["visualize-graph"])


if __name__ == "__main__":
    unittest.main()
