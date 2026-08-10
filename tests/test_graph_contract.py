import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from build import BuildConfig, GraphDatasetBuilder
from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from graphs import build_original_graph, build_relation_topk_graph
from hypergraph import build_attention_hypergraph


def attention_sample() -> AttentionSample:
    return AttentionSample(
        "r1", "s1", 2,
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[[1.0, 0.5, 0.2, 0.1]]], dtype=torch.float16),
        torch.tensor([0, 2, 4]),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([0.02, 0.08, 0.10, 0.04], dtype=torch.float16),
        0.01,
    )


def write_split(root: Path) -> Path:
    sample = attention_sample()
    path = root / "attention" / "r1.npz"
    save_attention_sample(sample, path)
    write_split_index(
        root, [index_row(root, sample, path)], attention_floor=0.01,
        num_layers=1, num_heads=1, alignment="post_token_query_at_same_position",
    )
    return root


class GraphContractTests(unittest.TestCase):
    def test_all_graph_kinds_persist_topology_only_with_a_bound_manifest(self):
        expected_keys = {
            "original": {"num_nodes", "response_idx", "edge_index", "edge_type", "edge_ptr", "edge_channel", "edge_value"},
            "relation_topk": {"num_nodes", "response_idx", "edge_index", "edge_type", "edge_weight"},
            "relation_topk_channels": {"num_nodes", "response_idx", "edge_index", "edge_type", "edge_weight", "edge_ptr", "edge_channel", "edge_value"},
            "hypergraph": {"num_nodes", "response_idx", "incidence_index", "incidence_weight", "hyperedge_target", "hyperedge_channel", "hyperedge_type"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = write_split(root / "attention")
            for kind, keys in expected_keys.items():
                with self.subTest(kind=kind):
                    output = root / kind
                    GraphDatasetBuilder(BuildConfig(cache, output, kind=kind, tau=0.05, k_prompt=1, k_history=1, device="cpu")).run()
                    graph = torch.load(output / "graphs" / "r1.pt", weights_only=True)
                    self.assertEqual(set(graph), keys)
                    self.assertFalse({"x", "token_ids", "label", "y", "y_token"}.intersection(graph))

                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(set(manifest), {
                        "schema", "representation", "kind", "count", "attention_floor", "num_layers",
                        "num_heads", "alignment", "input_manifest_sha256", "input_index_sha256", "index_sha256", "parameters",
                    })
                    self.assertEqual(manifest["kind"], kind)
                    self.assertEqual(manifest["count"], 1)
                    self.assertEqual(manifest["attention_floor"], 0.01)
                    self.assertEqual(manifest["num_layers"], 1)
                    self.assertEqual(manifest["num_heads"], 1)
                    self.assertEqual(manifest["alignment"], "post_token_query_at_same_position")
                    self.assertEqual(manifest["input_manifest_sha256"], sha256(cache / "manifest.json"))
                    self.assertEqual(manifest["input_index_sha256"], sha256(cache / "index.jsonl"))
                    self.assertEqual(manifest["index_sha256"], sha256(output / "index.jsonl"))
                    self.assertEqual(manifest["parameters"], {"tau": 0.05} if kind in ("original", "hypergraph") else {"k_prompt": 1, "k_history": 1})

                    row = json.loads((output / "index.jsonl").read_text())
                    size_key = "num_hyperedges" if kind == "hypergraph" else "num_edges"
                    self.assertEqual(set(row), {"sample_id", "source_id", "path", "num_nodes", size_key, "sha256", "bytes"})
                    artifact = output / row["path"]
                    self.assertEqual(row["sha256"], sha256(artifact))
                    self.assertEqual(row["bytes"], artifact.stat().st_size)

    def test_build_configuration_has_no_node_feature_option(self):
        self.assertNotIn("node_features", BuildConfig.__dataclass_fields__)

    def test_invalid_graph_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = write_split(root / "attention")
            for tau in (math.nan, math.inf, -math.inf, 0.009, 1.01):
                with self.subTest(tau=tau), self.assertRaises(ValueError):
                    GraphDatasetBuilder(BuildConfig(cache, root / f"tau-{tau}", kind="original", tau=tau, device="cpu")).run()
            for k_prompt, k_history in ((-1, 1), (1, -1)):
                with self.subTest(k_prompt=k_prompt, k_history=k_history), self.assertRaises(ValueError):
                    GraphDatasetBuilder(BuildConfig(cache, root / f"k-{k_prompt}-{k_history}", kind="relation_topk", k_prompt=k_prompt, k_history=k_history, device="cpu")).run()
            for limit in (0, -1, True, 1.5):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    GraphDatasetBuilder(BuildConfig(cache, root / f"limit-{limit}", limit=limit, device="cpu")).run()

    def test_direct_tau_builders_reject_nonfinite_and_out_of_range_values(self):
        sample = attention_sample()
        builders = (build_original_graph, build_attention_hypergraph)

        for tau in (math.nan, math.inf, -math.inf, 0.009, 1.01, True, "0.05"):
            for builder in builders:
                with self.subTest(builder=builder.__name__, tau=tau), self.assertRaises(ValueError):
                    builder(sample, tau=tau)

    def test_direct_tau_builders_accept_attention_floor_and_one(self):
        sample = attention_sample()

        for tau in (sample.attention_floor, 1.0):
            with self.subTest(tau=tau):
                self.assertIsNotNone(build_original_graph(sample, tau=tau))
                self.assertIsNotNone(build_attention_hypergraph(sample, tau=tau))

    def test_direct_relation_topk_builder_rejects_invalid_limits(self):
        sample = attention_sample()

        for value in (True, 1.5, "1", -1):
            for parameter in ("k_prompt", "k_history"):
                kwargs = {"k_prompt": 1, "k_history": 1, parameter: value}
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    build_relation_topk_graph(sample, **kwargs)

    def test_direct_relation_topk_builder_accepts_zero_limits(self):
        graph = build_relation_topk_graph(attention_sample(), k_prompt=0, k_history=0)

        self.assertEqual(graph.edge_index.shape, (2, 0))

    def test_topk_ties_are_stable_causal_and_keep_prompt_history_relations_separate(self):
        sample = AttentionSample(
            "ties", "source", 2,
            torch.tensor([10, 11, 12, 13]),
            torch.ones((1, 1, 4), dtype=torch.float16),
            torch.tensor([0, 2, 4]),
            torch.tensor([0, 1, 0, 2], dtype=torch.int32),
            torch.tensor([0.2, 0.2, 0.3, 0.3], dtype=torch.float16),
            0.01,
        )
        graph = build_relation_topk_graph(sample, k_prompt=1, k_history=1)
        source, target = graph.edge_index.tolist()
        self.assertEqual(list(zip(source, target, graph.edge_type.tolist())), [(0, 2, 0), (0, 3, 0), (2, 3, 1)])
        self.assertTrue(all(left < right for left, right in zip(source, target)))

    def test_topk_is_target_major_with_prompt_before_history_and_aligned_channels(self):
        sample = AttentionSample(
            "ordered", "source", 2,
            torch.tensor([10, 11, 12, 13, 14]),
            torch.ones((1, 2, 5), dtype=torch.float16),
            torch.tensor([0, 0, 3, 5, 5, 7, 10]),
            torch.tensor([0, 1, 2, 0, 3, 1, 2, 0, 2, 3], dtype=torch.int32),
            torch.tensor([0.1, 0.3, 0.4, 0.4, 0.2, 0.5, 0.2, 0.2, 0.7, 0.1], dtype=torch.float16),
            0.01,
        )

        graph = build_relation_topk_graph(sample, k_prompt=1, k_history=1, with_channels=True)

        self.assertEqual(graph.edge_index.tolist(), [[1, 2, 0, 2], [3, 3, 4, 4]])
        self.assertEqual(graph.edge_type.tolist(), [0, 1, 0, 1])
        torch.testing.assert_close(graph.edge_weight, torch.tensor([0.4, 0.3, 0.3, 0.35]), rtol=0, atol=1e-3)
        self.assertEqual(graph.edge_ptr.tolist(), [0, 2, 4, 6, 7])
        self.assertEqual(graph.edge_channel.tolist(), [0, 1, 0, 1, 0, 1, 1])
        torch.testing.assert_close(
            graph.edge_value,
            torch.tensor([0.3, 0.5, 0.4, 0.2, 0.4, 0.2, 0.7], dtype=torch.float16),
        )

    def test_builder_rejects_same_size_attention_sample_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = write_split(root / "attention")
            path = cache / "attention" / "r1.npz"
            tampered = bytearray(path.read_bytes())
            tampered[len(tampered) // 2] ^= 1
            path.write_bytes(tampered)

            with self.assertRaisesRegex(ValueError, "SHA256"):
                GraphDatasetBuilder(BuildConfig(cache, root / "graphs", kind="original", tau=0.05, device="cpu")).run()

    def test_builder_rejects_input_manifest_or_index_changes_during_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = write_split(root / "attention")
            original_build = GraphDatasetBuilder._build

            def mutate_input(builder, sample):
                index = cache / "index.jsonl"
                index.write_text(index.read_text() + "\n", encoding="utf-8")
                return original_build(builder, sample)

            with patch.object(GraphDatasetBuilder, "_build", new=mutate_input), self.assertRaisesRegex(ValueError, "changed"):
                GraphDatasetBuilder(BuildConfig(cache, root / "graphs", kind="original", tau=0.05, device="cpu")).run()

    def test_hypergraph_keeps_source_then_target_order_within_one_hyperedge(self):
        sample = AttentionSample(
            "hyper", "source", 2,
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([[[1.0, 1.0, 0.7, 1.0]]], dtype=torch.float16),
            torch.tensor([0, 2, 2]),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0.2, 0.3], dtype=torch.float16),
            0.01,
        )

        graph = build_attention_hypergraph(sample, tau=0.05)

        self.assertEqual(graph.incidence_index.tolist(), [[0, 1, 2], [0, 0, 0]])
        torch.testing.assert_close(graph.incidence_weight, torch.tensor([0.2, 0.3, 0.7], dtype=torch.float16))
        self.assertEqual(graph.hyperedge_target.tolist(), [2])
        self.assertEqual(graph.hyperedge_channel.tolist(), [0])
        self.assertEqual(graph.hyperedge_type.tolist(), [0])

    def test_hypergraph_empty_input_has_empty_incidence(self):
        sample = AttentionSample(
            "empty", "source", 2,
            torch.tensor([10, 11, 12, 13]),
            torch.ones((1, 1, 4), dtype=torch.float16),
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0.02, 0.03], dtype=torch.float16),
            0.01,
        )

        graph = build_attention_hypergraph(sample, tau=0.05)

        self.assertEqual(graph.incidence_index.shape, (2, 0))
        self.assertEqual(graph.incidence_weight.numel(), 0)
        self.assertEqual(graph.hyperedge_target.numel(), 0)


if __name__ == "__main__":
    unittest.main()
