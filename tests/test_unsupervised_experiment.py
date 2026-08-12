import unittest

import numpy as np
import torch

from attention_gnn import build_attention_graph
from unsupervised_experiment import (
    AllDataEvaluator,
    LearnedEmbeddingVisualizer,
    UnsupervisedGraphMethod,
)


class _Attention:
    def __init__(self, response_tokens):
        self.num_response_tokens = response_tokens


class _Sample:
    def __init__(self, sample_id, source_id, response_tokens=2):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.data_source = "RAGTruth"
        self.generator_model = "llama31-8b"
        self._attention = _Attention(response_tokens)

    def attention(self):
        return self._attention

    def release_attention(self):
        pass


class _Labels:
    def response_labels(self, sample):
        return torch.arange(sample.attention().num_response_tokens) % 2


class _Dataset:
    def __init__(self):
        self.samples = {
            f"sample-{index}": _Sample(f"sample-{index}", f"source-{index // 2}")
            for index in range(6)
        }
        self.label_reads = 0

    @property
    def sample_ids(self):
        return list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[sample_id]

    def labels(self):
        self.label_reads += 1
        return _Labels()


class AllDataEvaluatorTests(unittest.TestCase):
    def test_oof_run_is_source_grouped_complete_and_label_blind_until_evaluation(self):
        dataset = _Dataset()
        fold_calls = []

        def fit_fold(train_samples, heldout_samples, fold):
            train_sources = {sample.source_id for sample in train_samples}
            heldout_sources = {sample.source_id for sample in heldout_samples}
            fold_calls.append((train_sources, heldout_sources))
            return {
                sample.sample_id: {
                    "embedding": np.full(
                        (sample.attention().num_response_tokens, 3), fold, dtype=np.float32
                    ),
                    "score": np.arange(
                        sample.attention().num_response_tokens, dtype=np.float32
                    ),
                }
                for sample in heldout_samples
            }

        evaluator = AllDataEvaluator(dataset, folds=3, seed=7)
        records = evaluator.run(fit_fold)

        self.assertEqual(dataset.label_reads, 0)
        self.assertTrue(fold_calls)
        self.assertTrue(
            all(train.isdisjoint(heldout) for train, heldout in fold_calls),
            "one source_id must never occur in both sides of a fold",
        )
        expected_tokens = {
            (sample.sample_id, token_index)
            for sample in dataset.samples.values()
            for token_index in range(sample.attention().num_response_tokens)
        }
        self.assertEqual(
            {(row["sample_id"], row["token_index"]) for row in records}, expected_tokens
        )
        self.assertEqual(len(records), len(expected_tokens))
        self.assertTrue(
            all(
                "embedding" in row
                and "score" in row
                and "fold" in row
                and "source_id" in row
                for row in records
            )
        )
        self.assertTrue(all("label" not in row for row in records))

        evaluated = evaluator.evaluate(records)

        self.assertEqual(dataset.label_reads, 1)
        self.assertEqual(len(evaluated), len(records))
        self.assertTrue(all("label" in row for row in evaluated))
        self.assertTrue(all("label" not in row for row in records))


class LearnedEmbeddingVisualizerTests(unittest.TestCase):
    def test_projection_fits_train_fold_and_only_transforms_heldout_embeddings(self):
        from sklearn.decomposition import PCA

        train = np.asarray(
            [[-3.0, 0.0, 0.1], [-1.0, 0.1, 0.0], [1.0, -0.1, 0.0], [3.0, 0.0, -0.1]]
        )
        heldout = np.asarray([[20.0, -4.0, 1.0], [20.0, 4.0, -1.0]])
        expected = PCA(n_components=2, random_state=5).fit(train).transform(heldout)

        coordinates = LearnedEmbeddingVisualizer(random_state=5).project_fold(
            train, heldout
        )

        np.testing.assert_allclose(coordinates, expected)

    def test_plot_records_uses_learned_embeddings_and_saves_coordinates(self):
        import tempfile
        from pathlib import Path

        records = [
            {
                "sample_id": f"sample-{index // 2}",
                "source_id": f"source-{index // 2}",
                "fold": 0,
                "token_index": index % 2,
                "embedding": np.asarray([index, index % 3, 1.0], dtype=np.float32),
                "score": float(index),
                "label": index % 2,
            }
            for index in range(6)
        ]
        train_embeddings = np.asarray(
            [[-3.0, 0.0, 0.1], [-1.0, 0.1, 0.0], [1.0, -0.1, 0.0], [3.0, 0.0, -0.1]]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fold_0_embedding.png"
            result = LearnedEmbeddingVisualizer(random_state=5).plot_fold(
                train_embeddings, records, output
            )

            self.assertTrue(output.is_file())
            self.assertEqual(result["coordinates"].shape, (6, 2))
            self.assertEqual(result["representation"], "learned_gnn_node_embedding")


class UnsupervisedGraphMethodTests(unittest.TestCase):
    def test_fit_and_score_use_learned_response_embeddings_without_labels(self):
        train_samples = [
            _GraphSample("train-1", "source-a", offset=0.00),
            _GraphSample("train-2", "source-b", offset=0.02),
        ]
        heldout = [_GraphSample("test-1", "source-c", offset=0.01)]
        method = UnsupervisedGraphMethod(
            num_channels=1,
            embedding_dim=4,
            message_passing_steps=1,
            epochs=2,
            fit_steps=2,
            seed=3,
        )

        method.fit(train_samples)
        output = method.score(heldout)["test-1"]
        train_embeddings = method.embed(train_samples)

        self.assertEqual(output["embedding"].shape, (2, 4))
        self.assertEqual(output["score"].shape, (2,))
        self.assertTrue(np.isfinite(output["embedding"]).all())
        self.assertTrue(np.isfinite(output["score"]).all())
        self.assertEqual(train_embeddings["train-1"].shape, (2, 4))
        self.assertTrue(set(method.density_source_ids).isdisjoint(method.calibration_source_ids))

    def test_training_masks_a_fraction_in_each_relation_group(self):
        method = UnsupervisedGraphMethod(
            num_channels=1,
            embedding_dim=4,
            message_passing_steps=1,
            epochs=1,
            fit_steps=1,
            edge_mask_rate=0.5,
            channel_mask_rate=0.0,
            seed=3,
        )
        graph = build_attention_graph(_GraphSample("x", "s", 0.0).attention())
        view = method._training_view(graph, torch.Generator().manual_seed(3))
        groups = graph.edge_index[1] * 2 + graph.edge_type

        for group in torch.unique(groups):
            members = groups == group
            count = int(members.sum())
            expected = 0 if count == 1 else max(1, round(count * 0.5))
            expected = min(expected, count - 1)
            self.assertEqual(int((~view.visible_edges & members).sum()), expected)


class _GraphAttention:
    def __init__(self, offset):
        self.sample_id = "unused"
        self.source_id = "unused"
        self.response_idx = 2
        self.token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        self.attention_diagonal = torch.tensor(
            [[[1.0, 0.8, 0.5 + offset, 0.4 + offset]]], dtype=torch.float16
        )
        self.response_row_ptr = torch.tensor([0, 2, 4], dtype=torch.int32)
        self.response_column_indices = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
        self.response_values = torch.tensor(
            [0.2 + offset, 0.4, 0.1 + offset, 0.3], dtype=torch.float16
        )
        self.attention_floor = 0.01

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_response_tokens(self):
        return self.num_tokens - self.response_idx

    @property
    def num_channels(self):
        return 1


class _GraphSample:
    def __init__(self, sample_id, source_id, offset):
        self.sample_id = sample_id
        self.source_id = source_id
        self._attention = _GraphAttention(offset)
        self._attention.sample_id = sample_id
        self._attention.source_id = source_id

    def attention(self):
        return self._attention

    def release_attention(self):
        pass


if __name__ == "__main__":
    unittest.main()
