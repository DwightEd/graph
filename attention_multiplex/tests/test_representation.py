import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "attention_multiplex"
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attention_multiplex.representation import (  # noqa: E402
    MultiplexConfig,
    build_multiplex_unfolding,
    represent_attention_multiplex,
)
from attention_multiplex.pipeline import build_dataset_representations  # noqa: E402
from cache import AttentionSample  # noqa: E402


class _Sample:
    """Small ResearchSample-compatible object; no file parsing in the method."""

    def __init__(self):
        self.sample_id = "r1"
        self.source_id = "s1"
        self.task_type = "QA"
        self.data_source = "MARCO"
        self.sparse_passes = 0
        # L=2, H=2, N=4, response starts at 2, R=2.
        diagonal = torch.tensor(
            [
                [[0.60, 0.50, 0.40, 0.30], [0.55, 0.45, 0.35, 0.25]],
                [[0.50, 0.40, 0.30, 0.20], [0.45, 0.35, 0.25, 0.15]],
            ],
            dtype=torch.float16,
        )
        # Rows are ordered by layer, head, response query.
        row_columns = [[0], [2], [1], [0, 2], [0], [1, 2], [1], [0, 2]]
        row_values = [
            [0.20], [0.30], [0.15], [0.12, 0.22],
            [0.25], [0.18, 0.28], [0.11], [0.14, 0.24],
        ]
        row_ptr = [0]
        columns = []
        values = []
        for local_columns, local_values in zip(row_columns, row_values):
            columns.extend(local_columns)
            values.extend(local_values)
            row_ptr.append(len(columns))
        self._attention = AttentionSample(
            "r1",
            "s1",
            2,
            torch.tensor([1, 2, 3, 4]),
            diagonal,
            torch.tensor(row_ptr, dtype=torch.int32),
            torch.tensor(columns, dtype=torch.int32),
            torch.tensor(values, dtype=torch.float16),
            0.01,
        )
        self._attention.validate()

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        self.sparse_passes += 1
        from research_dataset import SparseAttentionBlock

        attention = self._attention
        rows_per_layer = attention.num_heads * attention.num_response_tokens
        total_rows = attention.num_channels * attention.num_response_tokens
        for start_row in range(0, total_rows, block_rows):
            stop_row = min(start_row + block_rows, total_rows)
            pointer = attention.response_row_ptr[start_row : stop_row + 1].long()
            lengths = pointer[1:] - pointer[:-1]
            rows = torch.repeat_interleave(
                torch.arange(start_row, stop_row, dtype=torch.long), lengths
            )
            start = int(pointer[0])
            stop = int(pointer[-1])
            query = rows.remainder(attention.num_response_tokens)
            yield SparseAttentionBlock(
                row=rows,
                layer=torch.div(rows, rows_per_layer, rounding_mode="floor"),
                head=torch.div(
                    rows.remainder(rows_per_layer),
                    attention.num_response_tokens,
                    rounding_mode="floor",
                ),
                query=query,
                target=attention.response_idx + query,
                source=attention.response_column_indices[start:stop].long(),
                weight=attention.response_values[start:stop],
            )

    def release_attention(self):
        return None


class _Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.sample_ids = ["r1"]
        self.manifest = {"split": "train", "num_layers": 2, "num_heads": 2}
        self.loads = 0

    def __getitem__(self, sample_id):
        self.loads += 1
        sample = _Sample()
        sample.sample_id = str(sample_id)
        return sample


class MultiplexRepresentationTests(unittest.TestCase):
    def test_unfolding_preserves_layer_head_and_query_source_roles(self):
        unfolding = build_multiplex_unfolding(
            _Sample(), config=MultiplexConfig(rank=2, block_rows=3)
        )
        self.assertEqual(unfolding.mass_excess.shape, (4, 8))
        self.assertEqual(unfolding.retained_off_diagonal_edges, 11)

        dense = unfolding.mass_excess.toarray()
        # layer 0/query 0 -> head 0/source 0: 0.20 - floor.
        self.assertAlmostEqual(float(dense[0, 0]), 0.19, places=3)
        # Same query in head 1/source 1 occupies a separate head block.
        self.assertAlmostEqual(float(dense[0, 4 + 1]), 0.14, places=3)
        # Exact self attention is present at head 0/source target 2.
        self.assertAlmostEqual(float(dense[0, 2]), 0.40, places=3)
        # A censored legal edge is not confused with an observed zero; after
        # floor-baseline removal it remains implicit zero.
        self.assertEqual(float(dense[0, 1]), 0.0)

    def test_representation_keeps_layer_and_head_trajectories(self):
        sample = _Sample()
        result = represent_attention_multiplex(
            sample, config=MultiplexConfig(rank=2, block_rows=3)
        )
        self.assertEqual(result.mass.query_by_layer.shape, (2, 2, 2))
        self.assertEqual(result.mass.source_by_head.shape, (2, 4, 2))
        self.assertEqual(result.shape.query_by_layer.shape, (2, 2, 2))
        self.assertEqual(result.shape.source_by_head.shape, (2, 4, 2))
        self.assertEqual(result.self_attention.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(result.mass.query_by_layer).all())
        self.assertTrue(np.isfinite(result.shape.source_by_head).all())
        self.assertGreater(result.mass.captured_energy, 0.0)
        self.assertLessEqual(result.mass.captured_energy, 1.00001)
        self.assertTrue((result.unresolved_row_mass >= 0).all())
        self.assertEqual(sample.sparse_passes, 1)

    def test_fixed_seed_is_deterministic(self):
        config = MultiplexConfig(rank=2, block_rows=2, random_seed=7)
        first = represent_attention_multiplex(_Sample(), config=config)
        second = represent_attention_multiplex(_Sample(), config=config)
        self.assertTrue(
            np.allclose(first.mass.query_by_layer, second.mass.query_by_layer)
        )
        self.assertTrue(
            np.allclose(first.shape.source_by_head, second.shape.source_by_head)
        )

    def test_pipeline_resumes_without_reloading_completed_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _Dataset(Path(temporary) / "input")
            output = Path(temporary) / "output"
            config = MultiplexConfig(rank=2, block_rows=3, random_seed=7)
            first = build_dataset_representations(
                dataset, output, config=config, checkpoint_every=1
            )
            self.assertEqual(first["computed_samples"], 1)
            self.assertEqual(dataset.loads, 1)

            second = build_dataset_representations(
                dataset,
                output,
                config=config,
                resume=True,
                checkpoint_every=1,
            )
            self.assertEqual(second["resumed_samples"], 1)
            self.assertEqual(second["computed_samples"], 0)
            self.assertEqual(dataset.loads, 1)
            state = json.loads((output / "run_state.json").read_text())
            self.assertEqual(state["state"], "complete")

    def test_resume_recomputes_incomplete_atomic_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _Dataset(Path(temporary) / "input")
            output = Path(temporary) / "output"
            config = MultiplexConfig(rank=2, block_rows=3, random_seed=7)
            build_dataset_representations(dataset, output, config=config)
            artifact = output / "samples" / "multiplex_r1.npz"
            artifact.write_bytes(b"interrupted")

            result = build_dataset_representations(
                dataset, output, config=config, resume=True
            )
            self.assertEqual(result["resumed_samples"], 0)
            self.assertEqual(result["computed_samples"], 1)
            self.assertEqual(dataset.loads, 2)
            with np.load(artifact, allow_pickle=False) as archive:
                self.assertEqual(str(archive["sample_id"]), "r1")

    def test_resume_rejects_changed_representation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _Dataset(Path(temporary) / "input")
            output = Path(temporary) / "output"
            build_dataset_representations(
                dataset,
                output,
                config=MultiplexConfig(rank=2, random_seed=7),
            )
            with self.assertRaisesRegex(ValueError, "incompatible run"):
                build_dataset_representations(
                    dataset,
                    output,
                    config=MultiplexConfig(rank=1, random_seed=7),
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
