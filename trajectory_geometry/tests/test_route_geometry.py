from pathlib import Path
import tempfile
import unittest

import numpy as np

from trajectory_geometry.data import SparseAttentionSample, discover_attention_files
from trajectory_geometry.pipeline import extract_one
from trajectory_geometry.routing import AnchorSpec, encode_route_dynamics


def _sample(switched: bool = False) -> SparseAttentionSample:
    layers, heads, tokens, response_idx = 2, 2, 5, 3
    response = tokens - response_idx
    diagonal = np.full((layers, heads, tokens), 0.1, dtype=np.float64)
    columns: list[int] = []
    values: list[float] = []
    row_ptr = [0]
    for layer in range(layers):
        for head in range(heads):
            for query in range(response):
                target = response_idx + query
                if switched and query == 1:
                    columns.append(response_idx)
                else:
                    columns.append(1)
                values.append(0.6)
                if target > response_idx:
                    columns.append(target - 1)
                    values.append(0.1)
                order = np.argsort(columns[row_ptr[-1] :])
                start = row_ptr[-1]
                block_columns = [columns[start + int(index)] for index in order]
                block_values = [values[start + int(index)] for index in order]
                columns[start:] = block_columns
                values[start:] = block_values
                row_ptr.append(len(columns))
    sample = SparseAttentionSample(
        path=Path("synthetic.pt"),
        sample_id="synthetic",
        response_idx=response_idx,
        token_ids=np.arange(tokens),
        diagonal=diagonal,
        row_ptr=np.asarray(row_ptr, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        attention_floor=0.01,
    )
    sample.validate()
    return sample


class RouteGeometryTests(unittest.TestCase):
    def test_npz_reader_and_extraction_pipeline(self) -> None:
        sample = _sample(True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "attention_synthetic.npz"
            output = Path(directory) / "route_synthetic.npz"
            np.savez_compressed(
                source,
                response_idx=np.asarray(sample.response_idx),
                token_ids=sample.token_ids,
                attention_diagonal=sample.diagonal,
                response_row_ptr=sample.row_ptr,
                response_column_indices=sample.columns,
                response_values=sample.values,
                attention_floor=np.asarray(sample.attention_floor),
            )
            row = extract_one(
                source,
                output,
                spec=AnchorSpec(),
                embedding_dim=32,
                seed=7,
                csr_row_block=3,
                save_raw_route=False,
            )
            self.assertEqual(row["response_tokens"], sample.response_tokens)
            self.assertTrue(output.is_file())
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["route_embedding"].shape, (2, 32))
                self.assertNotIn("raw_route_mass", result.files)

    def test_documentation_placeholder_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "documentation placeholder"):
            discover_attention_files("/path/to/attention_cache", "train")

    def test_dense_rows_zero_fill_without_full_tensor(self) -> None:
        sample = _sample()
        rows = list(sample.iter_dense_rows())
        self.assertEqual(len(rows), sample.response_rows)
        first = rows[0]
        self.assertEqual((first.layer, first.head, first.query, first.target), (0, 0, 0, 3))
        self.assertEqual(first.values.shape, (sample.token_count,))
        self.assertAlmostEqual(float(first.values[1]), 0.6, places=6)
        self.assertAlmostEqual(float(first.values[3]), 0.1, places=6)
        self.assertEqual(float(first.values[0]), 0.0)
        self.assertEqual(float(first.values[2]), 0.0)

    def test_sparse_blocks_cover_every_retained_value(self) -> None:
        sample = _sample(True)
        blocks = list(sample.iter_sparse_row_blocks(block_rows=3))
        self.assertGreater(len(blocks), 1)
        self.assertEqual(sum(block.weight.size for block in blocks), sample.values.size)
        np.testing.assert_array_equal(
            np.concatenate([block.source for block in blocks]), sample.columns
        )

    def test_route_mass_preserves_censored_mass(self) -> None:
        result = encode_route_dynamics(
            _sample(), spec=AnchorSpec(prompt_bins=3), embedding_dim=32, seed=7
        )
        self.assertEqual(result.route_mass.shape, (2, 2, 2, 12))
        np.testing.assert_allclose(result.route_mass.sum(axis=-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.unresolved_mass[0], 0.3, atol=1e-6)
        np.testing.assert_allclose(result.unresolved_mass[1], 0.2, atol=1e-6)

    def test_route_switch_increases_temporal_js(self) -> None:
        stable = encode_route_dynamics(_sample(False), embedding_dim=32, seed=7)
        switched = encode_route_dynamics(_sample(True), embedding_dim=32, seed=7)
        self.assertGreater(switched.temporal_js[1], stable.temporal_js[1])

    def test_projection_is_deterministic_and_seeded(self) -> None:
        sample = _sample(True)
        first = encode_route_dynamics(sample, embedding_dim=32, seed=7)
        second = encode_route_dynamics(sample, embedding_dim=32, seed=7)
        other = encode_route_dynamics(sample, embedding_dim=32, seed=8)
        np.testing.assert_array_equal(first.route_embedding, second.route_embedding)
        self.assertFalse(np.array_equal(first.route_embedding, other.route_embedding))


if __name__ == "__main__":
    unittest.main()
