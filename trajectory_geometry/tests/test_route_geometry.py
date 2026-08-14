from pathlib import Path
import unittest

import numpy as np

from trajectory_geometry.data import SparseAttentionSample
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
