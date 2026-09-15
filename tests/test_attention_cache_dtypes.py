"""Storage dtypes must be promoted before any SciPy sparse construction."""

import numpy as np
import pytest

from experiments.unsupervised_token_graph.channels import iter_channels
from experiments.unsupervised_token_graph.data import ResponseCache


DTYPES = (np.float16, np.float32, np.float64)


@pytest.mark.parametrize("value_dtype", DTYPES)
@pytest.mark.parametrize("diagonal_dtype", DTYPES)
def test_canonical_npz_storage_dtypes(tmp_path, value_dtype, diagonal_dtype):
    p, tokens, layers, heads = 3, 6, 2, 2
    diagonal = np.zeros((layers, heads, tokens), dtype=diagonal_dtype)
    pointer, columns, values = [0], [], []
    expected = np.zeros((layers, heads, tokens - p, tokens), dtype=np.float64)
    for layer in range(layers):
        for head in range(heads):
            diagonal[layer, head, p:] = .2 + .025 * (layer + head)
            for q in range(p, tokens):
                keys = [0, q - 1]
                weights = np.array([.15 + .05 * layer, .1 + .025 * head], dtype=value_dtype)
                columns.extend(keys)
                values.extend(weights)
                pointer.append(len(values))
                expected[layer, head, q - p, keys] = weights
                expected[layer, head, q - p, q] = diagonal[layer, head, q]
    path = tmp_path / "attention_1.npz"
    np.savez_compressed(path, token_ids=np.arange(tokens), response_idx=p,
                        attention_diagonal=diagonal, response_row_ptr=np.asarray(pointer),
                        response_column_indices=np.asarray(columns),
                        response_values=np.asarray(values, dtype=value_dtype))
    original = path.read_bytes()
    record = ResponseCache().load(path)
    for channel in iter_channels(record):
        assert channel.attention.dtype == np.float64
        np.testing.assert_array_equal(channel.attention.toarray(), expected[channel.layer, channel.head])
        np.testing.assert_array_equal(channel.queries, np.arange(p, tokens))
        np.testing.assert_array_equal(channel.prediction_positions, np.arange(p + 1, tokens + 1))
    selected = list(iter_channels(record, layers=[1], heads=[0]))
    assert [(c.layer, c.head) for c in selected] == [(1, 0)]
    assert record.sparse["attention_diagonal"].dtype == diagonal_dtype
    assert record.sparse["response_values"].dtype == value_dtype
    np.testing.assert_array_equal(record.sparse["attention_diagonal"], diagonal)
    np.testing.assert_array_equal(record.sparse["response_values"], np.asarray(values, dtype=value_dtype))
    assert path.read_bytes() == original


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ("square", "compact", "adjacency"))
def test_dense_npz_storage_dtypes(tmp_path, dtype, layout):
    p, tokens = 2, 6
    if layout == "square":
        queries = np.arange(tokens)
    elif layout == "compact":
        queries = p - 1 + np.arange(tokens - p)
    else:
        queries = np.arange(p, tokens)
    attention = np.zeros((len(queries), tokens), dtype=dtype)
    for row, q in enumerate(queries):
        attention[row, 0] += .125
        attention[row, q] += .375
    path = tmp_path / "dense.npz"
    arrays = dict(token_ids=np.arange(tokens), response_idx=p)
    if layout == "adjacency":
        arrays.update(adjacency=attention, layer=10, head=7)
    else:
        key = "attention" if layout == "square" else "data"
        arrays[key] = attention[None, None]
    np.savez_compressed(path, **arrays)
    original = path.read_bytes()
    record = ResponseCache().load(path)
    before = record.attention.copy()
    channel = next(iter_channels(record))
    keep = queries >= p - 1
    assert channel.attention.dtype == np.float64
    np.testing.assert_array_equal(channel.attention.toarray(), attention[keep].astype(np.float64))
    np.testing.assert_array_equal(channel.queries, queries[keep])
    np.testing.assert_array_equal(channel.prediction_positions, queries[keep] + 1)
    assert (channel.layer, channel.head) == ((10, 7) if layout == "adjacency" else (0, 0))
    assert record.attention.dtype == dtype
    np.testing.assert_array_equal(record.attention, before)
    assert path.read_bytes() == original


@pytest.mark.parametrize("diagonal_value", (0., .25))
def test_float16_canonical_without_off_diagonal_edges(tmp_path, diagonal_value):
    path = tmp_path / "empty_edges.npz"
    diagonal = np.zeros((1, 1, 4), dtype=np.float16)
    diagonal[:, :, 2:] = diagonal_value
    np.savez_compressed(path, token_ids=np.arange(4), response_idx=2,
                        attention_diagonal=diagonal, response_row_ptr=np.array([0, 0, 0]),
                        response_column_indices=np.array([], dtype=np.int32),
                        response_values=np.array([], dtype=np.float16))
    channel = next(iter_channels(ResponseCache().load(path)))
    expected = np.zeros((2, 4))
    expected[np.arange(2), [2, 3]] = diagonal_value
    np.testing.assert_array_equal(channel.attention.toarray(), expected)
    assert channel.attention.dtype == np.float64
