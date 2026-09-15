"""Sparse per-head token graphs with explicit absolute query coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class ChannelGraph:
    layer: int
    head: int
    queries: np.ndarray
    attention: sparse.csr_matrix
    prompt_length: int

    def row(self, index):
        begin, end = self.attention.indptr[index:index + 2]
        return self.attention.indices[begin:end], self.attention.data[begin:end]

    @property
    def prediction_positions(self):
        return self.queries + 1


def _channel(layer, head, queries, matrix, prompt_length):
    """Validate the cache contract once; retain sub-stochastic row mass."""
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    queries = np.asarray(queries, dtype=np.int64)
    if len(queries) != matrix.shape[0] or np.any(np.diff(queries) <= 0):
        raise ValueError("one increasing absolute query position per attention row is required")
    rows = np.repeat(np.arange(len(queries)), np.diff(matrix.indptr))
    if (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
            or np.any(matrix.indices < 0) or np.any(matrix.indices >= matrix.shape[1])
            or np.any(matrix.indices > queries[rows])):
        raise ValueError("attention must contain finite, nonnegative, causal weights")
    if np.any(queries < 0) or np.any(queries >= matrix.shape[1]):
        raise ValueError("query position is outside the token coordinates")
    mass = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(mass > 1.005):
        raise ValueError("attention row mass exceeds one beyond cache rounding tolerance")
    # Only correct small floating-point overflow; missing mass stays missing.
    matrix.data /= np.repeat(np.maximum(mass, 1.), np.diff(matrix.indptr))
    return ChannelGraph(layer, head, queries, matrix, prompt_length)


def iter_channels(record, layers=None, heads=None):
    """Yield one (layer, head) CSR; never materialize [L,H,N,N]."""
    p = record.prompt_length if record.prompt_length is not None else record.response_idx
    if record.sparse is not None:
        cache = record.sparse
        diagonal = cache["attention_diagonal"]
        n_layers, n_heads, tokens = diagonal.shape
        count = tokens - p
        pointer = cache["response_row_ptr"]
        columns, values = cache["response_column_indices"], cache["response_values"]
        if (len(pointer) != n_layers * n_heads * count + 1 or pointer[0] != 0
                or pointer[-1] != len(values) or len(columns) != len(values)
                or np.any(np.diff(pointer) < 0)):
            raise ValueError("canonical CSR pointers do not match [layer,head,response_query]")
        queries = np.arange(p, tokens)
    else:
        attention = record.attention
        n_layers, n_heads, count, tokens = attention.shape
        if record.query_positions is not None:
            queries = record.query_positions
        elif count == tokens:
            queries = np.arange(tokens)
        else:
            queries = p - 1 + np.arange(count)
    identity = record.metadata or {}
    layer_ids = [identity["layer"]] if n_layers == 1 and "layer" in identity else list(range(n_layers))
    head_ids = [identity["head"]] if n_heads == 1 and "head" in identity else list(range(n_heads))
    for layer, physical_layer in enumerate(layer_ids):
        if layers is not None and physical_layer not in layers:
            continue
        for head, physical_head in enumerate(head_ids):
            if heads is not None and physical_head not in heads:
                continue
            if record.sparse is not None:
                first = (layer * n_heads + head) * count
                ptr = pointer[first:first + count + 1]
                begin, end = ptr[0], ptr[-1]
                # Promote only this channel before construction; SciPy sparse rejects float16.
                matrix = sparse.csr_matrix((values[begin:end], columns[begin:end], ptr - begin),
                                           shape=(count, tokens), dtype=np.float64)
                matrix = matrix + sparse.csr_matrix((diagonal[layer, head, p:],
                                                    (np.arange(count), queries)),
                                                   shape=(count, tokens), dtype=np.float64)
            else:
                matrix = sparse.csr_matrix(attention[layer, head], dtype=np.float64)
            # Prompt queries are not modelled internally; the last prompt query
            # can still supply the first next-token prediction when it was saved.
            keep = np.asarray(queries) >= p - 1
            yield _channel(physical_layer, physical_head, np.asarray(queries)[keep], matrix[keep], p)
