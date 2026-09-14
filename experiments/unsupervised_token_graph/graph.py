"""Build one causal token graph per prompt+response."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse



class TokenGraph:
    """Canonical graph tensors for one response."""

    def __init__(self, response_id, x, edge_index, edge_weight, response_idx, token_ids=None):
        self.response_id = response_id
        self.x = x.astype(np.float32)
        self.edge_index = edge_index.astype(np.int64)
        self.edge_weight = edge_weight.astype(np.float32)
        self.response_idx = response_idx
        self.token_ids = token_ids

    @classmethod
    def from_cache(cls, cache, attention_floor):
        if cache.attention is not None:
            attention = np.asarray(cache.attention, dtype=np.float32)
            layers, heads = attention.shape[:2]
            if attention.shape[-2] == attention.shape[-1]:
                tokens = attention.shape[-1]
                aggregate = attention.mean(axis=(0, 1))
                diagonal = attention.diagonal(axis1=2, axis2=3).mean(axis=(0, 1))
                target, source = np.nonzero(np.tril(aggregate > attention_floor, -1))
                edge_weight = aggregate[target, source]
            else:
                raise ValueError(
                    "rectangular response-query attention cannot form complete token nodes; "
                    "use the formal CSR cache or a square dense cache"
                )
        else:
            diagonal = cache.sparse["attention_diagonal"].mean(axis=(0, 1)).astype(np.float32)
            layers, heads, tokens = cache.sparse["attention_diagonal"].shape
            row_ptr = cache.sparse["response_row_ptr"]
            columns = cache.sparse["response_column_indices"]
            values = cache.sparse["response_values"]
            edge_rows, edge_cols, edge_values = [], [], []
            row = 0
            for layer in range(layers):
                for head in range(heads):
                    for target in range(cache.response_idx, tokens):
                        begin, end = row_ptr[row], row_ptr[row + 1]
                        source_row = columns[begin:end]
                        value_row = values[begin:end] / (layers * heads)
                        keep = (source_row < target) & (value_row > 0)
                        edge_rows.append(np.full(int(keep.sum()), target, dtype=np.int64))
                        edge_cols.append(source_row[keep].astype(np.int64))
                        edge_values.append(value_row[keep].astype(np.float32))
                        row += 1
            if edge_rows:
                target = np.concatenate(edge_rows)
                source = np.concatenate(edge_cols)
                aggregated = sparse.coo_matrix(
                    (np.concatenate(edge_values), (target, source)), shape=(tokens, tokens)
                ).tocsr()
                aggregated.data[aggregated.data <= attention_floor] = 0.
                aggregated.eliminate_zeros()
                target, source = aggregated.nonzero()
                edge_weight = aggregated.data.astype(np.float32, copy=True)
            else:
                target = source = np.empty(0, dtype=np.int64)
                edge_weight = np.empty(0, dtype=np.float32)
        edge_index = np.stack((source, target))
        incoming = np.bincount(target, weights=edge_weight, minlength=tokens)
        outgoing = np.bincount(source, weights=edge_weight, minlength=tokens)
        in_degree = np.bincount(target, minlength=tokens)
        out_degree = np.bincount(source, minlength=tokens)
        split = cache.prompt_length if cache.prompt_length is not None else cache.response_idx
        # Do not normalize by total response length: that exposes future tokens.
        position = np.log1p(np.arange(tokens, dtype=np.float32)) / max(np.log1p(split), 1.)
        segment = (np.arange(tokens) >= split).astype(np.float32)
        # Outgoing mass/degree depends on later queries and is deliberately omitted.
        x = np.column_stack((position, segment, diagonal, incoming,
                             in_degree / max(tokens, 1)))
        return cls(cache.response_id, x, edge_index, edge_weight,
                   cache.response_idx, cache.token_ids)

    def features(self):
        return self.x
