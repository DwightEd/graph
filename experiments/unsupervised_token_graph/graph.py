"""Build one causal token graph per prompt+response."""

from dataclasses import dataclass

import numpy as np

from .reanchor import ReanchorAnalyzer


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
                query_count, tokens = attention.shape[-2:]
                aggregate = attention.mean(axis=(0, 1))
                diagonal = np.zeros(tokens, dtype=np.float32)
                target_rows = np.arange(query_count) + cache.prompt_length - 1
                query, source = np.nonzero(aggregate > attention_floor)
                target = target_rows[query]
                keep = source < target
                edge_weight = aggregate[query[keep], source[keep]]
                target, source = target[keep], source[keep]
        else:
            diagonal = cache.sparse["attention_diagonal"].mean(axis=(0, 1)).astype(np.float32)
            layers, heads, tokens = cache.sparse["attention_diagonal"].shape
            row_ptr = cache.sparse["response_row_ptr"]
            columns = cache.sparse["response_column_indices"]
            values = cache.sparse["response_values"]
            aggregate = np.zeros((tokens, tokens), dtype=np.float32)
            row = 0
            for layer in range(layers):
                for head in range(heads):
                    for target in range(cache.response_idx, tokens):
                        begin, end = row_ptr[row], row_ptr[row + 1]
                        aggregate[target, columns[begin:end]] += values[begin:end] / (layers * heads)
                        row += 1
            target, source = np.nonzero(np.tril(aggregate > attention_floor, -1))
            edge_weight = aggregate[target, source]
        edge_index = np.stack((source, target))
        incoming = np.bincount(target, weights=edge_weight, minlength=tokens)
        outgoing = np.bincount(source, weights=edge_weight, minlength=tokens)
        in_degree = np.bincount(target, minlength=tokens)
        out_degree = np.bincount(source, minlength=tokens)
        position = np.arange(tokens, dtype=np.float32) / max(tokens - 1, 1)
        split = cache.prompt_length if cache.prompt_length is not None else cache.response_idx
        segment = (np.arange(tokens) >= split).astype(np.float32)
        x = np.column_stack((position, segment, diagonal, incoming, outgoing,
                             in_degree / max(tokens, 1), out_degree / max(tokens, 1)))
        if cache.attention is not None and cache.prompt_length is not None:
            lookback = ReanchorAnalyzer().run(cache.attention, cache.prompt_length)
            structural = np.zeros((tokens, 5), dtype=np.float32)
            start = cache.prompt_length
            stop = min(tokens, start + len(lookback.waad))
            structural[start:stop] = np.column_stack((
                lookback.waad[:stop - start], lookback.fai[:stop - start],
                lookback.lookback_ratio[:stop - start],
                lookback.evidence_entropy[:stop - start],
                lookback.distribution_shift[:stop - start],
            ))
            x = np.column_stack((x, structural))
        return cls(cache.response_id, x, edge_index, edge_weight,
                   cache.response_idx, cache.token_ids)

    def features(self):
        return self.x
