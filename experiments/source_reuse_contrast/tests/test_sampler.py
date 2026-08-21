import random

import torch

from experiments.source_reuse_contrast.data import collect_source_reuse_graph
from experiments.source_reuse_contrast.sampler import (
    matched_candidate_batch,
    relation_of,
    source_bin,
    usage_bucket,
)

from .helpers import sequence_sample, tiny_config


def test_matched_candidates_are_unique_and_never_use_current_sources():
    graph = collect_source_reuse_graph(sequence_sample())
    config = tiny_config(negative_count=1)
    token = 4
    current = graph.token_slice(token)
    true_sources = torch.unique(graph.source[current], sorted=True)
    use_count = torch.tensor([1, 1, 1, 0, 3, 2, 1, 0, 0])
    cumulative_mass = torch.tensor([0.4, 0.2, 0.3, 0.0, 0.8, 0.6, 0.2, 0.0, 0.0])
    last_used = torch.tensor([2, 2, 3, -1, 3, 3, 3, -1, -1])
    memory_norm = torch.ones(graph.num_tokens)

    result = matched_candidate_batch(
        graph,
        query=token,
        true_sources=true_sources,
        use_count=use_count,
        cumulative_mass=cumulative_mass,
        last_used=last_used,
        memory_norm=memory_norm,
        config=config,
        rng=random.Random(11),
    )

    current_set = set(true_sources.tolist())
    assert bool(result.valid.any())
    for row in torch.nonzero(result.valid, as_tuple=False).flatten().tolist():
        candidates = result.candidate_source[
            row, result.candidate_mask[row]
        ].tolist()
        assert len(candidates) == len(set(candidates))
        assert candidates[0] == int(true_sources[row])
        for candidate in candidates[1:]:
            assert candidate not in current_set
            assert relation_of(graph, candidate) == relation_of(
                graph, candidates[0]
            )
            assert source_bin(graph, token, candidate, config) == source_bin(
                graph, token, candidates[0], config
            )
            assert usage_bucket(
                int(use_count[candidate]), config.usage_bins
            ) == usage_bucket(int(use_count[candidates[0]]), config.usage_bins)
