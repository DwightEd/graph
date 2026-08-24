from __future__ import annotations

import torch

from experiments.attention_holonomy_audit.graph import AttentionEventGraph


def synthetic_graph(heads: int = 4) -> AttentionEventGraph:
    source = torch.tensor([0, 0, 2, 2, 1, 1, 3, 2])
    target = torch.tensor([2, 2, 3, 3, 3, 3, 4, 4])
    layer = torch.tensor([0, 1, 1, 2, 1, 2, 2, 2])
    role = (source >= 2).long()
    lag = target - source
    generator = torch.Generator().manual_seed(7)
    value = 0.02 + 0.25 * torch.rand((8, heads), generator=generator)
    observed = torch.rand((8, heads), generator=generator) > 0.15
    value = value * observed
    query_event_index = torch.tensor([0, 1, 2, 4, 3, 5, 6, 7])
    query_ptr = torch.tensor([0, 1, 2, 4, 6, 8])
    return AttentionEventGraph(
        sample_id="synthetic",
        source_id="source",
        task_type="QA",
        response_idx=2,
        num_tokens=6,
        num_response_tokens=4,
        num_layers=3,
        num_heads=heads,
        attention_floor=0.01,
        event_source=source,
        event_target=target,
        event_layer=layer,
        event_role=role,
        event_lag=lag,
        event_head_value=value,
        event_head_observed=observed,
        depth_edge_index=torch.tensor([[0, 2, 4], [1, 3, 5]]),
        relay_edge_index=torch.tensor([[0, 1, 2, 4], [2, 3, 6, 6]]),
        diamond_index=torch.tensor([[0], [1], [2], [3]]),
        query_event_index=query_event_index,
        query_ptr=query_ptr,
        query_target=torch.tensor([2, 2, 3, 3, 4]),
        query_layer=torch.tensor([0, 1, 1, 2, 2]),
        diagonal=torch.full((4, 3, heads), 0.1),
        unresolved_mass=torch.full((4, 3, heads), 0.2),
        response_token_ids=torch.tensor([10, 11, 12, 13]),
    )
