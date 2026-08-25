import torch

from experiments.holoroute.graph import EventGraph, Events, QueryGroups


def synthetic_graph(heads: int = 4) -> EventGraph:
    source = torch.tensor([0, 0, 2, 2, 1, 1, 3, 2])
    target = torch.tensor([2, 2, 3, 3, 3, 3, 4, 4])
    layer = torch.tensor([0, 1, 1, 2, 1, 2, 2, 2])
    generator = torch.Generator().manual_seed(7)
    value = 0.02 + 0.25 * torch.rand((8, heads), generator=generator)
    observed = torch.rand((8, heads), generator=generator) > 0.15
    value = value * observed

    events = Events(
        source=source,
        target=target,
        layer=layer,
        role=(source >= 2).long(),
        lag=target - source,
        value=value,
        observed=observed,
    )
    queries = QueryGroups(
        events=torch.tensor([0, 1, 2, 4, 3, 5, 6, 7]),
        pointer=torch.tensor([0, 1, 2, 4, 6, 8]),
        target=torch.tensor([2, 2, 3, 3, 4]),
        layer=torch.tensor([0, 1, 1, 2, 2]),
    )
    return EventGraph(
        sample_id="synthetic",
        source_id="source",
        task_type="QA",
        response_start=2,
        token_count=6,
        response_count=4,
        layer_count=3,
        head_count=heads,
        attention_floor=0.01,
        events=events,
        depth_edges=torch.tensor([[0, 2, 4], [1, 3, 5]]),
        relay_edges=torch.tensor([[0, 1, 2, 4], [2, 3, 6, 6]]),
        queries=queries,
        diamonds=torch.tensor([[0], [1], [2], [3]]),
        diagonal=torch.full((4, 3, heads), 0.1),
        unresolved=torch.full((4, 3, heads), 0.2),
        response_token_ids=torch.tensor([10, 11, 12, 13]),
    ).check()
