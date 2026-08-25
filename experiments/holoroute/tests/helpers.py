import torch

from experiments.holoroute.graph import AttentionEdges, AttentionGraph


def synthetic_graph() -> AttentionGraph:
    # Prompt tokens: 0,1. Response tokens: 2,3,4. Three layers, two heads.
    source = torch.tensor([
        0, 1, 0, 1,
        0, 2, 1, 2, 3,
        2, 3, 1, 3,
    ])
    target = torch.tensor([
        2, 2, 3, 3,
        2, 3, 3, 4, 4,
        3, 4, 4, 4,
    ])
    layer = torch.tensor([
        0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2,
    ])
    head = torch.tensor([
        0, 1, 0, 1,
        0, 0, 1, 0, 1,
        0, 1, 0, 1,
    ])
    weight = torch.tensor([
        0.55, 0.50, 0.40, 0.45,
        0.35, 0.35, 0.30, 0.55, 0.50,
        0.50, 0.55, 0.30, 0.40,
    ])
    pointer = torch.tensor([0, 4, 9, 13])
    diagonal = torch.full((3, 3, 2), 0.25)
    retained = torch.zeros_like(diagonal)
    retained.index_put_((target - 2, layer, head), weight, accumulate=True)
    unresolved = (1.0 - retained - diagonal).clamp_min(0.0)
    return AttentionGraph(
        sample_id="synthetic",
        source_id="source",
        task_type="QA",
        response_start=2,
        token_count=5,
        response_count=3,
        layer_count=3,
        head_count=2,
        attention_floor=0.01,
        edges=AttentionEdges(source, target, layer, head, weight, pointer),
        diagonal=diagonal,
        unresolved=unresolved,
        response_token_ids=torch.tensor([10, 11, 12]),
    ).check()
