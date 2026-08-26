"""Prompt, response-closed and unresolved path lineage."""

import torch
from torch import nn

PROMPT_ORIGIN = 0
RESPONSE_CLOSED = 1
UNRESOLVED = 2
LINEAGE_STATES = 3


class HeadTransition(nn.Module):
    """Learn how head identities correspond between adjacent layers."""

    def __init__(self, layers: int, heads: int, identity_bias: float) -> None:
        super().__init__()
        initial = torch.eye(heads).expand(layers, -1, -1) * identity_bias
        self.logit = nn.Parameter(initial.clone())

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logit, dim=-1)


def source_lineage(
    graph,
    previous: torch.Tensor | None,
    transition: torch.Tensor,
    source: torch.Tensor,
    head: torch.Tensor,
) -> torch.Tensor:
    prompt = transition.new_tensor((1.0, 0.0, 0.0)).expand(len(source), -1)
    if previous is None:
        response = transition.new_tensor((0.0, 1.0, 0.0)).expand(len(source), -1)
    else:
        source_index = (source - graph.response_start).clamp_min(0)
        response = torch.einsum(
            "eh,ehk->ek",
            transition[head],
            previous[source_index],
        )
    return torch.where((source < graph.response_start)[:, None], prompt, response)


def lineage_layer(
    graph,
    previous: torch.Tensor | None,
    transition: torch.Tensor,
    layer: int,
    edges,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = transition.new_zeros(
        (graph.response_count, graph.head_count, LINEAGE_STATES)
    )
    provenance = source_lineage(
        graph,
        previous,
        transition,
        edges.source,
        edges.head,
    )

    if edges.count:
        target = (edges.target - graph.response_start) * graph.head_count + edges.head
        current = current.view(-1, LINEAGE_STATES).index_add(
            0,
            target,
            provenance * edges.weight[:, None],
        ).view(graph.response_count, graph.head_count, LINEAGE_STATES)

    if previous is None:
        previous_target = current.new_zeros(current.shape)
        previous_target[..., RESPONSE_CLOSED] = 1.0
    else:
        previous_target = torch.einsum("hj,rjk->rhk", transition, previous)

    diagonal = graph.diagonal[:, layer].to(current.device)
    unresolved = graph.unresolved[:, layer].to(current.device)
    current = current + diagonal[..., None] * previous_target
    current[..., UNRESOLVED] = current[..., UNRESOLVED] + unresolved
    return current, provenance


def trace_lineage(graph, transition: torch.Tensor) -> torch.Tensor:
    history = []
    previous = None
    for layer in range(graph.layer_count):
        edges = graph.layer_edges(layer, transition.device)
        previous, _ = lineage_layer(graph, previous, transition[layer], layer, edges)
        history.append(previous)
    return torch.stack(history, dim=1)
