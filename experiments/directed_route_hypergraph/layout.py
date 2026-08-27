"""Layer-ordered endpoint layouts for the sparse attention-flow proxy."""

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenGraph


@dataclass(frozen=True)
class EndpointLayout:
    """Final input-endpoint distribution for selected response tokens.

    Columns ``[:graph.token_count]`` are exact token endpoints.  The final
    column is an absorbing bucket for attention mass censored by the sparse
    cache.  This is an attention-only routing layout; it is not an OV-aware
    functional contribution matrix.
    """

    distribution: torch.Tensor


@dataclass(frozen=True)
class EndpointLayoutPlan:
    """Exact dependency closure for a subset of final response rows."""

    response_index: torch.Tensor
    required_rows: tuple[torch.Tensor, ...]
    layer_order: tuple[int, ...]
    endpoint_token_count: int
    work_element_count: int
    peak_state_elements: int


def _validated_layer_order(
    graph: TokenGraph,
    layer_order: tuple[int, ...] | None,
) -> tuple[int, ...]:
    order = (
        tuple(range(graph.layer_count))
        if layer_order is None
        else tuple(map(int, layer_order))
    )
    if sorted(order) != list(range(graph.layer_count)):
        raise ValueError("layer_order must be a permutation of all layers")
    return order


def _response_index(
    graph: TokenGraph,
    response_index: torch.Tensor,
) -> torch.Tensor:
    index = torch.as_tensor(response_index, dtype=torch.long).detach().cpu()
    if index.ndim != 1:
        raise ValueError("response_index must be one-dimensional")
    if len(index) and bool(((index < 0) | (index >= graph.response_count)).any()):
        raise ValueError("response_index is outside the graph")
    return index


def endpoint_layout_plan(
    graph: TokenGraph,
    response_index: torch.Tensor,
    *,
    layer_order: tuple[int, ...] | None = None,
) -> EndpointLayoutPlan:
    """Plan the exact sparse dependency closure for selected final rows.

    A selected row at layer ``l`` only requires its own preceding state and the
    preceding states of response tokens that have retained edges into it.  The
    backward closure therefore avoids the full ``R x (N + 1)`` teacher matrix
    whenever a small response-row subset is sufficient for the loss.
    """

    graph = graph.canonicalize()
    order = _validated_layer_order(graph, layer_order)
    selected = torch.unique(_response_index(graph, response_index), sorted=True)
    endpoint_token_count = (
        graph.response_start + int(selected[-1].item()) + 1
        if len(selected)
        else graph.response_start
    )
    endpoint_count = endpoint_token_count + 1
    required: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.long) for _ in range(graph.layer_count + 1)
    ]
    required[-1] = selected
    work_element_count = 0
    peak_state_elements = len(selected) * endpoint_count

    for position in range(graph.layer_count - 1, -1, -1):
        layer = order[position]
        output_rows = required[position + 1]
        output_mask = torch.zeros(graph.response_count, dtype=torch.bool)
        output_mask[output_rows] = True
        edges = graph.layer_edges(layer)
        target_response = edges.target - graph.response_start
        targets_selected = output_mask[target_response]
        response_relay = targets_selected & (
            edges.source >= graph.response_start
        )

        input_mask = output_mask.clone()
        if bool(response_relay.any()):
            input_mask[
                edges.source[response_relay] - graph.response_start
            ] = True
        input_rows = torch.nonzero(input_mask, as_tuple=False).flatten()
        required[position] = input_rows

        relay_count = int(response_relay.sum().item())
        work_element_count += endpoint_count * (
            len(output_rows) + relay_count
        )
        peak_state_elements = max(
            peak_state_elements,
            endpoint_count * (len(input_rows) + len(output_rows)),
        )

    return EndpointLayoutPlan(
        response_index=selected,
        required_rows=tuple(required),
        layer_order=order,
        endpoint_token_count=endpoint_token_count,
        work_element_count=work_element_count,
        peak_state_elements=peak_state_elements,
    )


@torch.no_grad()
def _selected_endpoint_layout(
    graph: TokenGraph,
    response_index: torch.Tensor,
    *,
    residual_weight: float,
    layer_order: tuple[int, ...] | None,
) -> EndpointLayout:
    requested = _response_index(graph, response_index)
    if not len(requested):
        return EndpointLayout(
            distribution=graph.diagonal.new_zeros((0, graph.token_count + 1))
        )

    unique, inverse = torch.unique(
        requested,
        sorted=True,
        return_inverse=True,
    )
    plan = endpoint_layout_plan(
        graph,
        unique,
        layer_order=layer_order,
    )
    device = graph.device
    dtype = graph.diagonal.dtype
    endpoint_count = plan.endpoint_token_count + 1
    unresolved_endpoint = plan.endpoint_token_count

    input_rows = plan.required_rows[0].to(device)
    response = torch.zeros(
        (len(input_rows), endpoint_count),
        device=device,
        dtype=dtype,
    )
    local = torch.arange(len(input_rows), device=device)
    response[local, graph.response_start + input_rows] = 1.0

    for position, layer in enumerate(plan.layer_order):
        input_rows_cpu = plan.required_rows[position]
        output_rows_cpu = plan.required_rows[position + 1]
        input_rows = input_rows_cpu.to(device)
        output_rows = output_rows_cpu.to(device)
        self_location = torch.searchsorted(input_rows, output_rows)
        previous = response[self_location]
        attention = torch.zeros(
            (len(output_rows), endpoint_count),
            device=device,
            dtype=dtype,
        )

        layer_edges = graph.layer_edges(layer)
        output_mask = torch.zeros(graph.response_count, dtype=torch.bool)
        output_mask[output_rows_cpu] = True
        keep = output_mask[layer_edges.target - graph.response_start]
        edges = layer_edges.select(keep).to(device)
        if edges.count:
            target_response = edges.target - graph.response_start
            target_local = torch.searchsorted(output_rows, target_response)
            prompt = edges.source < graph.response_start
            if bool(prompt.any()):
                attention.index_put_(
                    (target_local[prompt], edges.source[prompt]),
                    edges.weight[prompt] / graph.head_count,
                    accumulate=True,
                )
            history = ~prompt
            if bool(history.any()):
                source_response = edges.source[history] - graph.response_start
                source_local = torch.searchsorted(input_rows, source_response)
                transition = torch.sparse_coo_tensor(
                    torch.stack((target_local[history], source_local)),
                    edges.weight[history] / graph.head_count,
                    (len(output_rows), len(input_rows)),
                    device=device,
                    dtype=dtype,
                    check_invariants=False,
                ).coalesce()
                attention = attention + torch.sparse.mm(
                    transition,
                    response,
                )

        diagonal = graph.diagonal[output_rows, layer].to(dtype=dtype).mean(1)
        unresolved = graph.unresolved[output_rows, layer].to(dtype=dtype).mean(1)
        attention = attention + diagonal[:, None] * previous
        attention[:, unresolved_endpoint] += unresolved
        response = (
            residual_weight * previous + attention
        ) / (residual_weight + 1.0)

    full = torch.zeros(
        (len(unique), graph.token_count + 1),
        device=device,
        dtype=dtype,
    )
    full[:, : plan.endpoint_token_count] = response[
        :, : plan.endpoint_token_count
    ]
    full[:, -1] = response[:, -1]
    return EndpointLayout(distribution=full[inverse.to(device)])


@torch.no_grad()
def ordered_endpoint_layout(
    graph: TokenGraph,
    *,
    residual_weight: float = 1.0,
    layer_order: tuple[int, ...] | None = None,
    response_index: torch.Tensor | None = None,
) -> EndpointLayout:
    """Compose response-row transitions in actual Transformer layer order.

    Prompt rows are unavailable in the cache and therefore remain identity
    endpoints.  Heads stay separate in the sparse edge table and are merged by
    a fixed uniform mean only when forming one layer transition.  The explicit
    residual term is a registered proxy rather than a measured residual-stream
    attribution.

    ``response_index`` requests an exact subset of final response rows.  Their
    layer-wise response dependencies are closed before the forward rollout, so
    the result is identical to selecting those rows from the full layout while
    avoiding unrelated response states.
    """

    if residual_weight < 0:
        raise ValueError("residual_weight must be non-negative")
    graph = graph.canonicalize()
    order = _validated_layer_order(graph, layer_order)
    if response_index is not None:
        return _selected_endpoint_layout(
            graph,
            response_index,
            residual_weight=residual_weight,
            layer_order=order,
        )

    device = graph.device
    dtype = graph.diagonal.dtype
    unresolved_endpoint = graph.token_count
    endpoint_count = graph.token_count + 1
    response = torch.zeros(
        (graph.response_count, endpoint_count),
        device=device,
        dtype=dtype,
    )
    response_index = torch.arange(graph.response_count, device=device)
    response[response_index, graph.response_start + response_index] = 1.0

    for layer in order:
        edges = graph.layer_edges(layer, device)
        attention = torch.zeros_like(response)
        if edges.count:
            target = edges.target - graph.response_start
            prompt = edges.source < graph.response_start
            if bool(prompt.any()):
                attention.index_put_(
                    (target[prompt], edges.source[prompt]),
                    edges.weight[prompt] / graph.head_count,
                    accumulate=True,
                )
            history = ~prompt
            if bool(history.any()):
                transition = torch.sparse_coo_tensor(
                    torch.stack(
                        (
                            target[history],
                            edges.source[history] - graph.response_start,
                        )
                    ),
                    edges.weight[history] / graph.head_count,
                    (graph.response_count, graph.response_count),
                    device=device,
                    dtype=dtype,
                    check_invariants=False,
                ).coalesce()
                attention = attention + torch.sparse.mm(
                    transition,
                    response,
                )

        diagonal = graph.diagonal[:, layer].to(device=device, dtype=dtype).mean(1)
        unresolved = graph.unresolved[:, layer].to(
            device=device,
            dtype=dtype,
        ).mean(1)
        attention = attention + diagonal[:, None] * response
        attention[:, unresolved_endpoint] += unresolved
        response = (
            residual_weight * response + attention
        ) / (residual_weight + 1.0)

    return EndpointLayout(distribution=response)
