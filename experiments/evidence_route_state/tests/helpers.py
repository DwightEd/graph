import torch

from experiments.evidence_route_state.graph import RouteEdges


def route_row(
    layer: int,
    query: int,
    *,
    source: tuple[int, ...] = (),
    support: tuple[float, ...] = (),
    residual_support: float = 1.0,
    unknown_support: float = 0.0,
    heads: int = 1,
) -> RouteEdges:
    """One hand-computable positive-support row for lineage tests."""

    source_tensor = torch.tensor(source, dtype=torch.long)
    support_tensor = torch.tensor(support, dtype=torch.float32)
    head = torch.zeros(len(source), dtype=torch.long)
    messages = torch.stack((support_tensor, torch.zeros_like(support_tensor)), dim=-1)
    unknown = torch.zeros(heads, dtype=torch.float32)
    unknown[0] = unknown_support
    return RouteEdges(
        layer=layer,
        query_position=query,
        source=source_tensor,
        head=head,
        message=messages,
        capacity=support_tensor.abs(),
        support=support_tensor,
        unknown_message=torch.zeros(heads, 2),
        unknown_capacity=unknown.clone(),
        unknown_support=unknown.clone(),
        unknown_positive_support=unknown,
        residual_support=torch.tensor(residual_support),
    )


def two_layer_history(*, grounded: bool) -> tuple[list[RouteEdges], torch.Tensor]:
    """A history edge whose source either does or does not inherit evidence."""

    rows = []
    for layer in range(2):
        for query in (2, 3, 4):
            if layer == 0 and query == 3 and grounded:
                rows.append(
                    route_row(
                        layer,
                        query,
                        source=(0,),
                        support=(1.0,),
                        residual_support=0.0,
                    )
                )
            elif layer == 1 and query == 4:
                rows.append(
                    route_row(
                        layer,
                        query,
                        source=(3,),
                        support=(1.0,),
                        residual_support=0.0,
                    )
                )
            else:
                rows.append(route_row(layer, query))

    # Root 0 is other prompt, root 1 is evidence, and each response token has
    # its own response-born root 2..4.
    roots = torch.tensor([1, 0, 0, 2, 3, 4])
    return rows, roots
