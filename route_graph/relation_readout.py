"""Unfitted node/edge compatibility readouts for controlled relation tests."""

import numpy as np
import torch
import torch.nn.functional as F

from route_graph.ownership import capture_source_factors


@torch.no_grad()
def capture_relation(model, ids, queries, sources, numeric_positions):
    """Retain actual source A/V and query residuals, without fitting a probe."""
    layers = list(range(len(model.model.layers)))
    residuals, projections, handles = {}, {}, []
    try:
        for index in layers:
            projections[index] = {}

            def hook(module, args, index=index):
                residuals[index] = args[0][0, queries].float().cpu()

            handles.append(model.model.layers[index].register_forward_pre_hook(hook))
            for name, positions, heads in (
                ("q", queries, model.config.num_attention_heads),
                ("k", sources, model.config.num_key_value_heads),
            ):

                def projection_hook(
                    module,
                    args,
                    output,
                    index=index,
                    name=name,
                    positions=positions,
                    heads=heads,
                ):
                    projections[index][f"{name}_before_rope"] = (
                        output[0, positions]
                        .reshape(len(positions), heads, -1)
                        .float()
                        .cpu()
                    )

                projection = getattr(
                    model.model.layers[index].self_attn, f"{name}_proj"
                )
                handles.append(projection.register_forward_hook(projection_hook))
        logits, factors = capture_source_factors(model, ids, queries, sources, layers)
    finally:
        for handle in handles:
            handle.remove()
    node_indices = [sources.index(p) for p in numeric_positions]
    readouts = {name: [] for name in ("node", "attention", "graph")}
    for index, captured in factors.items():
        captured["query_residual"] = residuals[index]
        captured.update(projections[index])
        scores = compatibility(
            captured["query_residual"].to(model.device),
            captured["values"][node_indices].to(model.device),
            captured["attention"][:, :, node_indices].to(model.device),
            model.model.layers[index].self_attn.o_proj.weight.float(),
        )
        for name, value in scores.items():
            readouts[name].append(value.cpu().numpy())
    return logits, factors, {k: np.stack(v) for k, v in readouts.items()}


def compatibility(query, values, attention, output_weight):
    """N: node cosine; A: mean head mass; G: routed message/query projection.

    All use fixed numeric endpoints. No candidate logits or relation labels enter
    the score. These are compatibility measurements, not factuality probabilities.
    """
    heads, count, nodes = attention.shape
    if (
        query.ndim != 2
        or values.ndim != 3
        or values.shape[0] != nodes
        or count != len(query)
        or heads % values.shape[1]
        or output_weight.shape != (query.shape[-1], heads * values.shape[-1])
    ):
        raise ValueError("unaligned query, node, edge or output-projection shape")
    value = values.float().repeat_interleave(heads // values.shape[1], dim=1)
    unit = F.linear(value.reshape(nodes, -1), output_weight.float())
    denominator = unit.norm(dim=-1).clamp_min(1e-12)
    q = F.normalize(query.float(), dim=-1)
    node = q @ F.normalize(unit, dim=-1).T
    routed = torch.einsum("hqn,nhd->qnhd", attention.float(), value)
    message = F.linear(routed.reshape(count, nodes, -1), output_weight.float())
    graph = torch.einsum("qd,qnd->qn", q, message) / denominator
    return {"node": node, "attention": attention.float().mean(0), "graph": graph}
