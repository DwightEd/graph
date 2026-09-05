"""Native attention/message transport with independent target functionality."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from experiments.common.llama_message_intervention import ForwardCache, baseline_forward

from .attribution import TargetGradients, native_target_gradients
from .flow import (
    FlowEdges,
    FlowSignal,
    PairedFlow,
    RowAggregation,
    attention_qkv,
    attention_rows,
    coverage_mask,
    margin,
    net_row_message_norm,
    project_selected_messages,
)
from .message_norm import model_gram_cache, output_gram, source_norm
from .native_world import NativeWorld
from .worlds import SourceUnits, TargetContrast


def represented_positions(
    world: NativeWorld,
    target: TargetContrast,
    scope: str,
) -> Tensor:
    if scope == "all":
        return torch.arange(target.query_position + 1)
    if scope == "response":
        return torch.arange(world.response_start - 1, target.query_position + 1)
    raise ValueError("carrier_scope must be 'response' or 'all'")


def _empty_vector(count: int) -> Tensor:
    return torch.empty((count, 0), dtype=torch.float32)


def capture_native_edges(
    model,
    cache: ForwardCache,
    units: SourceUnits,
    positions: Tensor,
    signal: FlowSignal,
    gradients: TargetGradients,
    *,
    coverage: float,
    query_chunk: int,
) -> tuple[FlowEdges, Tensor, Tensor, RowAggregation, Tensor]:
    """Capture native transport and signed functionality without averaging."""

    if query_chunk < 1:
        raise ValueError("query chunk must be positive")
    layers = cache.layer_count
    heads = int(model.config.num_attention_heads)
    rows = len(positions)
    row_total = torch.zeros(layers, heads, rows)
    row_retained = torch.zeros_like(row_total)
    row_budget = torch.zeros(layers, rows)
    row_net = torch.zeros_like(row_budget)
    row_signed = torch.zeros_like(row_budget)
    row_positive = torch.zeros_like(row_budget)
    row_negative = torch.zeros_like(row_budget)
    residual_weight = torch.zeros_like(row_budget)
    columns: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "layer",
            "head",
            "source",
            "target",
            "source_unit",
            "attention",
            "transport",
            "functional",
            "message_norm",
            "code",
        )
    }
    device = model.get_input_embeddings().weight.device
    token_units = units.token_unit_id.to(device)
    gram_cache = model_gram_cache(model)
    gradient_lookup = torch.full(
        (len(units.token_unit_id),), -1, dtype=torch.long, device=device
    )
    gradient_lookup[gradients.position.to(device)] = torch.arange(
        len(gradients.position), device=device
    )

    for layer_index, layer in enumerate(model.model.layers):
        query, key, value = attention_qkv(
            model, layer_index, cache.layer_input[layer_index]
        )
        head_dim = value.shape[-1]
        output = layer.self_attn.o_proj.weight.detach()
        gram = gram_cache.get(layer_index)
        if gram is None:
            gram = output_gram(output, heads, head_dim)
            gram_cache[layer_index] = gram
        native_source_norm = source_norm(value, output, gram)
        scaling = float(getattr(layer.self_attn, "scaling", head_dim**-0.5))
        if signal is FlowSignal.ATTENTION:
            residual_weight[layer_index].fill_(float(heads))
        else:
            residual_weight[layer_index] = (
                cache.layer_input[layer_index]
                .index_select(0, positions)
                .float()
                .norm(dim=-1)
            )

        for begin in range(0, rows, query_chunk):
            end = min(begin + query_chunk, rows)
            query_position = positions[begin:end].to(device)
            attention = attention_rows(query, key, query_position, scaling)
            message_norm = attention.float() * native_source_norm[:, None, :]
            transport = (
                attention.float() if signal is FlowSignal.ATTENTION else message_norm
            )

            gradient_slot = gradient_lookup.index_select(0, query_position)
            if bool((gradient_slot < 0).any()):
                raise ValueError("native edge rows lack target gradients")
            gradient = (
                gradients.head_output[layer_index]
                .index_select(1, gradient_slot.cpu())
                .to(device)
            )
            value_action = torch.einsum("hsd,hqd->hqs", value.float(), gradient.float())
            functional = attention.float() * value_action

            keep = coverage_mask(transport, coverage)
            row_total[layer_index, :, begin:end] = transport.sum(-1).cpu()
            row_retained[layer_index, :, begin:end] = (
                transport.masked_fill(~keep, 0).sum(-1).cpu()
            )
            head, local_query, source = torch.nonzero(keep, as_tuple=True)
            if not len(head):
                continue
            target = query_position.index_select(0, local_query)
            selected_attention = attention[head, local_query, source].float()
            selected_code = selected_attention[:, None] * value[head, source].float()
            clean_norm, _, _, _, _, _ = project_selected_messages(
                output,
                head,
                selected_code,
                selected_code,
                heads,
                materialize=False,
                gram=gram.to(device),
            )
            selected_functional = functional[head, local_query, source].float().cpu()
            chunk_budget = torch.zeros(end - begin)
            chunk_budget.index_add_(0, local_query.cpu(), clean_norm.cpu())
            row_budget[layer_index, begin:end] = chunk_budget
            row_net[layer_index, begin:end] = net_row_message_norm(
                output,
                head,
                local_query,
                selected_code,
                end - begin,
                heads,
            ).cpu()
            for values, destination in (
                (selected_functional, row_signed),
                (selected_functional.clamp_min(0), row_positive),
                (selected_functional.clamp_max(0), row_negative),
            ):
                chunk_sum = torch.zeros(end - begin)
                chunk_sum.index_add_(0, local_query.cpu(), values)
                destination[layer_index, begin:end] = chunk_sum

            columns["layer"].append(
                torch.full_like(source, layer_index, dtype=torch.int16).cpu()
            )
            columns["head"].append(head.to(torch.int16).cpu())
            columns["source"].append(source.to(torch.int32).cpu())
            columns["target"].append(target.to(torch.int32).cpu())
            columns["source_unit"].append(
                token_units.index_select(0, source).to(torch.int32).cpu()
            )
            columns["attention"].append(selected_attention.cpu())
            columns["transport"].append(
                transport[head, local_query, source].float().cpu()
            )
            columns["functional"].append(selected_functional)
            columns["message_norm"].append(clean_norm.cpu())
            columns["code"].append(selected_code.cpu())
        del query, key, value, output, gram, native_source_norm

    def concatenate(name: str, dtype: torch.dtype, width: int | None = None):
        values = columns[name]
        if values:
            return torch.cat(values)
        if width is not None:
            return torch.empty((0, width), dtype=dtype)
        return torch.empty(0, dtype=dtype)

    head_dim = int(
        model.model.layers[0].self_attn.v_proj.out_features
        // model.config.num_key_value_heads
    )
    edge_layer = concatenate("layer", torch.int16)
    count = len(edge_layer)
    nan = torch.full((count,), float("nan"))
    code = concatenate("code", torch.float32, head_dim)
    attention = concatenate("attention", torch.float32)
    functional = concatenate("functional", torch.float32)
    message_norm = concatenate("message_norm", torch.float32)
    edges = FlowEdges(
        edge_layer,
        concatenate("head", torch.int16),
        concatenate("source", torch.int32),
        concatenate("target", torch.int32),
        concatenate("source_unit", torch.int32),
        attention,
        attention.clone(),
        concatenate("transport", torch.float32),
        functional,
        functional.clone(),
        nan,
        nan.clone(),
        message_norm,
        message_norm.clone(),
        torch.zeros_like(message_norm),
        code,
        code.clone(),
        _empty_vector(count),
        _empty_vector(count),
        _empty_vector(count),
    )
    coherence = torch.where(row_budget > 0, row_net / row_budget, 0)
    nan_row = torch.full_like(row_budget, float("nan"))
    aggregation = RowAggregation(
        row_budget,
        row_net,
        coherence,
        row_signed,
        row_positive,
        row_negative,
        nan_row,
        nan_row.clone(),
    )
    return edges, row_total, row_retained, aggregation, residual_weight


def attach_cut_edge_codes(
    model,
    edges: FlowEdges,
    cut: ForwardCache,
    gradients: TargetGradients,
    cut_source_mask: Tensor,
) -> FlowEdges:
    """Evaluate root-cut attention/Value codes at native retained endpoints."""

    count = edges.count
    cut_attention = torch.empty(count)
    cut_functional = torch.empty(count)
    cut_norm = torch.empty(count)
    delta_norm = torch.empty(count)
    cut_code = torch.empty_like(edges.clean_code)
    device = model.get_input_embeddings().weight.device
    heads = int(model.config.num_attention_heads)
    gradient_lookup = {
        int(position): index
        for index, position in enumerate(gradients.position.tolist())
    }
    gram_cache = model_gram_cache(model)

    for layer_index, layer in enumerate(model.model.layers):
        selected = torch.nonzero(edges.layer == layer_index, as_tuple=False).flatten()
        if not len(selected):
            continue
        query, key, value = attention_qkv(
            model, layer_index, cut.layer_input[layer_index]
        )
        targets = edges.target.index_select(0, selected).long()
        unique_target, inverse = torch.unique(targets, sorted=True, return_inverse=True)
        head_dim = value.shape[-1]
        scaling = float(getattr(layer.self_attn, "scaling", head_dim**-0.5))
        probability = attention_rows(query, key, unique_target.to(device), scaling)
        head = edges.head.index_select(0, selected).long().to(device)
        source = edges.source.index_select(0, selected).long().to(device)
        local = inverse.to(device)
        selected_attention = probability[head, local, source].float()
        selected_code = selected_attention[:, None] * value[head, source].float()
        deleted = cut_source_mask.index_select(0, source.cpu()).to(device)
        selected_code[deleted] = 0
        gradient_slot = torch.tensor(
            [gradient_lookup[int(position)] for position in targets.tolist()],
            dtype=torch.long,
        )
        gradient = gradients.head_output[layer_index][head.cpu(), gradient_slot].to(
            device
        )
        functional = (selected_code * gradient.float()).sum(-1)
        output = layer.self_attn.o_proj.weight.detach()
        gram = gram_cache.get(layer_index)
        if gram is None:
            gram = output_gram(output, heads, head_dim)
            gram_cache[layer_index] = gram
        _, current_norm, current_delta, _, _, _ = project_selected_messages(
            output,
            head,
            edges.clean_code.index_select(0, selected).to(device),
            selected_code,
            heads,
            materialize=False,
            gram=gram.to(device),
        )
        cut_attention[selected] = selected_attention.cpu()
        cut_functional[selected] = functional.cpu()
        cut_norm[selected] = current_norm.cpu()
        delta_norm[selected] = current_delta.cpu()
        cut_code[selected] = selected_code.cpu()
        del query, key, value, probability, output, gram

    return replace(
        edges,
        attention_corrupt=cut_attention,
        corrupt_target_score=cut_functional,
        corrupt_message_norm=cut_norm,
        delta_message_norm=delta_norm,
        corrupt_code=cut_code,
    )


def native_flow_screen(
    model,
    world: NativeWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    *,
    carrier_scope: str,
    coverage: float,
    query_chunk: int,
) -> tuple[PairedFlow, TargetGradients]:
    """Build one native transport graph and independent functional ledger."""

    signal = FlowSignal(signal)
    prefix = world.prefix(target)
    clean = baseline_forward(
        model,
        prefix.token_ids,
        prefix.response_start,
        checkpoint_layers=range(len(model.model.layers)),
        checkpoint_stages=True,
        attention_query_chunk=query_chunk,
    )
    positions = represented_positions(prefix, target, carrier_scope)
    root_position = prefix.units.positions(prefix.evidence_unit_id)
    gradient_position = torch.unique(torch.cat((positions, root_position)), sorted=True)
    gradients = native_target_gradients(
        model,
        clean,
        target,
        gradient_position,
        query_chunk=query_chunk,
    )
    edges, total, retained, aggregation, residual_weight = capture_native_edges(
        model,
        clean,
        prefix.units,
        positions,
        signal,
        gradients,
        coverage=coverage,
        query_chunk=query_chunk,
    )
    clean_margin = margin(model, clean, target)
    return (
        PairedFlow(
            signal,
            target,
            clean_margin,
            clean_margin,
            edges,
            positions,
            total,
            retained,
            aggregation,
            None,
            clean,
            clean,
            residual_weight,
        ),
        gradients,
    )
