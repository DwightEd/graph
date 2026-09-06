"""Native attention/message transport with independent target functionality."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from experiments.common.llama_message_intervention import ForwardCache, baseline_forward

from .attribution import TargetGradients, native_target_gradients
from .flow import (
    SOURCE_LOCATION_BUCKET_NAMES,
    FlowEdges,
    FlowSignal,
    PairedFlow,
    RowAggregation,
    SourceLocationBuckets,
    attention_qkv,
    attention_rows,
    margin,
    net_row_message_norm,
    project_selected_messages,
)
from .message_norm import model_gram_cache, output_gram, source_norm
from .native_world import NativeWorld
from .worlds import SourceUnits, TargetContrast

DEFAULT_MAX_EDGES_PER_HEAD_ROW = 8
PROMPT_EVIDENCE_BUCKET = 0
OTHER_PROMPT_BUCKET = 1
REMOTE_RESPONSE_BUCKET = 2
RECENT_LOCAL_BUCKET = 3


def represented_positions(
    world: NativeWorld,
    target: TargetContrast,
    scope: str,
    *,
    max_rows: int | None = None,
) -> Tensor:
    """Keep a contiguous destination window ending at the audited query.

    ``scope`` chooses eligible destinations; ``max_rows`` bounds the most
    recent ones. Every causal token remains eligible as a message source.
    Lineage entering from an unrepresented response state is marked unknown
    by the route model rather than treated as a fully traced history path.
    """

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    if scope == "all":
        begin, end = 0, target.query_position + 1
    elif scope == "response":
        begin, end = world.response_start - 1, target.query_position + 1
    else:
        raise ValueError("carrier_scope must be 'response' or 'all'")
    if max_rows is not None:
        begin = max(begin, end - max_rows)
    return torch.arange(begin, end)


def _empty_vector(count: int) -> Tensor:
    return torch.empty((count, 0), dtype=torch.float32)


def _budgeted_edge_mask(
    transport: Tensor,
    functional: Tensor,
    coverage: float,
    max_edges_per_head_row: int,
) -> Tensor:
    """Keep bounded transport- and function-selected edges for every head row.

    The transport branch keeps the smallest prefix of its in-budget top-k set
    that reaches the requested fraction of the *full* row mass.  If k edges
    cannot reach that fraction, it keeps those k and leaves the missing mass
    unrepresented.  The functional branch independently keeps up to k
    non-zero edges by absolute target action.  Their union is therefore
    bounded by ``2 * k`` without averaging heads.
    """

    if max_edges_per_head_row < 1:
        raise ValueError("max_edges_per_head_row must be positive")
    if not 0 < coverage <= 1:
        raise ValueError("edge coverage must lie in (0,1]")
    if transport.shape != functional.shape or transport.ndim != 3:
        raise ValueError("transport and functional rows must share [head,row,source]")
    if not bool(torch.isfinite(transport).all()):
        raise FloatingPointError("edge transport contains a non-finite value")
    if bool((transport < 0).any()):
        raise ValueError("edge transport must be non-negative")

    sources = transport.shape[-1]
    count = min(max_edges_per_head_row, sources)
    transport_value, transport_index = torch.topk(
        transport, count, dim=-1, largest=True, sorted=True
    )
    full_mass = transport.sum(dim=-1, keepdim=True)
    mass_before = transport_value.cumsum(dim=-1) - transport_value
    transport_selected = (mass_before < coverage * full_mass) & (transport_value > 0)

    functional_value, functional_index = torch.topk(
        functional.abs(), count, dim=-1, largest=True, sorted=False
    )
    functional_selected = functional_value > 0
    keep = torch.zeros_like(transport, dtype=torch.bool)
    keep.scatter_(dim=-1, index=transport_index, src=transport_selected)
    keep.scatter_(
        dim=-1,
        index=functional_index,
        src=keep.gather(-1, functional_index) | functional_selected,
    )
    return keep


class SourceLocationAccumulator:
    """Stream exact full attention rows into a bounded location ledger.

    Only the four bucket totals and their strongest source are retained.  The
    full lower-triangular attention/message tensors stay chunk-local, so the
    persistent memory cost is ``O(layer * head * row * bucket)``.
    """

    def __init__(
        self,
        *,
        layers: int,
        heads: int,
        rows: int,
        token_unit_id: Tensor,
        response_start: int,
        evidence_unit_id: tuple[int, ...],
        local_window: int,
        with_action: bool,
        device: torch.device,
    ) -> None:
        bucket_count = len(SOURCE_LOCATION_BUCKET_NAMES)
        shape = (layers, heads, rows, bucket_count)
        self.local_window = local_window
        self.token_unit_id = token_unit_id.to(device)
        self.source_position = torch.arange(len(token_unit_id), device=device)
        self.response_start = response_start
        self.with_action = with_action
        self.attention = torch.zeros(shape)
        self.transport = torch.zeros(shape)
        self.downstream_action = torch.zeros(shape) if with_action else None
        self.winner_position = torch.full(shape, -1, dtype=torch.int32)
        self.winner_unit = torch.full(shape, -1, dtype=torch.int32)
        self.winner_attention = torch.zeros(shape)
        self.winner_transport = torch.zeros(shape)
        self.winner_action = torch.zeros(shape) if with_action else None

        prompt_evidence = torch.zeros(
            len(token_unit_id), dtype=torch.bool, device=device
        )
        for unit_id in evidence_unit_id:
            prompt_evidence |= self.token_unit_id == unit_id
        self.prompt_evidence = prompt_evidence & (self.source_position < response_start)

    def _bucket_mask(self, query_position: Tensor) -> Tensor:
        distance = query_position[:, None] - self.source_position[None, :]
        causal = distance >= 0
        response = self.source_position[None, :] >= self.response_start
        prompt = ~response
        evidence = self.prompt_evidence[None, :]
        return torch.stack(
            (
                prompt & evidence & causal,
                prompt & ~evidence & causal,
                response & (distance > self.local_window),
                response & (distance >= 0) & (distance <= self.local_window),
            )
        )

    def observe(
        self,
        layer: int,
        begin: int,
        query_position: Tensor,
        attention: Tensor,
        message_transport: Tensor,
        downstream_action: Tensor | None = None,
    ) -> None:
        """Aggregate one ``[head, query, source]`` chunk before pruning."""

        if self.with_action != (downstream_action is not None):
            raise ValueError("downstream action availability changed within capture")
        end = begin + len(query_position)
        mask = self._bucket_mask(query_position)
        self.attention[layer, :, begin:end] = torch.einsum(
            "hqs,bqs->hqb", attention.float(), mask.float()
        ).cpu()
        self.transport[layer, :, begin:end] = torch.einsum(
            "hqs,bqs->hqb", message_transport.float(), mask.float()
        ).cpu()
        if downstream_action is not None:
            self.downstream_action[layer, :, begin:end] = torch.einsum(
                "hqs,bqs->hqb", downstream_action.float(), mask.float()
            ).cpu()

        heads = attention.shape[0]
        for bucket in range(len(SOURCE_LOCATION_BUCKET_NAMES)):
            eligible = mask[bucket][None].expand(heads, -1, -1)
            peak, winner = message_transport.float().masked_fill(~eligible, -1).max(-1)
            present = eligible.any(-1)
            winner_attention = attention.gather(-1, winner[..., None]).squeeze(-1)
            winner_unit = self.token_unit_id.index_select(0, winner.flatten()).view_as(
                winner
            )
            self.winner_position[layer, :, begin:end, bucket] = (
                torch.where(present, winner, -1).to(torch.int32).cpu()
            )
            self.winner_unit[layer, :, begin:end, bucket] = (
                torch.where(present, winner_unit, -1).to(torch.int32).cpu()
            )
            self.winner_attention[layer, :, begin:end, bucket] = (
                torch.where(present, winner_attention, 0).float().cpu()
            )
            self.winner_transport[layer, :, begin:end, bucket] = torch.where(
                present, peak, 0
            ).cpu()
            if downstream_action is not None:
                winner_action = downstream_action.gather(-1, winner[..., None]).squeeze(
                    -1
                )
                self.winner_action[layer, :, begin:end, bucket] = (
                    torch.where(present, winner_action, 0).float().cpu()
                )

    def summary(self) -> SourceLocationBuckets:
        return SourceLocationBuckets(
            local_window=self.local_window,
            attention=self.attention,
            transport=self.transport,
            downstream_action=self.downstream_action,
            source_position=self.winner_position,
            source_unit_id=self.winner_unit,
            source_attention=self.winner_attention,
            source_transport=self.winner_transport,
            source_downstream_action=self.winner_action,
        )


@torch.no_grad()
def capture_source_location_buckets(
    model,
    cache: ForwardCache,
    units: SourceUnits,
    positions: Tensor,
    *,
    response_start: int,
    evidence_unit_id: tuple[int, ...],
    local_window: int,
    query_chunk: int,
    gradients: TargetGradients | None = None,
) -> SourceLocationBuckets:
    """Capture full-row location buckets from a clean cache, optionally functional."""

    if query_chunk < 1:
        raise ValueError("query chunk must be positive")
    if local_window < 1:
        raise ValueError("local_window must be positive")
    device = model.get_input_embeddings().weight.device
    heads = int(model.config.num_attention_heads)
    accumulator = SourceLocationAccumulator(
        layers=cache.layer_count,
        heads=heads,
        rows=len(positions),
        token_unit_id=units.token_unit_id,
        response_start=response_start,
        evidence_unit_id=evidence_unit_id,
        local_window=local_window,
        with_action=gradients is not None,
        device=device,
    )
    gradient_lookup = None
    if gradients is not None:
        gradient_lookup = torch.full(
            (len(units.token_unit_id),), -1, dtype=torch.long, device=device
        )
        gradient_lookup[gradients.position.to(device)] = torch.arange(
            len(gradients.position), device=device
        )
    gram_cache = model_gram_cache(model)
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
        for begin in range(0, len(positions), query_chunk):
            end = min(begin + query_chunk, len(positions))
            query_position = positions[begin:end].to(device)
            attention = attention_rows(query, key, query_position, scaling)
            action = None
            if gradients is not None:
                gradient_slot = gradient_lookup.index_select(0, query_position)
                if bool((gradient_slot < 0).any()):
                    raise ValueError("source-location rows lack target gradients")
                gradient = (
                    gradients.head_output[layer_index]
                    .index_select(1, gradient_slot.cpu())
                    .to(device)
                )
                action = attention.float() * torch.einsum(
                    "hsd,hqd->hqs", value.float(), gradient.float()
                )
            accumulator.observe(
                layer_index,
                begin,
                query_position,
                attention,
                attention.float() * native_source_norm[:, None, :],
                action,
            )
        del query, key, value, output, gram, native_source_norm
    return accumulator.summary()


def capture_native_edges(
    model,
    cache: ForwardCache,
    units: SourceUnits,
    positions: Tensor,
    signal: FlowSignal,
    gradients: TargetGradients,
    *,
    response_start: int,
    evidence_unit_id: tuple[int, ...],
    local_window: int,
    coverage: float,
    query_chunk: int,
    max_edges_per_head_row: int = DEFAULT_MAX_EDGES_PER_HEAD_ROW,
) -> tuple[
    FlowEdges,
    Tensor,
    Tensor,
    RowAggregation,
    Tensor,
    SourceLocationBuckets,
]:
    """Capture native transport and signed functionality without averaging."""

    if query_chunk < 1:
        raise ValueError("query chunk must be positive")
    if max_edges_per_head_row < 1:
        raise ValueError("max_edges_per_head_row must be positive")
    if local_window < 1:
        raise ValueError("local_window must be positive")
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
        )
    }
    device = model.get_input_embeddings().weight.device
    token_units = units.token_unit_id.to(device)
    source_location = SourceLocationAccumulator(
        layers=layers,
        heads=heads,
        rows=rows,
        token_unit_id=units.token_unit_id,
        response_start=response_start,
        evidence_unit_id=evidence_unit_id,
        local_window=local_window,
        with_action=True,
        device=device,
    )
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

            source_location.observe(
                layer_index,
                begin,
                query_position,
                attention,
                message_norm,
                functional,
            )

            keep = _budgeted_edge_mask(
                transport,
                functional,
                coverage,
                max_edges_per_head_row,
            )
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
        del query, key, value, output, gram, native_source_norm

    def concatenate(name: str, dtype: torch.dtype, width: int | None = None):
        values = columns[name]
        if values:
            return torch.cat(values)
        if width is not None:
            return torch.empty((0, width), dtype=dtype)
        return torch.empty(0, dtype=dtype)

    edge_layer = concatenate("layer", torch.int16)
    count = len(edge_layer)
    nan = torch.full((count,), float("nan"))
    attention = concatenate("attention", torch.float32)
    functional = concatenate("functional", torch.float32)
    message_norm = concatenate("message_norm", torch.float32)
    empty = _empty_vector(count)
    edges = FlowEdges(
        edge_layer,
        concatenate("head", torch.int16),
        concatenate("source", torch.int32),
        concatenate("target", torch.int32),
        concatenate("source_unit", torch.int32),
        attention,
        attention,
        concatenate("transport", torch.float32),
        functional,
        functional,
        nan,
        nan,
        message_norm,
        message_norm,
        torch.zeros_like(message_norm),
        empty,
        empty,
        empty,
        empty,
        empty,
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
    return (
        edges,
        row_total,
        row_retained,
        aggregation,
        residual_weight,
        source_location.summary(),
    )


def attach_cut_edge_codes(
    model,
    edges: FlowEdges,
    cut: ForwardCache,
    gradients: TargetGradients,
    cut_source_mask: Tensor,
    *,
    clean_cache: ForwardCache | None = None,
    query_chunk: int | None = None,
) -> FlowEdges:
    """Materialize clean/root-cut codes only for retained discovery endpoints."""

    if query_chunk is None:
        query_chunk = cut.attention_query_chunk
    if query_chunk is not None and query_chunk < 1:
        raise ValueError("query chunk must be positive")
    count = edges.count
    cut_attention = torch.empty(count)
    cut_functional = torch.empty(count)
    clean_functional = torch.empty(count)
    clean_attention = torch.empty(count)
    clean_norm = torch.empty(count)
    cut_norm = torch.empty(count)
    delta_norm = torch.empty(count)
    device = model.get_input_embeddings().weight.device
    heads = int(model.config.num_attention_heads)
    if clean_cache is None:
        raise ValueError("clean_cache is required to materialize discovery edge codes")
    head_dim = int(
        model.model.layers[0].self_attn.v_proj.out_features
        // model.config.num_key_value_heads
    )
    clean_code = torch.empty((count, head_dim), dtype=torch.float32)
    cut_code = torch.empty_like(clean_code)
    gradient_lookup = {
        int(position): index
        for index, position in enumerate(gradients.position.tolist())
    }
    gram_cache = model_gram_cache(model)

    for layer_index, layer in enumerate(model.model.layers):
        selected = torch.nonzero(edges.layer == layer_index, as_tuple=False).flatten()
        if not len(selected):
            continue
        clean_query, clean_key, clean_value = attention_qkv(
            model, layer_index, clean_cache.layer_input[layer_index]
        )
        cut_query, cut_key, cut_value = attention_qkv(
            model, layer_index, cut.layer_input[layer_index]
        )
        targets = edges.target.index_select(0, selected).long()
        unique_target, inverse = torch.unique(targets, sorted=True, return_inverse=True)
        head_dim = cut_value.shape[-1]
        scaling = float(getattr(layer.self_attn, "scaling", head_dim**-0.5))
        output = layer.self_attn.o_proj.weight.detach()
        gram = gram_cache.get(layer_index)
        if gram is None:
            gram = output_gram(output, heads, head_dim)
            gram_cache[layer_index] = gram
        chunk = len(unique_target) if query_chunk is None else query_chunk
        for begin in range(0, len(unique_target), chunk):
            end = min(begin + chunk, len(unique_target))
            chunk_position = unique_target[begin:end].to(device)
            clean_probability = attention_rows(
                clean_query,
                clean_key,
                chunk_position,
                scaling,
            )
            cut_probability = attention_rows(
                cut_query,
                cut_key,
                chunk_position,
                scaling,
            )
            local_edge = torch.nonzero(
                (inverse >= begin) & (inverse < end), as_tuple=False
            ).flatten()
            edge_index = selected.index_select(0, local_edge)
            head = edges.head.index_select(0, edge_index).long().to(device)
            source = edges.source.index_select(0, edge_index).long().to(device)
            local = inverse.index_select(0, local_edge).to(device) - begin
            selected_clean_attention = clean_probability[head, local, source].float()
            selected_cut_attention = cut_probability[head, local, source].float()
            selected_clean_code = (
                selected_clean_attention[:, None] * clean_value[head, source].float()
            )
            selected_cut_code = (
                selected_cut_attention[:, None] * cut_value[head, source].float()
            )
            deleted = cut_source_mask.index_select(0, source.cpu()).to(device)
            selected_cut_code = selected_cut_code.masked_fill(deleted[:, None], 0)
            chunk_targets = targets.index_select(0, local_edge)
            gradient_slot = torch.tensor(
                [gradient_lookup[int(position)] for position in chunk_targets.tolist()],
                dtype=torch.long,
            )
            gradient = gradients.head_output[layer_index][head.cpu(), gradient_slot].to(
                device
            )
            clean_action = (selected_clean_code * gradient.float()).sum(-1)
            cut_action = (selected_cut_code * gradient.float()).sum(-1)
            current_clean, current_cut, current_delta, _, _, _ = (
                project_selected_messages(
                    output,
                    head,
                    selected_clean_code,
                    selected_cut_code,
                    heads,
                    materialize=False,
                    gram=gram.to(device),
                )
            )
            clean_attention[edge_index] = selected_clean_attention.cpu()
            cut_attention[edge_index] = selected_cut_attention.cpu()
            clean_functional[edge_index] = clean_action.cpu()
            cut_functional[edge_index] = cut_action.cpu()
            clean_norm[edge_index] = current_clean.cpu()
            cut_norm[edge_index] = current_cut.cpu()
            delta_norm[edge_index] = current_delta.cpu()
            clean_code[edge_index] = selected_clean_code.cpu()
            cut_code[edge_index] = selected_cut_code.cpu()
            del clean_probability, cut_probability
        del clean_query, clean_key, clean_value
        del cut_query, cut_key, cut_value, output, gram

    return replace(
        edges,
        attention_clean=clean_attention,
        attention_corrupt=cut_attention,
        clean_target_score=clean_functional,
        corrupt_target_score=cut_functional,
        clean_message_norm=clean_norm,
        corrupt_message_norm=cut_norm,
        delta_message_norm=delta_norm,
        clean_code=clean_code,
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
    max_rows: int | None = None,
    max_edges_per_head_row: int = DEFAULT_MAX_EDGES_PER_HEAD_ROW,
    local_window: int = 10,
) -> tuple[PairedFlow, TargetGradients]:
    """Build one native transport graph and independent functional ledger."""

    signal = FlowSignal(signal)
    prefix = world.prefix(target)
    positions = represented_positions(
        prefix,
        target,
        carrier_scope,
        max_rows=max_rows,
    )
    clean = baseline_forward(
        model,
        prefix.token_ids,
        prefix.response_start,
        checkpoint_layers=range(len(model.model.layers)),
        checkpoint_stages=True,
        attention_query_chunk=query_chunk,
        fixed_runner={target.query_position: target.negative_token_id},
    )
    gradients = native_target_gradients(
        model,
        clean,
        target,
        positions,
        query_chunk=query_chunk,
    )
    (
        edges,
        total,
        retained,
        aggregation,
        residual_weight,
        source_location,
    ) = capture_native_edges(
        model,
        clean,
        prefix.units,
        positions,
        signal,
        gradients,
        response_start=prefix.response_start,
        evidence_unit_id=prefix.evidence_unit_id,
        local_window=local_window,
        coverage=coverage,
        query_chunk=query_chunk,
        max_edges_per_head_row=max_edges_per_head_row,
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
            source_location=source_location,
        ),
        gradients,
    )
