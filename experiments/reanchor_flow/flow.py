"""Attention-only and true-message edge backends for one paired ETCC target."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from experiments.common.llama_message_intervention import (
    ForwardCache,
    apply_rotary,
    baseline_forward,
    position_embeddings,
    repeat_kv,
)

from .attribution import (
    TargetGradients,
    contrast_direction,
    integrated_target_gradients,
)
from .message_norm import model_gram_cache, output_gram
from .worlds import PairedWorld, TargetContrast


class FlowSignal(str, Enum):
    """The data allowed to rank graph edges."""

    ATTENTION = "attention"
    MESSAGE = "message"


@dataclass(frozen=True)
class FlowEdges:
    """Sparse exact endpoints retained by row-wise coverage."""

    layer: Tensor
    head: Tensor
    source: Tensor
    target: Tensor
    source_unit: Tensor
    attention_clean: Tensor
    attention_corrupt: Tensor
    score: Tensor
    clean_target_score: Tensor
    corrupt_target_score: Tensor
    selector_score: Tensor
    content_score: Tensor
    clean_message_norm: Tensor
    corrupt_message_norm: Tensor
    delta_message_norm: Tensor
    clean_code: Tensor
    corrupt_code: Tensor
    clean_message_vector: Tensor
    corrupt_message_vector: Tensor
    delta_message_vector: Tensor

    @property
    def count(self) -> int:
        return len(self.layer)

    def select(self, index) -> "FlowEdges":
        return FlowEdges(
            self.layer[index],
            self.head[index],
            self.source[index],
            self.target[index],
            self.source_unit[index],
            self.attention_clean[index],
            self.attention_corrupt[index],
            self.score[index],
            self.clean_target_score[index],
            self.corrupt_target_score[index],
            self.selector_score[index],
            self.content_score[index],
            self.clean_message_norm[index],
            self.corrupt_message_norm[index],
            self.delta_message_norm[index],
            self.clean_code[index],
            self.corrupt_code[index],
            self.clean_message_vector[index],
            self.corrupt_message_vector[index],
            self.delta_message_vector[index],
        )


@dataclass(frozen=True)
class StageTrace:
    """Source-conditioned state, attention-write, and MLP-write accounting."""

    position: Tensor
    state_delta_norm: Tensor
    state_score: Tensor
    attention_delta_norm: Tensor
    attention_score: Tensor
    mlp_delta_norm: Tensor
    mlp_score: Tensor


@dataclass(frozen=True)
class RowAggregation:
    """Retained message aggregation before any source/head mean."""

    message_budget: Tensor
    net_message_norm: Tensor
    coherence: Tensor
    signed_score: Tensor
    positive_score: Tensor
    negative_score: Tensor
    selector_score: Tensor
    content_score: Tensor


@dataclass(frozen=True)
class PairedFlow:
    signal: FlowSignal
    target: TargetContrast
    clean_margin: float
    corrupt_margin: float
    edges: FlowEdges
    row_position: Tensor
    row_total: Tensor
    row_retained: Tensor
    aggregation: RowAggregation
    stages: StageTrace | None
    clean_cache: ForwardCache
    corrupt_cache: ForwardCache

    @property
    def pair_effect(self) -> float:
        return self.clean_margin - self.corrupt_margin


@torch.no_grad()
def attention_qkv(
    model,
    layer_index: int,
    state: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute native GQA Q/K/V from one cached layer input."""

    device = model.get_input_embeddings().weight.device
    layer = model.model.layers[layer_index]
    module = layer.self_attn
    hidden = layer.input_layernorm(state.to(device)[None])
    tokens = hidden.shape[1]
    head_dim = int(
        getattr(
            module,
            "head_dim",
            module.q_proj.out_features // module.config.num_attention_heads,
        )
    )
    heads = module.q_proj.out_features // head_dim
    kv_heads = module.k_proj.out_features // head_dim
    query = module.q_proj(hidden).view(1, tokens, heads, head_dim).transpose(1, 2)
    key = module.k_proj(hidden).view(1, tokens, kv_heads, head_dim).transpose(1, 2)
    value = module.v_proj(hidden).view(1, tokens, kv_heads, head_dim).transpose(1, 2)
    query, key = apply_rotary(query, key, *position_embeddings(model, hidden))
    return query[0], repeat_kv(key, module.num_key_value_groups)[0], repeat_kv(
        value, module.num_key_value_groups
    )[0]


def attention_rows(
    query: Tensor,
    key: Tensor,
    positions: Tensor,
    scaling: float,
) -> Tensor:
    score = torch.matmul(
        query.index_select(1, positions), key.transpose(1, 2)
    ) * scaling
    sources = torch.arange(key.shape[1], device=key.device)
    future = sources[None] > positions[:, None]
    score = score.masked_fill(future[None], torch.finfo(score.dtype).min)
    return score.softmax(dim=-1, dtype=torch.float32).to(query.dtype)


def coverage_mask(weight: Tensor, coverage: float) -> Tensor:
    """Keep the smallest per-row edge set reaching the requested mass."""

    if not 0 < coverage <= 1:
        raise ValueError("edge coverage must lie in (0,1]")
    if not bool(torch.isfinite(weight).all()):
        raise FloatingPointError("edge weights contain a non-finite value")
    if bool((weight < 0).any()):
        raise ValueError("coverage weights must be non-negative")
    ordered, index = weight.sort(dim=-1, descending=True)
    total = ordered.sum(-1, keepdim=True)
    before = ordered.cumsum(-1) - ordered
    ordered_keep = (before < coverage * total) & (ordered > 0)
    keep = torch.zeros_like(ordered_keep)
    keep.scatter_(-1, index, ordered_keep)
    return keep


def margin(model, cache: ForwardCache, target: TargetContrast) -> float:
    direction, bias = contrast_direction(model, target)
    state = cache.final_hidden[target.query_position].to(direction.device)
    return float(torch.dot(state.float(), direction) + bias)


def stage_trace(
    clean: ForwardCache,
    corrupt: ForwardCache,
    gradients: TargetGradients,
) -> StageTrace:
    position = gradients.position.long()
    state_norm, state_score = [], []
    attention_norm, attention_score = [], []
    mlp_norm, mlp_score = [], []
    for layer in range(clean.layer_count):
        clean_state = clean.layer_input[layer].index_select(0, position)
        corrupt_state = corrupt.layer_input[layer].index_select(0, position)
        state_delta = clean_state.float() - corrupt_state.float()
        state_norm.append(state_delta.norm(dim=-1))
        state_score.append((state_delta * gradients.layer_input[layer]).sum(-1))

        clean_attention = clean.attention_write[layer].index_select(0, position)
        corrupt_attention = corrupt.attention_write[layer].index_select(0, position)
        attention_delta = clean_attention.float() - corrupt_attention.float()
        attention_norm.append(attention_delta.norm(dim=-1))
        attention_score.append(
            (attention_delta * gradients.attention_write[layer]).sum(-1)
        )

        clean_mlp = clean.mlp_write[layer].index_select(0, position)
        corrupt_mlp = corrupt.mlp_write[layer].index_select(0, position)
        mlp_delta = clean_mlp.float() - corrupt_mlp.float()
        mlp_norm.append(mlp_delta.norm(dim=-1))
        mlp_score.append((mlp_delta * gradients.mlp_write[layer]).sum(-1))
    return StageTrace(
        position,
        torch.stack(state_norm),
        torch.stack(state_score),
        torch.stack(attention_norm),
        torch.stack(attention_score),
        torch.stack(mlp_norm),
        torch.stack(mlp_score),
    )


def project_selected_messages(
    output_weight: Tensor,
    head_index: Tensor,
    clean_code: Tensor,
    corrupt_code: Tensor,
    heads: int,
    *,
    materialize: bool,
    gram: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply each selected head's ``W_O`` block without repeating the matrix."""

    count, head_dim = clean_code.shape
    hidden = output_weight.shape[0]
    if output_weight.shape[1] != heads * head_dim:
        raise ValueError("selected message code does not match the W_O head blocks")
    clean_norm = torch.empty(
        count, device=clean_code.device, dtype=torch.float32
    )
    corrupt_norm = torch.empty_like(clean_norm)
    delta_norm = torch.empty_like(clean_norm)
    width = hidden if materialize else 0
    clean_output = torch.empty(
        count,
        width,
        device=clean_code.device,
        dtype=torch.float32,
    )
    corrupt_output = torch.empty_like(clean_output)
    delta_output = torch.empty_like(clean_output)
    for head in torch.unique(head_index).tolist():
        selected = torch.nonzero(head_index == head, as_tuple=False).flatten()
        clean_selected = clean_code.index_select(0, selected)
        corrupt_selected = corrupt_code.index_select(0, selected)
        delta_selected = clean_selected - corrupt_selected
        if materialize:
            block = output_weight[
                :, head * head_dim : (head + 1) * head_dim
            ].float()
            clean_vector = clean_selected @ block.T
            corrupt_vector = corrupt_selected @ block.T
            delta_vector = clean_vector - corrupt_vector
            clean_norm[selected] = clean_vector.norm(dim=-1)
            corrupt_norm[selected] = corrupt_vector.norm(dim=-1)
            delta_norm[selected] = delta_vector.norm(dim=-1)
            clean_output[selected] = clean_vector
            corrupt_output[selected] = corrupt_vector
            delta_output[selected] = delta_vector
        else:
            if gram is None:
                raise ValueError("W_O Gram matrices are required for compact norms")
            current_gram = gram[head]
            for code, destination in (
                (clean_selected, clean_norm),
                (corrupt_selected, corrupt_norm),
                (delta_selected, delta_norm),
            ):
                squared = torch.einsum(
                    "nd,de,ne->n", code, current_gram, code
                )
                destination[selected] = squared.clamp_min(0).sqrt()
    return (
        clean_norm,
        corrupt_norm,
        delta_norm,
        clean_output,
        corrupt_output,
        delta_output,
    )


def net_row_message_norm(
    output_weight: Tensor,
    head_index: Tensor,
    row_index: Tensor,
    delta_code: Tensor,
    rows: int,
    heads: int,
) -> Tensor:
    """Norm of the retained cross-head message sum for each destination row."""

    head_dim = delta_code.shape[1]
    code_sum = torch.zeros(
        rows,
        heads,
        head_dim,
        device=delta_code.device,
        dtype=torch.float32,
    )
    flat = code_sum.view(rows * heads, head_dim)
    flat_index = row_index * heads + head_index
    flat.index_add_(0, flat_index, delta_code.float())
    net = torch.zeros(
        rows,
        output_weight.shape[0],
        device=delta_code.device,
        dtype=torch.float32,
    )
    for head in range(heads):
        block = output_weight[
            :, head * head_dim : (head + 1) * head_dim
        ].float()
        net += code_sum[:, head] @ block.T
    return net.norm(dim=-1)


def capture_edges(
    model,
    clean: ForwardCache,
    corrupt: ForwardCache,
    token_unit_id: Tensor,
    positions: Tensor,
    signal: FlowSignal,
    gradients: TargetGradients | None,
    *,
    coverage: float,
    query_chunk: int,
    materialize_messages: bool,
) -> tuple[FlowEdges, Tensor, Tensor, RowAggregation]:
    """Build one sparse layer/head/source/target table without averaging."""

    if query_chunk < 1:
        raise ValueError("query chunk must be positive")
    layers = clean.layer_count
    heads = int(model.config.num_attention_heads)
    row_total = torch.zeros(layers, heads, len(positions))
    row_retained = torch.zeros_like(row_total)
    row_budget = torch.zeros(layers, len(positions))
    row_net = torch.zeros_like(row_budget)
    target_rows = torch.full_like(row_budget, float("nan"))
    positive_rows = torch.full_like(row_budget, float("nan"))
    negative_rows = torch.full_like(row_budget, float("nan"))
    selector_rows = torch.full_like(row_budget, float("nan"))
    content_rows = torch.full_like(row_budget, float("nan"))
    if signal is FlowSignal.MESSAGE:
        target_rows.zero_()
        positive_rows.zero_()
        negative_rows.zero_()
        selector_rows.zero_()
        content_rows.zero_()
    columns: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "layer",
            "head",
            "source",
            "target",
            "source_unit",
            "attention_clean",
            "attention_corrupt",
            "score",
            "clean_target_score",
            "corrupt_target_score",
            "selector_score",
            "content_score",
            "clean_message_norm",
            "corrupt_message_norm",
            "delta_message_norm",
            "clean_code",
            "corrupt_code",
            "clean_message_vector",
            "corrupt_message_vector",
            "delta_message_vector",
        )
    }
    device = model.get_input_embeddings().weight.device
    units = token_unit_id.to(device)
    gram_cache = model_gram_cache(model)
    gradient_lookup = None
    if gradients is not None:
        gradient_lookup = torch.full(
            (len(token_unit_id),), -1, dtype=torch.long, device=device
        )
        gradient_lookup[gradients.position.to(device)] = torch.arange(
            len(gradients.position), device=device
        )
    for layer_index, layer in enumerate(model.model.layers):
        clean_q, clean_k, clean_v = attention_qkv(
            model, layer_index, clean.layer_input[layer_index]
        )
        corrupt_q, corrupt_k, corrupt_v = attention_qkv(
            model, layer_index, corrupt.layer_input[layer_index]
        )
        head_dim = clean_v.shape[-1]
        output = layer.self_attn.o_proj.weight.detach()
        gram = None
        if not materialize_messages:
            gram = gram_cache.get(layer_index)
            if gram is None:
                gram = output_gram(output, heads, head_dim)
                gram_cache[layer_index] = gram
            gram = gram.to(device)
        scaling = float(getattr(layer.self_attn, "scaling", head_dim**-0.5))
        for begin in range(0, len(positions), query_chunk):
            end = min(begin + query_chunk, len(positions))
            query_position = positions[begin:end].to(device)
            clean_a = attention_rows(clean_q, clean_k, query_position, scaling)
            corrupt_a = attention_rows(corrupt_q, corrupt_k, query_position, scaling)

            clean_target = corrupt_target = selector = content = None
            if signal is FlowSignal.ATTENTION:
                primary = clean_a
            else:
                assert gradients is not None
                assert gradient_lookup is not None
                gradient_slot = gradient_lookup.index_select(0, query_position)
                if bool((gradient_slot < 0).any()):
                    raise ValueError("message rows lack target-gradient positions")
                gradient = gradients.head_output[
                    layer_index
                ].index_select(
                    1, gradient_slot.cpu()
                ).to(device)
                clean_value = torch.einsum("hsd,hqd->hqs", clean_v.float(), gradient)
                corrupt_value = torch.einsum(
                    "hsd,hqd->hqs", corrupt_v.float(), gradient
                )
                pp = clean_a * clean_value
                pm = clean_a * corrupt_value
                mp = corrupt_a * clean_value
                mm = corrupt_a * corrupt_value
                primary = pp - mm
                clean_target = pp
                corrupt_target = mm
                selector = 0.5 * (pp + pm - mp - mm)
                content = 0.5 * (pp - pm + mp - mm)

            weight = primary.abs() if signal is FlowSignal.MESSAGE else primary
            keep = coverage_mask(weight, coverage)
            row_total[layer_index, :, begin:end] = weight.sum(-1).cpu()
            row_retained[layer_index, :, begin:end] = (
                weight.masked_fill(~keep, 0).sum(-1).cpu()
            )
            head_index, local_query, source = torch.nonzero(
                keep, as_tuple=True
            )
            if not len(head_index):
                continue
            target = query_position.index_select(0, local_query)
            clean_weight = clean_a[head_index, local_query, source]
            corrupt_weight = corrupt_a[head_index, local_query, source]
            clean_code = clean_weight[:, None] * clean_v[head_index, source].float()
            corrupt_code = (
                corrupt_weight[:, None] * corrupt_v[head_index, source].float()
            )
            delta_code = clean_code - corrupt_code
            (
                clean_norm,
                corrupt_norm,
                delta_norm,
                clean_vector,
                corrupt_vector,
                delta_vector,
            ) = project_selected_messages(
                output,
                head_index,
                clean_code,
                corrupt_code,
                heads,
                materialize=materialize_messages,
                gram=gram,
            )
            chunk_budget = torch.zeros(end - begin)
            chunk_budget.index_add_(0, local_query.cpu(), delta_norm.cpu())
            row_budget[layer_index, begin:end] = chunk_budget
            row_net[layer_index, begin:end] = net_row_message_norm(
                output,
                head_index,
                local_query,
                delta_code,
                end - begin,
                heads,
            ).cpu()
            if signal is FlowSignal.MESSAGE:
                assert selector is not None and content is not None
                selected_score = primary[
                    head_index, local_query, source
                ].float().cpu()
                selected_selector = selector[
                    head_index, local_query, source
                ].float().cpu()
                selected_content = content[
                    head_index, local_query, source
                ].float().cpu()
                for values, destination in (
                    (selected_score, target_rows),
                    (selected_score.clamp_min(0), positive_rows),
                    (selected_score.clamp_max(0), negative_rows),
                    (selected_selector, selector_rows),
                    (selected_content, content_rows),
                ):
                    chunk_sum = torch.zeros(end - begin)
                    chunk_sum.index_add_(0, local_query.cpu(), values)
                    destination[layer_index, begin:end] = chunk_sum

            columns["layer"].append(
                torch.full_like(source, layer_index, dtype=torch.int16).cpu()
            )
            columns["head"].append(head_index.to(torch.int16).cpu())
            columns["source"].append(source.to(torch.int32).cpu())
            columns["target"].append(target.to(torch.int32).cpu())
            columns["source_unit"].append(
                units.index_select(0, source).to(torch.int32).cpu()
            )
            columns["attention_clean"].append(clean_weight.float().cpu())
            columns["attention_corrupt"].append(corrupt_weight.float().cpu())
            columns["score"].append(
                primary[head_index, local_query, source].float().cpu()
            )
            if clean_target is None:
                nan = torch.full_like(clean_weight.float(), float("nan")).cpu()
                columns["clean_target_score"].append(nan)
                columns["corrupt_target_score"].append(nan.clone())
                columns["selector_score"].append(nan)
                columns["content_score"].append(nan.clone())
            else:
                columns["clean_target_score"].append(
                    clean_target[head_index, local_query, source].float().cpu()
                )
                columns["corrupt_target_score"].append(
                    corrupt_target[head_index, local_query, source].float().cpu()
                )
                columns["selector_score"].append(
                    selector[head_index, local_query, source].cpu()
                )
                columns["content_score"].append(
                    content[head_index, local_query, source].cpu()
                )
            columns["clean_message_norm"].append(clean_norm.cpu())
            columns["corrupt_message_norm"].append(corrupt_norm.cpu())
            columns["delta_message_norm"].append(delta_norm.cpu())
            # Keep intervention payloads in float32.  Quantising here would make
            # the delete-and-restore positive control test storage error rather
            # than the intervention operator itself.
            columns["clean_code"].append(clean_code.float().cpu())
            columns["corrupt_code"].append(corrupt_code.float().cpu())
            for name, vector in (
                ("clean_message_vector", clean_vector),
                ("corrupt_message_vector", corrupt_vector),
                ("delta_message_vector", delta_vector),
            ):
                columns[name].append(vector.float().cpu())
        del clean_q, clean_k, clean_v, corrupt_q, corrupt_k, corrupt_v, output, gram

    def concatenate(name: str, *, dtype: torch.dtype, width: int | None = None):
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
    hidden = int(model.config.hidden_size) if materialize_messages else 0
    edges = FlowEdges(
        concatenate("layer", dtype=torch.int16),
        concatenate("head", dtype=torch.int16),
        concatenate("source", dtype=torch.int32),
        concatenate("target", dtype=torch.int32),
        concatenate("source_unit", dtype=torch.int32),
        concatenate("attention_clean", dtype=torch.float32),
        concatenate("attention_corrupt", dtype=torch.float32),
        concatenate("score", dtype=torch.float32),
        concatenate("clean_target_score", dtype=torch.float32),
        concatenate("corrupt_target_score", dtype=torch.float32),
        concatenate("selector_score", dtype=torch.float32),
        concatenate("content_score", dtype=torch.float32),
        concatenate("clean_message_norm", dtype=torch.float32),
        concatenate("corrupt_message_norm", dtype=torch.float32),
        concatenate("delta_message_norm", dtype=torch.float32),
        concatenate("clean_code", dtype=torch.float32, width=head_dim),
        concatenate("corrupt_code", dtype=torch.float32, width=head_dim),
        concatenate("clean_message_vector", dtype=torch.float32, width=hidden),
        concatenate("corrupt_message_vector", dtype=torch.float32, width=hidden),
        concatenate("delta_message_vector", dtype=torch.float32, width=hidden),
    )
    coherence = torch.where(row_budget > 0, row_net / row_budget, 0)
    aggregation = RowAggregation(
        row_budget,
        row_net,
        coherence,
        target_rows,
        positive_rows,
        negative_rows,
        selector_rows,
        content_rows,
    )
    return edges, row_total, row_retained, aggregation


def capture_paired_flow(
    model,
    world: PairedWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    *,
    carrier_scope: str = "all",
    coverage: float = 0.95,
    gradient_steps: int = 1,
    query_chunk: int = 8,
    materialize_messages: bool = False,
) -> PairedFlow:
    """Capture one target-specific paired graph and its exact stage ledger."""

    signal = FlowSignal(signal)
    pair = world.prefix(target)
    vocabulary = int(model.config.vocab_size)
    all_ids = torch.cat(
        (
            pair.clean_token_ids,
            pair.corrupt_token_ids,
            torch.tensor([target.positive_token_id, target.negative_token_id]),
        )
    )
    if int(all_ids.min()) < 0 or int(all_ids.max()) >= vocabulary:
        raise ValueError("paired-world token ID lies outside the model vocabulary")
    layers = len(model.model.layers)
    checkpoints = range(layers)
    clean = baseline_forward(
        model,
        pair.clean_token_ids,
        pair.response_start,
        checkpoint_layers=checkpoints,
        checkpoint_stages=signal is FlowSignal.MESSAGE,
        attention_query_chunk=query_chunk,
    )
    corrupt = baseline_forward(
        model,
        pair.corrupt_token_ids,
        pair.response_start,
        checkpoint_layers=checkpoints,
        checkpoint_stages=signal is FlowSignal.MESSAGE,
        attention_query_chunk=query_chunk,
    )
    if carrier_scope == "all":
        positions = torch.arange(target.query_position + 1)
    elif carrier_scope == "response":
        positions = torch.arange(pair.response_start - 1, target.query_position + 1)
    else:
        raise ValueError("carrier_scope must be 'response' or 'all'")

    gradients = None
    stages = None
    if signal is FlowSignal.MESSAGE:
        candidate_position = pair.units.positions(pair.candidate_unit_id)
        attribution_position = torch.unique(
            torch.cat((positions, candidate_position)), sorted=True
        )
        gradients = integrated_target_gradients(
            model,
            pair.clean_token_ids,
            pair.corrupt_token_ids,
            target,
            attribution_position,
            steps=gradient_steps,
            query_chunk=query_chunk,
        )
        stages = stage_trace(clean, corrupt, gradients)
    edges, row_total, row_retained, aggregation = capture_edges(
        model,
        clean,
        corrupt,
        pair.units.token_unit_id,
        positions,
        signal,
        gradients,
        coverage=coverage,
        query_chunk=query_chunk,
        materialize_messages=materialize_messages,
    )
    return PairedFlow(
        signal,
        target,
        margin(model, clean, target),
        margin(model, corrupt, target),
        edges,
        positions,
        row_total,
        row_retained,
        aggregation,
        stages,
        clean,
        corrupt,
    )
