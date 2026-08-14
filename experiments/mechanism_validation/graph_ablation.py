"""Sparse-trace graph ablations and channel-preserving descriptors.

The functions here deliberately consume ``AttentionSample`` CSR traces directly.
They never materialise the layer/head-averaged graph edge score.
"""

from dataclasses import dataclass

import torch

RP = 0
RR = 1


@dataclass
class TraceBatch:
    """One sparse attention trace per retained CSR entry."""

    target: torch.Tensor  # response-relative
    source: torch.Tensor  # absolute token index
    channel: torch.Tensor  # layer * num_heads + head
    relation: torch.Tensor  # RP or RR
    value: torch.Tensor

    def __getitem__(self, index: torch.Tensor) -> "TraceBatch":
        return TraceBatch(*(getattr(self, name)[index] for name in self.__dataclass_fields__))

    def clone(self) -> "TraceBatch":
        return TraceBatch(*(getattr(self, name).clone() for name in self.__dataclass_fields__))

    @property
    def device(self) -> torch.device:
        return self.value.device


@dataclass
class DescriptorBatch:
    features: torch.Tensor
    feature_names: list[str]
    source_aware: torch.Tensor
    family_masks: dict[str, torch.Tensor]


def extract_traces(sample) -> TraceBatch:
    """Vectorize canonical/formal response CSR into atomic layer-head traces."""
    row_ptr = sample.response_row_ptr.long()
    lengths = row_ptr[1:] - row_ptr[:-1]
    rows = torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths)
    response_tokens = sample.num_response_tokens
    source = sample.response_column_indices.long()
    target = rows.remainder(response_tokens)
    return TraceBatch(
        target=target,
        source=source,
        channel=rows.div(response_tokens, rounding_mode="floor"),
        relation=(source >= sample.response_idx).long(),
        value=sample.response_values.float(),
    )


def _row_ids(trace: TraceBatch, response_tokens: int) -> tuple[torch.Tensor, int]:
    count = int(trace.channel.max()) + 1 if trace.channel.numel() else 0
    ids = ((trace.channel * response_tokens + trace.target) * 2 + trace.relation).long()
    return ids, 2 * count * response_tokens


def _group_sum(ids: torch.Tensor, values: torch.Tensor, size: int) -> torch.Tensor:
    result = values.new_zeros(size)
    if ids.numel():
        result.index_add_(0, ids, values)
    return result


def _changed_fraction(before: TraceBatch, after: TraceBatch) -> float:
    if before.value.numel() == 0:
        return 0.0
    if before.value.numel() != after.value.numel():
        return 1.0
    same = (
        (before.target == after.target)
        & (before.source == after.source)
        & (before.channel == after.channel)
        & (before.relation == after.relation)
        & (before.value == after.value)
    )
    return float((~same).float().mean())


def apply_trace_variant(trace: TraceBatch, variant: str, *, response_idx: int, seed: int = 0) -> tuple[TraceBatch, dict]:
    """Apply one deterministic, sparse trace intervention and return its audit."""
    allowed = {"exact", "no_edges", "unit_mass", "uniform_on_support", "weight_shuffle", "source_rewire", "rp_only", "rr_only"}
    if variant not in allowed:
        raise ValueError(f"unknown graph ablation variant: {variant}")
    if variant == "exact":
        changed = trace
    elif variant == "no_edges":
        empty_long = trace.source.new_empty(0)
        changed = TraceBatch(empty_long, empty_long, empty_long, empty_long, trace.value.new_empty(0))
    elif variant in {"rp_only", "rr_only"}:
        changed = trace[trace.relation == (RP if variant == "rp_only" else RR)].clone()
    else:
        changed = trace.clone()
        response_tokens = int(trace.target.max()) + 1 if trace.target.numel() else 0
        ids, size = _row_ids(trace, response_tokens)
        mass = _group_sum(ids, trace.value, size)
        degree = _group_sum(ids, torch.ones_like(trace.value), size)
        if variant == "unit_mass":
            changed.value = trace.value / mass[ids].clamp_min(torch.finfo(trace.value.dtype).eps)
        elif variant == "uniform_on_support":
            changed.value = mass[ids] / degree[ids].clamp_min(1)
        elif variant == "weight_shuffle":
            generator = torch.Generator(device=trace.device)
            generator.manual_seed(int(seed))
            random_order = torch.argsort(torch.rand(trace.value.numel(), generator=generator, device=trace.device))
            shuffled = random_order[torch.argsort(ids[random_order], stable=True)]
            destination = torch.argsort(ids, stable=True)
            changed.value[destination] = trace.value[shuffled]
        else:  # source_rewire: the mapping is a rotation shared by all channels.
            target_absolute = trace.target + int(response_idx)
            domain = torch.where(trace.relation == RP, torch.full_like(target_absolute, response_idx), target_absolute - response_idx)
            nontrivial = domain > 1
            shift = torch.remainder(
                target_absolute * 1103515245 + trace.relation * 12345 + int(seed),
                (domain - 1).clamp_min(1),
            ) + 1
            shifted_rp = torch.remainder(trace.source + shift, response_idx)
            shifted_rr = response_idx + torch.remainder(trace.source - response_idx + shift, domain.clamp_min(1))
            changed.source = torch.where(nontrivial, torch.where(trace.relation == RP, shifted_rp, shifted_rr), trace.source)
    audit = {
        "variant": variant,
        "edges_before": int(trace.value.numel()),
        "edges_after": int(changed.value.numel()),
        "changed_fraction": _changed_fraction(trace, changed),
    }
    return changed, audit


def _moments(row: torch.Tensor, values: torch.Tensor, weights: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = _group_sum(row, torch.ones_like(weights), size)
    weight_sum = _group_sum(row, weights, size)
    weighted_mean = _group_sum(row, weights * values, size) / weight_sum.clamp_min(torch.finfo(weights.dtype).eps)
    weighted_second = _group_sum(row, weights * values.square(), size) / weight_sum.clamp_min(torch.finfo(weights.dtype).eps)
    mean = _group_sum(row, values, size) / count.clamp_min(1)
    second = _group_sum(row, values.square(), size) / count.clamp_min(1)
    return weighted_mean, (weighted_second - weighted_mean.square()).clamp_min(0).sqrt(), mean, (second - mean.square()).clamp_min(0).sqrt()


def _channel_block(trace: TraceBatch, node_features: torch.Tensor, response_idx: int, start: int, stop: int) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Compute [target, channel, primitive] descriptors for one channel block."""
    tokens = node_features.shape[0]
    targets = tokens - response_idx
    channels = stop - start
    mask = (trace.channel >= start) & (trace.channel < stop)
    part = trace[mask]
    local = part.channel - start
    rows = ((local * targets + part.target) * 2 + part.relation).long()
    size = channels * targets * 2
    values = part.value.float()
    mass = _group_sum(rows, values, size)
    degree = _group_sum(rows, torch.ones_like(values), size)
    square = _group_sum(rows, values.square(), size)
    top = values.new_zeros(size)
    if rows.numel():
        top.scatter_reduce_(0, rows, values, reduce="amax", include_self=True)
    hhi = square / mass.square().clamp_min(torch.finfo(values.dtype).eps)
    top_share = top / mass.clamp_min(torch.finfo(values.dtype).eps)
    shape = (channels, targets, 2)
    row_view = lambda tensor: tensor.view(shape).permute(1, 0, 2)
    rp_marginal = torch.stack([row_view(mass)[..., RP], row_view(degree)[..., RP], row_view(hhi)[..., RP], row_view(top_share)[..., RP]], dim=-1)
    rr_marginal = torch.stack([row_view(mass)[..., RR], row_view(degree)[..., RR], row_view(hhi)[..., RR], row_view(top_share)[..., RR]], dim=-1)
    marginal_names = ["mass", "degree", "hhi", "top1_share"]
    marginal_aware = torch.zeros(4, dtype=torch.bool, device=node_features.device)

    rp_mask = part.relation == RP
    rr_mask = part.relation == RR
    rp_rows = rows[rp_mask]
    rr_rows = rows[rr_mask]
    rp_values = values[rp_mask]
    rr_values = values[rr_mask]
    rp_position = part.source[rp_mask].float() / max(response_idx - 1, 1)
    lag = (response_idx + part.target[rr_mask] - part.source[rr_mask]).float()
    rp_stats = _moments(rp_rows, rp_position, rp_values, size)
    rr_stats = _moments(rr_rows, lag, rr_values, size)
    rp_source = torch.stack([row_view(stat)[..., RP] for stat in rp_stats], dim=-1)
    rr_source = torch.stack([row_view(stat)[..., RR] for stat in rr_stats], dim=-1)
    source_names = ["weighted_mean", "weighted_std", "mean", "std"]
    source_aware = torch.ones(4, dtype=torch.bool, device=node_features.device)

    bins = 8
    rp_bin = (part.source[rp_mask] * bins // response_idx).clamp(max=bins - 1)
    bin_rows = rp_rows * bins + rp_bin
    bin_mass = _group_sum(bin_rows, rp_values, size * bins).view(size, bins)
    prompt_bins = bin_mass / mass[:, None].clamp_min(torch.finfo(values.dtype).eps)
    prompt_bins = prompt_bins.view(channels, targets, 2, bins).permute(1, 0, 2, 3)[..., RP, :]
    previous = torch.cat([torch.zeros_like(prompt_bins[:1]), prompt_bins[:-1]], dim=0)
    cosine = (prompt_bins * previous).sum(-1) / (prompt_bins.norm(dim=-1) * previous.norm(dim=-1)).clamp_min(torch.finfo(values.dtype).eps)
    turnover = .5 * (prompt_bins - previous).abs().sum(-1)
    cosine[0] = 0
    turnover[0] = 0
    prompt_source = torch.cat([prompt_bins, cosine[..., None], turnover[..., None]], dim=-1)
    prompt_names = [f"position_bin_{index}" for index in range(bins)] + ["adjacent_cosine", "adjacent_turnover"]
    prompt_aware = torch.ones(len(prompt_names), dtype=torch.bool, device=node_features.device)

    feature_dim = node_features.shape[1]
    weighted_aggregate = node_features.new_zeros((size, feature_dim))
    unweighted_aggregate = node_features.new_zeros((size, feature_dim))
    if rr_rows.numel():
        predecessor = node_features[part.source[rr_mask].long()].float()
        weighted_aggregate.index_add_(0, rr_rows, predecessor * rr_values[:, None])
        unweighted_aggregate.index_add_(0, rr_rows, predecessor)
    weighted_aggregate = weighted_aggregate / mass[:, None].clamp_min(torch.finfo(values.dtype).eps)
    unweighted_aggregate = unweighted_aggregate / degree[:, None].clamp_min(1)
    target_index = (
        torch.arange(targets, device=node_features.device)
        .view(1, targets, 1)
        .expand(channels, targets, 2)
        .reshape(-1)
        + response_idx
    )
    target_feature = node_features[target_index].float()
    rr_active = ((torch.arange(size, device=mass.device).remainder(2) == RR) & (mass > 0))[:, None]
    weighted_aggregate = weighted_aggregate * rr_active
    unweighted_aggregate = unweighted_aggregate * rr_active
    weighted_residual = (target_feature - weighted_aggregate) * rr_active
    unweighted_residual = (target_feature - unweighted_aggregate) * rr_active
    weighted_cosine = (target_feature * weighted_aggregate).sum(-1) / (target_feature.norm(dim=-1) * weighted_aggregate.norm(dim=-1)).clamp_min(torch.finfo(values.dtype).eps)
    unweighted_cosine = (target_feature * unweighted_aggregate).sum(-1) / (target_feature.norm(dim=-1) * unweighted_aggregate.norm(dim=-1)).clamp_min(torch.finfo(values.dtype).eps)
    rr_node = torch.cat([weighted_aggregate, unweighted_aggregate, weighted_residual, unweighted_residual, weighted_cosine[:, None], unweighted_cosine[:, None]], dim=-1)
    rr_node = rr_node.view(channels, targets, 2, -1).permute(1, 0, 2, 3)[..., RR, :]
    node_names = (
        [f"predecessor_weighted_{index}" for index in range(feature_dim)]
        + [f"predecessor_mean_{index}" for index in range(feature_dim)]
        + [f"target_minus_weighted_{index}" for index in range(feature_dim)]
        + [f"target_minus_mean_{index}" for index in range(feature_dim)]
        + ["target_weighted_cosine", "target_mean_cosine"]
    )
    node_aware = torch.ones(len(node_names), dtype=torch.bool, device=node_features.device)

    values_out = torch.cat([rp_marginal, rp_source, prompt_source, rr_marginal, rr_source, rr_node], dim=-1)
    names = (
        [f"rp:{name}" for name in marginal_names]
        + [f"rp:source_position_{name}" for name in source_names]
        + [f"rp:prompt_{name}" for name in prompt_names]
        + [f"rr:{name}" for name in marginal_names]
        + [f"rr:lag_{name}" for name in source_names]
        + [f"rr:{name}" for name in node_names]
    )
    aware = torch.cat([marginal_aware, source_aware, prompt_aware, marginal_aware, source_aware, node_aware])
    return values_out, names, aware


def fixed_graph_descriptors(trace: TraceBatch, node_features: torch.Tensor, response_idx: int, num_layers: int, num_heads: int, *, source_free: bool = False, channel_block: int = 256) -> DescriptorBatch:
    """Compress per-layer/head sparse descriptors without averaging traces first."""
    if node_features.ndim != 2:
        raise ValueError("node_features must be [tokens, features]")
    response_tokens = node_features.shape[0] - response_idx
    channels = num_layers * num_heads
    if response_tokens < 1 or channels < 1:
        raise ValueError("response and channel counts must be positive")
    if trace.channel.numel() and int(trace.channel.max()) >= channels:
        raise ValueError("trace channel exceeds descriptor geometry")
    feature_names = None
    source_aware = None
    total = squares = layer_sum = head_sum = None
    layer_count = torch.zeros(4, dtype=torch.float32, device=node_features.device)
    head_count = torch.zeros(4, dtype=torch.float32, device=node_features.device)
    for start in range(0, channels, channel_block):
        stop = min(start + channel_block, channels)
        block, names, aware = _channel_block(trace, node_features, response_idx, start, stop)
        if feature_names is None:
            feature_names, source_aware = names, aware
            dimensions = block.shape[-1]
            total = block.new_zeros((response_tokens, dimensions))
            squares = block.new_zeros((response_tokens, dimensions))
            layer_sum = block.new_zeros((4, response_tokens, dimensions))
            head_sum = block.new_zeros((4, response_tokens, dimensions))
        total += block.sum(dim=1)
        squares += block.square().sum(dim=1)
        absolute_channels = torch.arange(start, stop, device=node_features.device)
        layer_bin = absolute_channels.div(num_heads, rounding_mode="floor") * 4 // num_layers
        head_bin = absolute_channels.remainder(num_heads) * 4 // num_heads
        for bin_index in range(4):
            layer_selected = layer_bin == bin_index
            head_selected = head_bin == bin_index
            if bool(layer_selected.any()):
                layer_sum[bin_index] += block[:, layer_selected].sum(dim=1)
                layer_count[bin_index] += layer_selected.sum()
            if bool(head_selected.any()):
                head_sum[bin_index] += block[:, head_selected].sum(dim=1)
                head_count[bin_index] += head_selected.sum()
    mean = total / channels
    std = (squares / channels - mean.square()).clamp_min(0).sqrt()
    layer_mean = layer_sum / layer_count[:, None, None].clamp_min(1)
    head_mean = head_sum / head_count[:, None, None].clamp_min(1)
    summary = torch.cat([mean[..., None], std[..., None], layer_mean.permute(1, 2, 0), head_mean.permute(1, 2, 0)], dim=-1)
    suffixes = ["global_mean", "global_std"] + [f"layer_bin_{index}" for index in range(4)] + [f"head_bin_{index}" for index in range(4)]
    names = [f"{name}:{suffix}" for name in feature_names for suffix in suffixes]
    mask = source_aware.repeat_interleave(len(suffixes))
    features = summary.reshape(response_tokens, -1)
    if source_free:
        features = features.clone()
        features[:, mask] = 0
    family_masks = {
        "row_marginal": ~mask,
        "source_aware": mask,
        "all": torch.ones_like(mask),
    }
    return DescriptorBatch(features, names, mask, family_masks)
