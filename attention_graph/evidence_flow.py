"""One-hop, per-layer-head causal attention evidence fields."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def direct_field_names(num_layers, num_heads):
    """Names for retained prompt and response masses, one value per channel."""
    channels = [
        f"L{layer}:H{head}"
        for layer in range(int(num_layers))
        for head in range(int(num_heads))
    ]
    return tuple(
        [f"prompt_mass:{channel}" for channel in channels]
        + [f"response_mass:{channel}" for channel in channels]
    )


def propagation_field_names(num_layers, num_heads):
    """Names for prompt and response residual flow, one value per channel."""
    channels = [
        f"L{layer}:H{head}"
        for layer in range(int(num_layers))
        for head in range(int(num_heads))
    ]
    return tuple(
        [f"prompt_flow:{channel}" for channel in channels]
        + [f"response_flow:{channel}" for channel in channels]
    )


def _stable_seed(seed, sample_id):
    digest = hashlib.blake2b(str(sample_id).encode("utf-8"), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2 ** 63 - 1)


def _route_tensor(route, name, *, device, dtype):
    value = route[name]
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _lag_bin_rewired_sources(
    source, target_relative, *, prompt_count, seed, sample_id, channel=None,
):
    """Choose another causal source in each edge's original log2 lag bin.

    The pseudo-random draw is a stateless hash of the edge, rather than a
    generator draw.  Consequently a CSR block boundary cannot change the
    rewired graph.
    """
    if not len(source):
        return source + int(prompt_count)
    lag = target_relative - source

    log_lag = torch.floor(torch.log2(lag.float())).long()
    lower = torch.ones_like(lag) << log_lag
    upper = torch.minimum((lower << 1) - 1, target_relative)
    count = upper - lower + 1
    original_offset = lag - lower
    has_alternative = count > 1
    if channel is None:
        channel = torch.zeros_like(source)
    # All arithmetic remains below 2**63 for the stored graph geometry.
    modulus = 2_147_483_647
    edge_key = (
        (channel.long() * 1_000_003)
        + (target_relative.long() * 97_409)
        + (source.long() * 8_191)
        + (_stable_seed(seed, sample_id) % modulus)
    )
    hashed = torch.remainder(edge_key * 48_271, modulus)
    candidate = torch.remainder(hashed, (count - 1).clamp_min(1))
    candidate += has_alternative & (candidate >= original_offset)
    chosen_lag = lower + candidate
    return (int(prompt_count) + target_relative - chosen_lag).to(dtype=source.dtype)


def _by_token(values, layers, heads, response_count):
    return values.reshape(layers, heads, response_count).permute(2, 0, 1).reshape(
        response_count, layers * heads
    )


def csr_entries(attention, row_start, row_end):
    """Return global CSR row IDs, columns and values for one row block."""
    row_ptr = attention.response_row_ptr.long()
    starts = row_ptr[row_start:row_end]
    lengths = row_ptr[row_start + 1:row_end + 1] - starts
    positions = torch.arange(
        row_ptr[row_start], row_ptr[row_end], device=row_ptr.device
    )
    rows = torch.repeat_interleave(
        torch.arange(row_start, row_end, device=row_ptr.device), lengths
    )
    return (
        rows,
        attention.response_column_indices[positions].long(),
        attention.response_values[positions].float(),
    )


def _empty_flow_state(lookback, *, prompt_count, floor):
    response_count, layers, heads = map(int, lookback.shape)
    channels = layers * heads
    rows = channels * response_count
    return {
        "values": lookback.float().permute(1, 2, 0).reshape(-1),
        "prompt_mass": torch.zeros(rows, dtype=torch.float32, device=lookback.device),
        "response_mass": torch.zeros(rows, dtype=torch.float32, device=lookback.device),
        "prompt_flow": torch.zeros(rows, dtype=torch.float32, device=lookback.device),
        "response_flow": torch.zeros(rows, dtype=torch.float32, device=lookback.device),
        "rewired_response_flow": torch.zeros(rows, dtype=torch.float32, device=lookback.device),
        "prompt_count": int(prompt_count),
        "response_count": response_count,
        "layers": layers,
        "heads": heads,
        "floor": float(floor),
        "rr_edges": torch.zeros((), dtype=torch.int64, device=lookback.device),
        "changed_edges": torch.zeros((), dtype=torch.int64, device=lookback.device),
    }


def _accumulate_block(state, rows, source, weight, *, sample_id, seed):
    """Accumulate direct, exact, and lag-rewired fields from one CSR block."""
    if not len(rows):
        return None
    response_count = state["response_count"]
    prompt_count = state["prompt_count"]
    channel = torch.div(rows, response_count, rounding_mode="floor")
    target_relative = rows.remainder(response_count)
    is_prompt = source < prompt_count
    row = rows[is_prompt]
    value = weight[is_prompt]
    state["prompt_mass"].index_add_(0, row, value)
    state["prompt_flow"].index_add_(
        0, row, value * (state["values"][row] - state["floor"])
    )
    is_rr = ~is_prompt
    row = rows[is_rr]
    value = weight[is_rr]
    source_relative = source[is_rr] - prompt_count
    target = target_relative[is_rr]
    source_row = channel[is_rr] * response_count + source_relative
    state["response_mass"].index_add_(0, row, value)
    state["response_flow"].index_add_(
        0, row, value * (state["values"][row] - state["values"][source_row])
    )
    rewired_source = _lag_bin_rewired_sources(
        source_relative, target, prompt_count=prompt_count, seed=seed,
        sample_id=sample_id, channel=channel[is_rr],
    )
    rewired_row = channel[is_rr] * response_count + (rewired_source - prompt_count)
    state["rewired_response_flow"].index_add_(
        0, row, value * (state["values"][row] - state["values"][rewired_row])
    )
    state["rr_edges"] += len(row)
    state["changed_edges"] += (rewired_source != source[is_rr]).sum()
    return rewired_source, is_rr


def _finish_flow_state(state):
    layers, heads, response_count = (
        state["layers"], state["heads"], state["response_count"]
    )
    direct = torch.cat((
        _by_token(state["prompt_mass"], layers, heads, response_count),
        _by_token(state["response_mass"], layers, heads, response_count),
    ), dim=1)
    true_propagation = torch.cat((
        _by_token(state["prompt_flow"], layers, heads, response_count),
        _by_token(state["response_flow"], layers, heads, response_count),
    ), dim=1)
    rewired_propagation = torch.cat((
        _by_token(state["prompt_flow"], layers, heads, response_count),
        _by_token(state["rewired_response_flow"], layers, heads, response_count),
    ), dim=1)
    return direct, true_propagation, rewired_propagation


def evidence_flow_from_attention(
    lookback, attention, *, csr_row_block=4096, sample_id="", seed=0,
):
    """Stream exact CSR edges into direct and one-hop graph node fields.

    This is the production path.  It never materializes a whole-sample COO
    route: each CSR block is consumed immediately on the attention device.
    """
    if not isinstance(lookback, torch.Tensor) or lookback.ndim != 3:
        raise ValueError("lookback must be a [response,layer,head] tensor")
    if int(lookback.shape[0]) != int(attention.num_response_tokens):
        raise ValueError("lookback and attention response lengths differ")
    state = _empty_flow_state(
        lookback, prompt_count=attention.response_idx,
        floor=attention.attention_floor,
    )
    rows_count = int(attention.num_channels) * int(attention.num_response_tokens)
    for row_start in range(0, rows_count, int(csr_row_block)):
        rows, source, weight = csr_entries(
            attention, row_start, min(row_start + int(csr_row_block), rows_count)
        )
        _accumulate_block(state, rows, source, weight, sample_id=sample_id, seed=seed)
    direct, true_propagation, rewired_propagation = _finish_flow_state(state)
    rr_edges = int(state["rr_edges"].item())
    changed_edges = int(state["changed_edges"].item())
    audit = {
        "rr_edges": rr_edges,
        "rewired_changed_edges": changed_edges,
        "rewired_changed_fraction": (
            changed_edges / rr_edges if rr_edges else 0.0
        ),
    }
    return direct, true_propagation, rewired_propagation, audit


def evidence_flow_fields(
    lookback,
    route,
    *,
    prompt_count,
    sample_id="",
    seed=0,
    randomize_rr=False,
):
    """Return direct masses and one-hop residual flows over exact CSR channels.

    Each stored route edge acts only on its own ``channel = layer * H + head``.
    The direct controls are prompt and response retained masses.  The two
    propagation blocks are ``Fp = sum_prompt w * (X_target - floor)`` and
    ``Fr = sum_RR w * (X_target - X_source)``.  Missing CSR edges contribute
    nothing; no dense topology is inferred.
    """
    if not isinstance(lookback, torch.Tensor) or lookback.ndim != 3:
        raise ValueError("lookback must be a [response,layer,head] tensor")
    response_count, layers, heads = map(int, lookback.shape)
    prompt_count = int(prompt_count)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("evidence-flow graph requires prompt and response nodes")
    required = (
        "channel", "layer", "head", "source", "target", "weight",
        "attention_floor",
    )
    if any(name not in route for name in required):
        raise ValueError("exact channel route is missing required COO fields")

    device = lookback.device
    channel = _route_tensor(route, "channel", device=device, dtype=torch.long)
    layer = _route_tensor(route, "layer", device=device, dtype=torch.long)
    head = _route_tensor(route, "head", device=device, dtype=torch.long)
    source = _route_tensor(route, "source", device=device, dtype=torch.long)
    target = _route_tensor(route, "target", device=device, dtype=torch.long)
    weight = _route_tensor(route, "weight", device=device, dtype=torch.float32)
    if not all(len(value) == len(channel) for value in (layer, head, source, target, weight)):
        raise ValueError("route COO arrays have inconsistent lengths")
    channels = layers * heads
    if len(channel) and (
        int(channel.min()) < 0 or int(channel.max()) >= channels
        or int(layer.min()) < 0 or int(layer.max()) >= layers
        or int(head.min()) < 0 or int(head.max()) >= heads
        or not torch.equal(channel, layer * heads + head)
        or int(target.min()) < prompt_count
        or int(target.max()) >= prompt_count + response_count
        or int(source.min()) < 0
        or bool((source >= target).any())
        or not bool(torch.isfinite(weight).all())
        or bool((weight < 0).any())
    ):
        raise ValueError("route COO violates exact causal channel geometry")
    floor = float(route["attention_floor"])
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("route attention_floor must be finite and nonnegative")

    row = channel * response_count + (target - prompt_count)
    state = _empty_flow_state(lookback, prompt_count=prompt_count, floor=floor)
    result = _accumulate_block(
        state, row, source, weight, sample_id=sample_id, seed=seed
    )
    direct, true_propagation, rewired_propagation = _finish_flow_state(state)
    propagation = rewired_propagation if randomize_rr else true_propagation
    if direct.shape != (response_count, len(direct_field_names(layers, heads))):
        raise RuntimeError("direct evidence-flow field differs from its schema")
    if propagation.shape != (response_count, len(propagation_field_names(layers, heads))):
        raise RuntimeError("propagation evidence-flow field differs from its schema")
    diagnostics = {
        "rewire_audit": {
            "rr_edges": int(state["rr_edges"].item()),
            "rewired_changed_edges": int(state["changed_edges"].item()),
            "rewired_changed_fraction": (
                int(state["changed_edges"].item()) / int(state["rr_edges"].item())
                if int(state["rr_edges"].item()) else 0.0
            ),
        }
    }
    if randomize_rr:
        rewired_source = source.clone()
        if result is not None:
            source_rewired, is_rr = result
            rewired_source[is_rr] = source_rewired
        diagnostics["rewired_route"] = {
            "channel": channel,
            "layer": layer,
            "head": head,
            "source": rewired_source,
            "target": target,
            "weight": weight,
        }
    return direct, propagation, diagnostics


def anomaly_components(route, *, prompt_count, scores, threshold):
    """Connected components induced by high-score response nodes and RR routes."""
    scores = np.asarray(scores, dtype=np.float32)
    active = scores >= float(threshold)
    parent = np.arange(len(scores), dtype=np.int32)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left, right):
        left, right = find(int(left)), find(int(right))
        if left != right:
            parent[right] = left

    source = np.asarray(route["source"], dtype=np.int64) - int(prompt_count)
    target = np.asarray(route["target"], dtype=np.int64) - int(prompt_count)
    valid = (
        (source >= 0) & (source < len(scores))
        & (target >= 0) & (target < len(scores))
    )
    for left, right in zip(source[valid], target[valid]):
        if active[left] and active[right]:
            union(left, right)
    component = np.full(len(scores), -1, dtype=np.int32)
    roots = {}
    for node in np.flatnonzero(active):
        root = find(int(node))
        if root not in roots:
            roots[root] = len(roots)
        component[node] = roots[root]
    return active, component


def anomaly_components_from_attention(
    attention, *, scores, threshold, csr_row_block=4096,
):
    """Find active-node RR components by streaming the canonical CSR graph."""
    scores = np.asarray(scores, dtype=np.float32)
    active = scores >= float(threshold)
    parent = np.arange(len(scores), dtype=np.int32)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left, right):
        left, right = find(int(left)), find(int(right))
        if left != right:
            parent[right] = left

    device = attention.response_values.device
    active_tensor = torch.as_tensor(active, dtype=torch.bool, device=device)
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    rows_count = int(attention.num_channels) * response_count
    for row_start in range(0, rows_count, int(csr_row_block)):
        rows, source, _ = csr_entries(
            attention, row_start, min(row_start + int(csr_row_block), rows_count)
        )
        source = source - prompt_count
        target = rows.remainder(response_count)
        history = source >= 0
        source, target = source[history], target[history]
        selected = active_tensor[source] & active_tensor[target]
        pairs = torch.stack((source[selected], target[selected]), dim=1).cpu().numpy()
        for left, right in pairs:
            union(left, right)

    component = np.full(len(scores), -1, dtype=np.int32)
    roots = {}
    for node in np.flatnonzero(active):
        root = find(int(node))
        if root not in roots:
            roots[root] = len(roots)
        component[node] = roots[root]
    return active, component
