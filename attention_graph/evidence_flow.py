"""Fixed, label-free graph filter bank for causal attention evidence flow.

This module deliberately contains no trainable message MLP.  It retains a
high-dimensional node signal, sends it over the actual layer-wise attention
topology, and exposes diffusion wavelets (local and scale innovations) to a
train-only one-class reference model.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def fixed_head_projection(num_heads, components, seed, *, device="cpu"):
    """Return a deterministic orthonormal head projection without fitting data."""
    heads = int(num_heads)
    width = min(int(components), heads)
    if heads < 1 or width < 1:
        raise ValueError("head projection dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((heads, width))
    orthogonal, _ = np.linalg.qr(matrix, mode="reduced")
    # Resolve QR's arbitrary column signs for byte-for-byte reproducibility.
    pivot = np.argmax(np.abs(orthogonal), axis=0)
    sign = np.sign(orthogonal[pivot, np.arange(width)])
    orthogonal *= np.where(sign == 0, 1.0, sign)
    return torch.as_tensor(
        orthogonal.astype(np.float32), dtype=torch.float32, device=device
    )


def direct_field_names(num_layers, prompt_bins):
    names = []
    for layer in range(int(num_layers)):
        names.extend(
            f"direct_prompt_mass:L{layer}:B{bin_id}"
            for bin_id in range(int(prompt_bins))
        )
    names.extend(f"rr_log_mass_hop1:L{layer}" for layer in range(int(num_layers)))
    return tuple(names)


def propagation_field_names(num_layers, head_components, prompt_bins):
    names = []
    for block in ("node_local_innovation", "node_scale_innovation"):
        for layer in range(int(num_layers)):
            names.extend(
                f"{block}:L{layer}:C{component}"
                for component in range(int(head_components))
            )
    for block in ("prompt_local_innovation", "prompt_scale_innovation"):
        for layer in range(int(num_layers)):
            names.extend(
                f"{block}:L{layer}:B{bin_id}"
                for bin_id in range(int(prompt_bins))
            )
    names.extend(f"rr_log_mass_hop2:L{layer}" for layer in range(int(num_layers)))
    return tuple(names)


def propagation_block_slices(num_layers, head_components, prompt_bins):
    """Named contiguous blocks in ``propagation_field_names`` order."""
    layer_head = int(num_layers) * int(head_components)
    layer_prompt = int(num_layers) * int(prompt_bins)
    layer_mass = int(num_layers)
    widths = (
        ("node_local_innovation", layer_head),
        ("node_scale_innovation", layer_head),
        ("prompt_local_innovation", layer_prompt),
        ("prompt_scale_innovation", layer_prompt),
        ("two_hop_path_mass", layer_mass),
    )
    output, start = {}, 0
    for name, width in widths:
        output[name] = slice(start, start + width)
        start += width
    return output


def _stable_seed(seed, sample_id):
    digest = hashlib.blake2b(str(sample_id).encode("utf-8"), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2 ** 63 - 1)


def _route_tensor(route, name, *, device, dtype):
    value = route[name]
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _random_causal_sources(source, target, *, seed, sample_id):
    """Randomize RR endpoints while preserving every target, layer and weight."""
    target_cpu = target.detach().cpu().numpy().astype(np.int64, copy=False)
    rng = np.random.default_rng(_stable_seed(seed, sample_id))
    if bool((target_cpu < 1).any()):
        raise ValueError("an RR route into the first response token is not causal")
    randomized = np.floor(rng.random(len(target_cpu)) * target_cpu).astype(np.int64)
    return torch.as_tensor(randomized, dtype=source.dtype, device=source.device)


def evidence_flow_fields(
    lookback,
    route,
    *,
    prompt_count,
    prompt_bins,
    head_projection,
    sample_id,
    seed,
    randomize_rr=False,
):
    """Build direct-edge and multi-hop diffusion fields for every response node.

    ``lookback`` has shape ``[response, layer, head]``.  The returned direct
    field has prompt-bin masses and one-hop RR mass.  The propagation field
    retains signed node/prompt diffusion wavelets plus two-hop path mass.
    """
    if not isinstance(lookback, torch.Tensor) or lookback.ndim != 3:
        raise ValueError("lookback must be a [response,layer,head] tensor")
    response_count, layers, heads = map(int, lookback.shape)
    prompt_count = int(prompt_count)
    prompt_bins = int(prompt_bins)
    if response_count < 1 or prompt_count < 1 or prompt_bins < 1:
        raise ValueError("evidence-flow graph requires prompt and response nodes")
    projection = head_projection.to(device=lookback.device, dtype=torch.float32)
    if projection.ndim != 2 or projection.shape[0] != heads:
        raise ValueError("head projection does not match Lookback geometry")
    projected_width = int(projection.shape[1])
    device = lookback.device
    layer = _route_tensor(route, "layer", device=device, dtype=torch.long)
    source = _route_tensor(route, "source", device=device, dtype=torch.long)
    target = _route_tensor(route, "target", device=device, dtype=torch.long)
    weight = _route_tensor(route, "weight", device=device, dtype=torch.float32)
    if not (len(layer) == len(source) == len(target) == len(weight)):
        raise ValueError("route COO arrays have inconsistent lengths")
    if len(layer) and (
        int(layer.min()) < 0 or int(layer.max()) >= layers
        or int(target.min()) < prompt_count
        or int(target.max()) >= prompt_count + response_count
        or not bool(torch.isfinite(weight).all())
        or bool((weight < 0).any())
    ):
        raise ValueError("route COO violates graph geometry")

    target_relative = target - prompt_count
    row = layer * response_count + target_relative
    rows = layers * response_count
    prompt_profile = torch.zeros(
        (rows, prompt_bins), dtype=torch.float32, device=device
    )
    is_prompt = source < prompt_count
    if bool(is_prompt.any()):
        prompt_bin = torch.div(
            source[is_prompt] * prompt_bins, prompt_count, rounding_mode="floor"
        ).clamp_max(prompt_bins - 1)
        flat_index = row[is_prompt] * prompt_bins + prompt_bin
        prompt_profile.view(-1).index_add_(0, flat_index, weight[is_prompt])

    is_rr = ~is_prompt
    rr_row = row[is_rr]
    rr_source = source[is_rr] - prompt_count
    rr_target = target_relative[is_rr]
    rr_weight = weight[is_rr]
    if len(rr_source) and bool((rr_source >= rr_target).any()):
        raise ValueError("response routes must point strictly into causal history")
    if randomize_rr and len(rr_source):
        rr_source = _random_causal_sources(
            rr_source, rr_target, seed=seed, sample_id=sample_id
        )
    rr_layer = layer[is_rr]
    rr_source_row = rr_layer * response_count + rr_source

    node = torch.einsum(
        "tlh,hc->tlc", lookback.float(), projection
    ).permute(1, 0, 2).reshape(rows, projected_width)
    mass1 = torch.zeros(rows, dtype=torch.float32, device=device)
    node_numerator1 = torch.zeros_like(node)
    prompt_numerator1 = torch.zeros_like(prompt_profile)
    if len(rr_row):
        mass1.index_add_(0, rr_row, rr_weight)
        node_numerator1.index_add_(
            0, rr_row, rr_weight[:, None] * node[rr_source_row]
        )
        prompt_numerator1.index_add_(
            0, rr_row, rr_weight[:, None] * prompt_profile[rr_source_row]
        )
    reachable1 = mass1 > 0
    message1_node = torch.where(
        reachable1[:, None], node_numerator1 / mass1[:, None].clamp_min(1e-12), 0.0
    )
    message1_prompt = torch.where(
        reachable1[:, None],
        prompt_numerator1 / mass1[:, None].clamp_min(1e-12),
        0.0,
    )

    mass2 = torch.zeros_like(mass1)
    node_numerator2 = torch.zeros_like(node)
    prompt_numerator2 = torch.zeros_like(prompt_profile)
    if len(rr_row):
        mass2.index_add_(0, rr_row, rr_weight * mass1[rr_source_row])
        node_numerator2.index_add_(
            0, rr_row, rr_weight[:, None] * node_numerator1[rr_source_row]
        )
        prompt_numerator2.index_add_(
            0, rr_row, rr_weight[:, None] * prompt_numerator1[rr_source_row]
        )
    reachable2 = mass2 > 0
    message2_node = torch.where(
        reachable2[:, None], node_numerator2 / mass2[:, None].clamp_min(1e-12), 0.0
    )
    message2_prompt = torch.where(
        reachable2[:, None],
        prompt_numerator2 / mass2[:, None].clamp_min(1e-12),
        0.0,
    )

    # Multiplying conditional innovations by raw path mass makes propagation
    # continuous at zero: an epsilon-weight route cannot transmit a full-size
    # neighbour residual merely because it is the only retained route.
    node_local = torch.where(
        reachable1[:, None], mass1[:, None] * (node - message1_node), 0.0
    )
    node_scale = torch.where(
        reachable2[:, None], mass2[:, None] * (message1_node - message2_node), 0.0
    )
    prompt_local = torch.where(
        reachable1[:, None],
        mass1[:, None] * (prompt_profile - message1_prompt), 0.0
    )
    prompt_scale = torch.where(
        reachable2[:, None],
        mass2[:, None] * (message1_prompt - message2_prompt), 0.0
    )

    def by_token(values):
        return values.reshape(layers, response_count, -1).permute(1, 0, 2).reshape(
            response_count, -1
        )

    direct = torch.cat((
        by_token(prompt_profile),
        mass1.reshape(layers, response_count).T.log1p(),
    ), dim=1)
    propagation = torch.cat((
        by_token(node_local), by_token(node_scale),
        by_token(prompt_local), by_token(prompt_scale),
        mass2.reshape(layers, response_count).T.log1p(),
    ), dim=1)
    expected_direct = len(direct_field_names(layers, prompt_bins))
    expected_propagation = len(
        propagation_field_names(layers, projected_width, prompt_bins)
    )
    if direct.shape != (response_count, expected_direct):
        raise RuntimeError("direct evidence-flow field differs from its schema")
    if propagation.shape != (response_count, expected_propagation):
        raise RuntimeError("propagation evidence-flow field differs from its schema")
    diagnostics = {
        "rr_mass_hop1": mass1.reshape(layers, response_count).T,
        "rr_mass_hop2": mass2.reshape(layers, response_count).T,
        "reachable_hop1": reachable1.reshape(layers, response_count).T,
        "reachable_hop2": reachable2.reshape(layers, response_count).T,
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
