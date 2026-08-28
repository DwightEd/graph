"""Extract exact frozen output blocks and attention-code operator geometry."""

from __future__ import annotations

from typing import Any

import torch

from .schema import OperatorBasis


def q_to_kv_mapping(query_heads: int, kv_heads: int) -> torch.Tensor:
    query_heads, kv_heads = int(query_heads), int(kv_heads)
    if query_heads < 1 or kv_heads < 1 or query_heads % kv_heads:
        raise ValueError("query head count must be a positive multiple of KV heads")
    return torch.arange(query_heads, dtype=torch.long) // (query_heads // kv_heads)


def _gram_factor(gram: torch.Tensor) -> torch.Tensor:
    gram = (gram.double() + gram.double().T) * 0.5
    eigenvalue, eigenvector = torch.linalg.eigh(gram)
    tolerance = max(float(eigenvalue.max().item()), 1.0) * 1e-10
    eigenvalue = torch.where(
        eigenvalue >= tolerance,
        eigenvalue.clamp_min(0),
        torch.zeros_like(eigenvalue),
    )
    return (eigenvector * eigenvalue.sqrt().unsqueeze(0)).float()


def _normalized_operator_factor(
    output: torch.Tensor,
    value: torch.Tensor,
    *,
    block_heads: int = 4,
) -> torch.Tensor:
    """Compute an exact normalized Frobenius geometry for ``W_O,h W_V,h``."""

    heads = int(output.shape[0])
    gram = torch.empty((heads, heads), dtype=torch.float32, device=output.device)
    for start in range(0, heads, int(block_heads)):
        stop = min(start + int(block_heads), heads)
        output_cross = torch.einsum("boa,goc->bgac", output[start:stop], output)
        value_cross = torch.einsum("bai,gci->bgac", value[start:stop], value)
        gram[start:stop] = (output_cross * value_cross).sum(dim=(-1, -2)).float()
    gram = (gram + gram.T) * 0.5
    norm = gram.diagonal().clamp_min(0).sqrt()
    denominator = norm[:, None] * norm[None, :]
    normalized = torch.where(
        denominator > 1e-12,
        gram / denominator.clamp_min(1e-12),
        torch.zeros_like(gram),
    )
    normalized = (normalized + normalized.T) * 0.5
    return _gram_factor(normalized).cpu()


def _layers(model: Any) -> list[Any]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("model must expose a Llama-like .model.layers backbone")
    return list(layers)


def extract_operator_basis(
    model: Any,
    *,
    checkpoint: str,
    compute_device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    gram_block_heads: int = 4,
) -> OperatorBasis:
    """Extract exact output blocks and normalized operator-code factors.

    No learned adapter or surrogate operator is introduced.  The value-path
    geometry is derived directly from the frozen checkpoint's ``v_proj`` and
    ``o_proj`` matrices, including grouped-query head sharing.
    """

    config = model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    if heads * head_dim != hidden:
        raise ValueError("hidden size is incompatible with query heads")
    mapping = q_to_kv_mapping(heads, kv_heads)

    output_layers = []
    output_biases = []
    has_bias = []
    normalized_factors = []
    for index, layer in enumerate(_layers(model)):
        attention = getattr(layer, "self_attn", None)
        if attention is None or not hasattr(attention, "o_proj") or not hasattr(
            attention, "v_proj"
        ):
            raise TypeError(f"layer {index} lacks Llama-like v_proj/o_proj")
        output_weight = attention.o_proj.weight.detach().to(
            device=compute_device, dtype=compute_dtype
        )
        value_weight = attention.v_proj.weight.detach().to(
            device=compute_device, dtype=compute_dtype
        )
        if output_weight.shape != (hidden, heads * head_dim):
            raise ValueError(f"layer {index} o_proj has unexpected geometry")
        if value_weight.shape != (kv_heads * head_dim, hidden):
            raise ValueError(f"layer {index} v_proj has unexpected geometry")
        if not torch.isfinite(output_weight).all() or not torch.isfinite(
            value_weight
        ).all():
            raise ValueError(f"layer {index} value/output projection is non-finite")
        output = output_weight.reshape(hidden, heads, head_dim).permute(1, 0, 2)
        value_unique = value_weight.reshape(kv_heads, head_dim, hidden)
        value = value_unique[mapping.to(value_unique.device)]
        # Cache one exact CPU layer in the checkpoint dtype.  This avoids a
        # repeated GPU-to-CPU transfer for every sample while also avoiding a
        # monolithic stacked allocation.
        output_view = (
            attention.o_proj.weight.detach()
            .reshape(hidden, heads, head_dim)
            .permute(1, 0, 2)
            .contiguous()
            .cpu()
        )
        output_layers.append(output_view)
        normalized_factors.append(
            _normalized_operator_factor(
                output,
                value,
                block_heads=gram_block_heads,
            )
        )
        bias = attention.o_proj.bias
        if bias is None:
            output_biases.append(torch.zeros(hidden, dtype=output_view.dtype))
            has_bias.append(False)
        else:
            output_biases.append(bias.detach().cpu())
            has_bias.append(True)

    return OperatorBasis(
        checkpoint=str(checkpoint),
        output_factor=tuple(output_layers),
        output_bias=tuple(output_biases),
        has_output_bias=torch.tensor(has_bias, dtype=torch.bool),
        normalized_operator_factor=torch.stack(normalized_factors),
        q_to_kv=mapping,
    ).validate()
