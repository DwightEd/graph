"""Actual attention writes and candidate readout; no label or experiment logic."""

import torch
from torch.nn import functional as F


def grouped_values(projected_values, heads, kv_heads):
    batch, length, width = projected_values.shape
    values = projected_values.view(batch, length, kv_heads, width // kv_heads)
    return values.transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)


def source_write(attention, values, output_weight, queries, sources, heads):
    """m_S = W_O concat_h(sum_{j in S} A[h,q,j] V[h,j]); keep native arithmetic."""
    weights = attention[0][:, queries][:, :, sources]
    weighted_values = weights @ values[0][:, sources]
    selected = torch.zeros_like(weighted_values)
    selected[list(heads)] = weighted_values[list(heads)]
    joined = selected.transpose(0, 1).reshape(len(queries), -1)
    return selected, F.linear(joined, output_weight), weights.sum(dim=-1)


def equal_norm_change(write, seed, reference_norm=None):
    generator = torch.Generator(device=write.device).manual_seed(seed)
    random = torch.randn(write.shape, generator=generator, device=write.device, dtype=torch.float32)
    random /= random.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    length = write.float().norm(dim=-1, keepdim=True)
    if reference_norm is not None:
        length = torch.full_like(length, reference_norm)
    return (random * length).to(write.dtype)


def lens_margin(model, states, correct, wrong):
    """Intermediate residual readout. It does not execute the remaining layers."""
    normalized = model.model.norm(states)
    direction = model.lm_head.weight[correct].float() - model.lm_head.weight[wrong].float()
    margin = (normalized.float() * direction).sum(dim=-1)
    if model.lm_head.bias is not None:
        margin += model.lm_head.bias[correct].float() - model.lm_head.bias[wrong].float()
    return margin


def local_readout_direction(model, state, correct, wrong):
    """FP32 gradient of the local final-norm lens, not the remaining network.

    All messages at this site share one direction, so their projections add.
    Supports the native Llama RMSNorm and the LayerNorm used in operator tests.
    """
    norm = model.model.norm
    value = state.float()
    direction = model.lm_head.weight[correct].float()
    direction = (direction - model.lm_head.weight[wrong].float()) * norm.weight.float()
    if isinstance(norm, torch.nn.LayerNorm):
        value = value - value.mean()
        direction = direction - direction.mean()
        epsilon = norm.eps
    else:
        epsilon = norm.variance_epsilon
    scale = (value.square().mean() + epsilon).sqrt()
    return direction / scale - value * (direction * value).mean() / scale.pow(3)


def vocabulary_logits(model, normalized_states, block_size=4096):
    """FP32 LM-head multiplication, avoiding bf16 quantization of final logits.

    Chunk the vocabulary so a 128k x 4096 float32 weight copy is never allocated.
    Transformer activations remain at the requested model dtype.
    """
    values = []
    head = model.lm_head
    for start in range(0, head.weight.shape[0], block_size):
        stop = start + block_size
        bias = None if head.bias is None else head.bias[start:stop].float()
        values.append(F.linear(normalized_states.float(), head.weight[start:stop].float(), bias))
    return torch.cat(values, dim=-1)
