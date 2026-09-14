"""Attention loading and extraction helpers."""

import numpy as np


def load_model(model_name_or_path, device="cpu", dtype="float32", local_files_only=True):
    import torch
    from transformers import AutoModelForCausalLM
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, local_files_only=local_files_only
    )
    return model.to(device).eval()


def select_middle_layers(num_layers, start_ratio=0.25, end_ratio=0.75):
    start = int(np.floor(num_layers * start_ratio))
    end = int(np.ceil(num_layers * end_ratio))
    return list(range(start, max(start + 1, end)))


def capture_attention(model, input_ids, layers=None, device=None):
    import torch
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    ids = ids.unsqueeze(0) if ids.ndim == 1 else ids
    device = device or next(model.parameters()).device
    with torch.inference_mode():
        output = model(input_ids=ids.to(device), output_attentions=True, use_cache=False)
    attentions = output.attentions
    selected = select_middle_layers(len(attentions)) if layers is None else list(layers)
    return np.stack([attentions[i][0].detach().float().cpu().numpy() for i in selected], axis=0)


def aggregate_attention(attention, layers=None, heads=None, causal=True):
    values = np.asarray(attention, dtype=float)
    values = values[None] if values.ndim == 3 else values
    layer_idx = list(range(values.shape[0])) if layers is None else list(layers)
    head_idx = list(range(values.shape[1])) if heads is None else list(heads)
    selected = values[np.ix_(layer_idx, head_idx)].mean(axis=(0, 1))
    if causal:
        selected = np.tril(selected, k=-1)
    totals = selected.sum(axis=1, keepdims=True)
    return np.divide(selected, totals, out=np.zeros_like(selected), where=totals != 0)
