"""Shared observer loader retained at its existing research import path."""

import torch
from transformers import AutoModelForCausalLM

from .precision import enable_fp32_execution


def load_model(path):
    """Keep frozen BF16 weights and FP32 native operations for input gradients."""
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        local_files_only=True,
    ).to('cuda:0').eval().requires_grad_(False)
    return enable_fp32_execution(model)
