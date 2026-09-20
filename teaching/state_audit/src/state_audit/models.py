"""The only model-family boundary; common native attention layout, no registry."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_MODELS = ("llama", "mistral", "qwen2")


def load_model(name: str, revision: str, device: str, dtype: str):
    model = (
        AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            dtype=getattr(torch, dtype),
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    layers(model)  # Reject unsupported native layouts at the model boundary.
    scaling = model.config.to_dict().get("rope_scaling") or {}
    if scaling.get("rope_type") in {"dynamic", "longrope"}:
        raise ValueError(
            "Length-dependent RoPE requires stepwise capture, not full-sequence replay"
        )
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for verified character offsets")
    return model, tokenizer


def layers(model):
    if model.config.model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Native state capture supports {SUPPORTED_MODELS}; add an explicit adapter"
        )
    return model.model.layers


def input_tensor(model, token_ids: list[int]):
    return torch.tensor([token_ids], device=model.device, dtype=torch.long)


def readout(model, hidden, target_ids):
    """Project only requested query states; report native, unwarped probabilities."""
    logits = model.lm_head(hidden).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite native logits; this replay is not valid")
    log_probs = logits.log_softmax(-1)
    actual = log_probs.gather(-1, target_ids[..., None]).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return actual, entropy
