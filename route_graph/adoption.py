"""Signed terminal-layer messages and controlled fixed-continuation influence.

These are mechanism measurements, not factuality probabilities. The suffix is
an explicit fp32 surrogate; its mismatch to native inference must be reported.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F


def opposition(contributions):
    """Return opposing signed mass, retaining an undefined zero-mass case."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("expected a finite contribution vector")
    positive = float(np.maximum(values, 0).sum())
    negative = float(np.maximum(-values, 0).sum())
    mass = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "absolute_mass": mass,
        "competition": None if mass == 0 else 2 * min(positive, negative) / mass,
    }


def outgoing_key_mask(length, blocked, activate_at):
    """Cut selected keys only for queries at/after activate_at, preserving self."""
    blocked = list(blocked)
    if not 0 <= activate_at <= length or any(
        k < 0 or k >= activate_at for k in blocked
    ):
        raise ValueError("blocked keys must precede intervention activation")
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    allowed = key <= query
    selected = torch.zeros(length, dtype=torch.bool)
    selected[blocked] = True
    allowed &= ~((query >= activate_at) & selected[None, :])
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]


def grouped_head_messages(attention, values, output_weight, groups):
    """Real V/O messages [group, head, hidden]; no head averaging before signs.

    attention: [heads, keys], values: [keys, KV heads, head dimension].
    groups must partition exactly the causal keys in this query's prefix.
    """
    heads, keys = attention.shape
    if values.ndim != 3 or values.shape[0] != keys or len(groups) != keys:
        raise ValueError("inconsistent attention/value/group shapes")
    kv_heads, head_dim = values.shape[1:]
    if heads % kv_heads or output_weight.shape[1] != heads * head_dim:
        raise ValueError("incompatible GQA or output projection")
    groups = torch.as_tensor(groups, device=attention.device, dtype=torch.long)
    count = int(groups.max()) + 1
    if groups.min() < 0 or not torch.equal(
        groups.unique(), torch.arange(count, device=groups.device)
    ):
        raise ValueError("groups must be contiguous nonnegative IDs")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("invalid attention")
    repeated = values.float().repeat_interleave(heads // kv_heads, dim=1)
    weighted = attention.float().T[:, :, None] * repeated
    contexts = torch.zeros(count, heads, head_dim, device=values.device)
    contexts.index_add_(0, groups, weighted)
    projection = output_weight.float().reshape(-1, heads, head_dim)
    return torch.einsum("ghk,dhk->ghd", contexts, projection)


class TerminalSuffix(torch.nn.Module):
    """Frozen fp32 final MLP + residual + RMSNorm + selected LM output rows."""

    def __init__(self, model, candidates):
        super().__init__()
        layer = model.model.layers[-1]
        self.post_norm = copy.deepcopy(layer.post_attention_layernorm).float()
        self.mlp = copy.deepcopy(layer.mlp).float()
        self.final_norm = copy.deepcopy(model.model.norm).float()
        self.register_buffer(
            "output_weight", model.lm_head.weight[candidates].detach().float().clone()
        )
        if model.lm_head.bias is not None:
            raise ValueError("this pilot requires a bias-free output head")
        self.requires_grad_(False)
        self.eval()

    def forward(self, residual, include_mlp=True):
        state = residual.float()
        if include_mlp:
            state = state + self.mlp(self.post_norm(state))
        return F.linear(self.final_norm(state), self.output_weight)


def probe_messages(suffix, residual, head_messages, native_logits):
    """Validate local signed derivatives with finite suffix message attenuation.

    Candidate 0 and 1 must be native top-1 and top-2, fixed before intervention.
    Returns JSON summary and arrays, including every candidate and head sign.
    """
    messages = head_messages.sum(1).detach()
    state = residual.detach().float().clone().requires_grad_(True)
    logits = suffix(state)
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("at least two fixed candidates required")
    contrasts = logits[0] - logits[1:]
    identity = torch.eye(len(contrasts), device=state.device)
    gradients = []
    for start in range(0, len(contrasts), 16):
        gradients.append(
            torch.autograd.grad(
                contrasts,
                state,
                grad_outputs=identity[start : start + 16],
                is_grads_batched=True,
                retain_graph=True,
            )[0]
        )
    gradients = torch.cat(gradients)
    signed_heads = torch.einsum("cd,ghd->cgh", gradients, head_messages)
    signed_groups = signed_heads.sum(-1)
    bare = suffix(state, include_mlp=False)
    bare_gradient = torch.autograd.grad(bare[0] - bare[1], state)[0]
    bare_signed = messages @ bare_gradient
    native_logits = native_logits.detach().float()
    finite, finite_arrays = {}, {}
    with torch.no_grad():
        base = suffix(state.detach())
        for fraction in (0.01, 0.1, 1.0):
            altered = suffix(state.detach()[None, :] - fraction * messages)
            all_delta = (base[0] - base[1:])[:, None] - (
                altered[:, :1] - altered[:, 1:]
            ).T
            all_predicted = fraction * signed_groups
            finite_arrays[f"finite_margin_changes_{fraction}"] = all_delta.cpu().numpy()
            delta = all_delta[0]
            predicted = fraction * signed_groups[0]
            finite[str(fraction)] = {
                "actual_margin_change": delta.cpu().tolist(),
                "predicted_margin_change": predicted.detach().cpu().tolist(),
                "mean_absolute_error": float((delta - predicted).abs().mean()),
                "absolute_effect_mass": float(delta.abs().sum()),
                "relative_l1_error": float(
                    (delta - predicted).abs().sum() / delta.abs().sum().clamp_min(1e-12)
                ),
                "all_candidate_relative_l1_error": float(
                    (all_delta - all_predicted).abs().sum()
                    / all_delta.abs().sum().clamp_min(1e-12)
                ),
            }
    summary = {
        "native_margin": float(native_logits[0] - native_logits[1]),
        "surrogate_margin": float(contrasts[0].detach()),
        "surrogate_max_logit_error": float(
            (logits.detach() - native_logits).abs().max()
        ),
        "surrogate_pool_top1_matches_native": bool(logits.argmax() == 0),
        "signed_group_contributions": signed_groups[0].detach().cpu().tolist(),
        "without_final_mlp_contributions": bare_signed.detach().cpu().tolist(),
        "opposition": opposition(signed_groups[0].detach().cpu().numpy()),
        "finite_attenuation": finite,
    }
    arrays = {
        **finite_arrays,
        "group_messages": messages.cpu().numpy(),
        "candidate_group_head_contributions": signed_heads.detach().cpu().numpy(),
        "candidate_logits": logits.detach().cpu().numpy(),
        "native_candidate_logits": native_logits.cpu().numpy(),
    }
    return summary, arrays


def distribution_effect(baseline, altered, saved_tokens):
    """Full-vocabulary JS and fixed-token log-probability effects, in nats."""
    if baseline.shape != altered.shape or baseline.shape[0] != len(saved_tokens):
        raise ValueError("inconsistent logit/token shapes")
    logp, logq = baseline.float().log_softmax(-1), altered.float().log_softmax(-1)
    mixture = torch.logaddexp(logp, logq) - np.log(2)
    js = 0.5 * (
        (logp.exp() * (logp - mixture)).sum(-1)
        + (logq.exp() * (logq - mixture)).sum(-1)
    )
    rows = torch.arange(len(saved_tokens), device=baseline.device)
    tokens = torch.as_tensor(saved_tokens, device=baseline.device)
    return {
        "js_nats": js.clamp_min(0).cpu().tolist(),
        "argmax_changed": (baseline.argmax(-1) != altered.argmax(-1)).cpu().tolist(),
        "saved_token_logp_change": (logq[rows, tokens] - logp[rows, tokens])
        .cpu()
        .tolist(),
        "altered_entropy_nats": (-(logq.exp() * logq).sum(-1)).cpu().tolist(),
    }
