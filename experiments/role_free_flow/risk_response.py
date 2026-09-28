"""Research primitives; no trained detector or native-LLM capture pipeline.

Risk derivatives explain a fitted probe, not factual correctness. The generator
must provide downstream gradients from the same forward for each target scalar.
"""

import torch
from torch import nn


def rms_scale(values):
    return values / values.square().mean(-1, keepdim=True).add(1e-8).sqrt()


class CandidateRiskProbe(nn.Module):
    """Condition a raw post-block query state on the actually sampled token."""

    def __init__(self, hidden_size, checkpoints=3, rank=8):
        super().__init__()
        state_size = hidden_size * checkpoints
        self.state_linear = nn.Linear(state_size, 1)
        self.token_linear = nn.Linear(hidden_size, 1, bias=False)
        self.state_factor = nn.Linear(state_size, rank, bias=False)
        self.token_factor = nn.Linear(hidden_size, rank, bias=False)

    def forward(self, raw_states, sampled_embedding):
        # Final checkpoint must precede the generator's final normalization.
        state = rms_scale(raw_states).flatten(-2)
        token = rms_scale(sampled_embedding)
        interaction = (self.state_factor(state) * self.token_factor(token)).sum(-1)
        linear = self.state_linear(state) + self.token_linear(token)
        return linear.squeeze(-1) + interaction


def message_responses(attention, expanded_values, output_weight, residual,
                      choice_gradient, risk_gradient):
    """Current-query gate responses without materializing [head,key,hidden].

    attention: [heads, keys]; values: [heads, keys, head_dim], already GQA-expanded.
    output_weight: [hidden, heads * head_dim]; other vectors: [hidden].
    Gradients are d(native margin)/d(attention write) and d(probe logit)/d(write).
    Both must include the native downstream computation, including final norm.
    """
    heads, _, head_dim = expanded_values.shape
    weights = output_weight.reshape(-1, heads, head_dim).permute(1, 0, 2)
    choice_direction = torch.einsum('d,hdk->hk', choice_gradient, weights)
    risk_direction = torch.einsum('d,hdk->hk', risk_gradient, weights)
    choice = attention * torch.einsum('hkd,hd->hk', expanded_values, choice_direction)
    risk = attention * torch.einsum('hkd,hd->hk', expanded_values, risk_direction)

    gram = torch.einsum('hdi,hdj->hij', weights, weights)
    squared_norm = torch.einsum('hki,hij,hkj->hk', expanded_values, gram, expanded_values)
    magnitude = attention.abs() * squared_norm.clamp_min(0).sqrt()
    access = magnitude / (torch.linalg.vector_norm(residual) + 1e-12)
    return access, choice, risk


def response_summary(access, choice, risk):
    """Eight fixed statistics for one address group and layer band.

    Caller selects group membership structurally; no labels or semantic masks.
    All tensors cover the same selected physical messages. Empty groups give 0.
    """
    choice_norm = torch.linalg.vector_norm(choice)
    risk_norm = torch.linalg.vector_norm(risk)
    choice_positive = torch.linalg.vector_norm(choice.clamp_min(0))
    choice_negative = torch.linalg.vector_norm((-choice).clamp_min(0))
    risk_positive = torch.linalg.vector_norm(risk.clamp_min(0))
    risk_negative = torch.linalg.vector_norm((-risk).clamp_min(0))
    denominator = (choice_norm * risk_norm).clamp_min(1e-12)
    alignment = (choice * risk).sum() / denominator
    joint_increase = (choice.clamp_min(0) * risk.clamp_min(0)).sum() / denominator
    choice_up_risk_down = (choice.clamp_min(0) * (-risk).clamp_min(0)).sum() / denominator
    return torch.stack((access.sum(), choice_positive, choice_negative,
                        risk_positive, risk_negative, alignment,
                        joint_increase, choice_up_risk_down))
