"""Small relation classifier; natural hallucination labels never enter training."""

import torch
from torch import nn
import torch.nn.functional as F


CLASS_NAMES = (
    "real",
    "head_identity",
    "source_head",
    "source_time",
    "previous_time",
)


class StructuredCompatibility(nn.Module):
    def __init__(self, layers, heads, views, hidden):
        super().__init__()
        self.layers = layers
        self.heads = heads
        self.views = views

        layer_input = 2 * heads * views + 2 * views
        self.layer_weight = nn.Parameter(torch.empty(layers, layer_input, hidden))
        self.layer_bias = nn.Parameter(torch.zeros(layers, hidden))
        for layer in range(layers):
            nn.init.xavier_uniform_(self.layer_weight[layer])

        global_input = layers * hidden + hidden
        self.readout = nn.Sequential(
            nn.Linear(global_input, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, len(CLASS_NAMES)),
        )

    def layer_features(self, current, previous):
        delta = current - previous
        current_mean = current.mean(dim=2)
        delta_mean = delta.mean(dim=2)

        current_contrast = current - current_mean.unsqueeze(2)
        delta_contrast = delta - delta_mean.unsqueeze(2)

        return torch.cat(
            (
                current_contrast.flatten(2),
                delta_contrast.flatten(2),
                current_mean,
                delta_mean,
            ),
            dim=-1,
        )

    def forward(self, current, previous):
        features = self.layer_features(current, previous)
        hidden = torch.einsum("bli,lid->bld", features, self.layer_weight)
        hidden = F.gelu(hidden + self.layer_bias)

        pooled = torch.cat(
            (hidden.flatten(1), hidden.mean(dim=1)),
            dim=1,
        )
        return self.readout(pooled)


def roll_heads(values, shifts):
    result = values.clone()
    for layer, shift in enumerate(shifts.tolist()):
        result[:, layer] = torch.roll(
            values[:, layer],
            shifts=int(shift),
            dims=1,
        )
    return result


def source_fraction(values):
    prompt_mass = values[..., 0] + values[..., 1]
    return torch.where(
        prompt_mass > 1e-8,
        values[..., 0] / prompt_mass,
        torch.zeros_like(prompt_mass),
    )


def replace_source_fraction(current, fraction):
    result = current.clone()
    prompt_mass = current[..., 0] + current[..., 1]
    result[..., 0] = prompt_mass * fraction
    result[..., 1] = prompt_mass * (1 - fraction)
    return result


def normalize_routes(values, center, scale):
    return (values - center) / scale


def corruption_batch(
    current,
    previous,
    source_donor,
    previous_donor,
    center,
    scale,
):
    device = current.device
    layers = current.shape[1]
    heads = current.shape[2]

    normalized_current = normalize_routes(current, center, scale)
    normalized_previous = normalize_routes(previous, center, scale)
    normalized_previous_donor = normalize_routes(
        previous_donor,
        center,
        scale,
    )

    shifts = torch.randint(1, heads, (layers,), device=device)
    head_current = roll_heads(normalized_current, shifts)
    head_previous = roll_heads(normalized_previous, shifts)

    source_shifts = torch.randint(1, heads, (layers,), device=device)
    current_fraction = source_fraction(current)
    source_head_fraction = roll_heads(current_fraction, source_shifts)
    source_head = replace_source_fraction(current, source_head_fraction)
    source_head = normalize_routes(source_head, center, scale)

    donor_fraction = source_fraction(source_donor)
    source_time = replace_source_fraction(current, donor_fraction)
    source_time = normalize_routes(source_time, center, scale)

    all_current = torch.cat(
        (
            normalized_current,
            head_current,
            source_head,
            source_time,
            normalized_current,
        ),
        dim=0,
    )
    all_previous = torch.cat(
        (
            normalized_previous,
            head_previous,
            normalized_previous,
            normalized_previous,
            normalized_previous_donor,
        ),
        dim=0,
    )

    batch = current.shape[0]
    labels = torch.arange(
        len(CLASS_NAMES),
        device=device,
    ).repeat_interleave(batch)
    return all_current, all_previous, labels


def compatibility_scores(logits):
    real = logits[:, :1]
    components = logits[:, 1:] - real
    overall = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
    return overall, components
