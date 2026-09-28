"""Exact native-head factorization; no learned projection or head pooling."""

import torch


def projected_norm(values, output_weight):
    """values [H,N,d], output_weight [H,D,d] -> ||W_O V|| [H,N]."""
    gram = output_weight.transpose(1, 2) @ output_weight
    squared = ((values @ gram) * values).sum(-1)
    return squared.clamp_min(0).sqrt()


def group_messages(attention, values, output_weight, masks):
    """Retain [group,H,T,d] vectors and [group,T,H,H] output Gram matrices."""
    magnitude = attention * projected_norm(values, output_weight)[:, None]
    factors, grams, masses = [], [], []
    for mask in masks:
        selected = attention * mask[None, None]
        factor = selected @ values
        output = torch.einsum('htd,hkd->thk', factor, output_weight)
        factors.append(factor)
        grams.append(output @ output.transpose(1, 2))
        masses.append((magnitude * mask[None, None]).sum(-1))
    return torch.stack(factors), torch.stack(grams), torch.stack(masses), magnitude


def norm_route(masses):
    """masses [layer,group,head,token], groups source/history/other prompt."""
    denominator = masses.sum((1, 2))
    difference = (masses[:, 1] - masses[:, 0]).sum(1)
    return (difference / denominator.clamp_min(1e-30)).mean(0)
