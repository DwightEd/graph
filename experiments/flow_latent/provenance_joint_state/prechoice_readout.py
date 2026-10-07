"""Shared candidate compatibility readout; all 96 sites and 4096 axes retained."""
import math

import torch
from torch import nn


class CandidateCompatibility(nn.Module):
    """One diagonal coordinate map shared across sites, with signed site weights.

    Input states: [batch, layer, site, coordinate], standardized on fit only.
    Candidates: [candidate, coordinate], frozen normalized output embeddings.
    Positive scores mean source compatibility in program-controlled training.
    """
    def __init__(self, layers=32, sites=3, width=4096):
        super().__init__()
        self.coordinate = nn.Parameter(torch.ones(width))
        self.site_weight = nn.Parameter(torch.full((layers, sites), 1. / (layers * sites)))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, states, candidates):
        return torch.einsum('blsd,ls,d,cd->bc', states, self.site_weight,
                            self.coordinate, candidates) / math.sqrt(states.shape[-1]) + self.bias


def standardize_fit(states, fit_mask):
    """Fixed statistics preserve every raw coordinate; no PCA or head averaging."""
    fit = states[fit_mask]
    mean = fit.mean(dim=0)
    scale = fit.std(dim=0, unbiased=False).clamp_min(1e-3)
    return (states - mean) / scale, mean, scale


def source_balanced_loss(scores, correct, source_ids, ridge_model, ridge=1e-4):
    """Candidate BCE and paired ranking, averaging each source equally."""
    labels = torch.zeros_like(scores).scatter_(1, correct[:, None], 1.)
    binary = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction='none').mean(1)
    right = scores.gather(1, correct[:, None]).squeeze(1)
    wrong = scores.gather(1, (1 - correct)[:, None]).squeeze(1)
    per_prefix = binary + torch.nn.functional.softplus(wrong - right)
    source_losses = [per_prefix[source_ids == identity].mean() for identity in source_ids.unique()]
    penalty = ridge_model.coordinate.square().mean() + ridge_model.site_weight.square().mean()
    return torch.stack(source_losses).mean() + ridge * penalty
