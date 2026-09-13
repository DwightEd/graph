"""Small source-pointer reconstruction head over frozen masked text features.

Source graph identity supplies training positives and confusable negatives.
This head returns reusable owner candidates, never support/conflict labels.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SourceRelMini(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=128, num_types=7, temperature=.07):
        super().__init__()
        if min(input_dim, hidden_dim, num_types) < 1 or temperature <= 0:
            raise ValueError("positive dimensions and temperature required")
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.candidate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.type_scale = nn.Embedding(num_types, hidden_dim)
        nn.init.ones_(self.type_scale.weight)
        self.temperature = float(temperature)

    def forward(self, queries, candidates, query_types, valid=None):
        if (queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]
                or query_types.shape != (len(queries),)):
            raise ValueError("expected query/candidate feature matrices and one broad type per query")
        q = F.normalize(self.query(queries.float()) * self.type_scale(query_types), dim=-1)
        c = F.normalize(self.candidate(candidates.float()), dim=-1)
        scores = q @ c.T / self.temperature
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != scores.shape or not valid.any(-1).all():
                raise ValueError("every query requires at least one allowed source candidate")
            scores = scores.masked_fill(~valid, -torch.inf)
        return scores


def owner_nce(scores, positives):
    """Multi-positive source-pointer objective; no hallucination target."""
    if (positives.dtype != torch.bool or positives.shape != scores.shape or scores.ndim != 2
            or not positives.any(-1).all() or not torch.isfinite(scores[positives]).all()
            or torch.isnan(scores).any() or torch.isposinf(scores).any()):
        raise ValueError("each source query needs finite, allowed positive occurrences")
    positive = scores.masked_fill(~positives, -torch.inf)
    return (torch.logsumexp(scores, -1) - torch.logsumexp(positive, -1)).mean()


def source_role_mass(scores, candidate_families):
    """Aggregate candidate mass by source role, without another training loss."""
    if len(candidate_families) != scores.shape[-1]:
        raise ValueError("source family roster differs from candidate dimension")
    families = sorted(set(candidate_families))
    weights = scores.softmax(-1)
    values = torch.stack([weights[:, [i for i, kind in enumerate(candidate_families) if kind == family]].sum(-1)
        for family in families], dim=-1)
    return families, values


def owner_beam(scores, *, topk=5):
    """Candidate identities and normalized ranking mass; no truth/NULL claims."""
    if topk < 1 or scores.ndim != 2 or not torch.isfinite(scores).any(-1).all():
        raise ValueError("nonempty finite candidate rows required")
    mass = scores.softmax(-1)
    records = []
    for row in range(len(scores)):
        # Stable ties preserve the explicitly supplied candidate ordering.
        order = torch.argsort(scores[row], descending=True, stable=True)
        order = [int(i) for i in order if torch.isfinite(scores[row, i])][:topk]
        records.append({"candidate_indices": order, "scores": [float(scores[row, i].detach()) for i in order],
            "ranking_mass": [float(mass[row, i].detach()) for i in order],
            "status": "candidate_supply_only", "source_reuse_allowed": True,
            "factual_status": "unverified", "native_certificate_count": 0})
    return records
