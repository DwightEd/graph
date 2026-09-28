"""Exact matrix-free downstream response on specified intervention directions."""

import torch

from experiments.decision_risk_flow.native import replay

DIRECTIONS = ('prompt_messages', 'history_messages', 'mlp_writes')


def response_tangents(model, cache, tokens, positions, prompt):
    checkpoints = (len(model.model.layers),)

    def apply(scales):
        return replay(model, cache, tokens, positions, checkpoints,
                      gate=dict(scales=scales, prompt_length=prompt))[0]

    baseline = torch.ones(3, device=model.device)
    tangents = []
    for direction in torch.eye(3, device=model.device):
        final, tangent = torch.autograd.functional.jvp(apply, baseline, direction)
        tangents.append(tangent.detach())
    return final.detach(), torch.stack(tangents, 1)


@torch.no_grad()
def full_vocabulary_metric(model, final, tangents, actual, alternatives):
    """Exact three-direction categorical Fisher Gram; no vocabulary sketch."""
    logits = model.lm_head(final).float()
    response = model.lm_head(tangents).float()
    probability = logits.softmax(-1)
    centered = response - (response * probability[:, None]).sum(-1, keepdim=True)
    gram = torch.einsum('bdv,bev,bv->bde', centered, centered, probability)
    rows = torch.arange(len(final), device=final.device)
    margin = response[rows, :, actual] - response[rows, :, alternatives]
    return gram, margin
