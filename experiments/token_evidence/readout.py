"""Full-vocabulary, target-specific scores; no label access or token averaging."""
import math

import torch


def surprise_tail(logits, targets):
    """Negative log mid-tail mass of tokens no likelier than the actual target."""
    actual = logits.gather(1, targets[:, None])
    weights = torch.full_like(logits, -torch.inf)
    weights[logits < actual] = 0.
    weights[logits == actual] = -math.log(2)
    return logits.logsumexp(-1) - (logits + weights).logsumexp(-1)


def distribution_scores(source_logits, masked_logits, targets):
    source_logp = source_logits.log_softmax(-1)
    masked_logp = masked_logits.log_softmax(-1)
    contrast = 2 * source_logits - masked_logits
    actual = targets[:, None]
    ratio = masked_logp - source_logp
    actual_ratio = ratio.gather(1, actual)
    # Upper-tail anomaly of the source likelihood ratio under p_source.
    weights = torch.full_like(ratio, -torch.inf)
    weights[ratio > actual_ratio] = 0.
    weights[ratio == actual_ratio] = -math.log(2)
    source_tail = -(source_logp + weights).logsumexp(-1)
    scores = dict(cad_tail=surprise_tail(contrast, targets),
        cad_nll=contrast.logsumexp(-1) - contrast.gather(1, actual)[:, 0],
        source_tail=source_tail, source_ratio=actual_ratio[:, 0],
        source_nll=-source_logp.gather(1, actual)[:, 0],
        confidence_tail=surprise_tail(source_logits, targets))
    candidates = torch.cat((source_logits.topk(8, -1).indices,
                            masked_logits.topk(8, -1).indices,
                            contrast.topk(8, -1).indices, actual), -1)
    details = dict(ids=candidates, source_logp=source_logp.gather(1, candidates),
                   masked_logp=masked_logp.gather(1, candidates),
                   contrast_logp=contrast.log_softmax(-1).gather(1, candidates))
    return scores, details
