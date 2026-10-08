"""Frozen source-derived self-supervised readout; no natural-label fitting.

Candidates are native top-K tokens followed by the actual observed token. The
scores express compatibility disagreement, not calibrated factual probabilities.
"""
import math

import torch

from ..ordered_source_transport.readout import row_permutations, transform_view

VARIANTS = ('ordered', 'mean', 'single', 'shuffled', 'drop_source', 'linear')


def load_checkpoint(path):
    """Read the existing frozen source-program checkpoint onto CPU."""
    return torch.load(path, map_location='cpu', weights_only=True)


def prediction_rows(prompt_length, token_count, window=8):
    """Return capture start and causal rows for zero-based answer targets t.

    Target y_t is predicted at q=P+t-1. Capturing from P-W through the final
    prediction row gives exactly T+W-1 states; y_t itself is outside its window.
    """
    if prompt_length < window or token_count < 1:
        raise ValueError('Capture requires P >= W and at least one answer token')
    rows = torch.arange(prompt_length - 1, prompt_length + token_count - 1)
    return prompt_length - window, rows


def candidate_vector(standardized_windows, checkpoint):
    """Contract [B,W,L,S,D] into [B,D], including the exact 1/sqrt(D).

    Candidate scores are vector dot candidate embeddings plus checkpoint bias.
    Linear checkpoints have a frozen zero transition, as in the original fit.
    """
    weights = checkpoint['state_dict']
    device = standardized_windows.device
    level_weights = weights['level'].to(device=device, dtype=torch.float32)
    transition_weights = weights['transition'].to(device=device, dtype=torch.float32)
    coordinates = weights['coordinate'].to(device=device, dtype=torch.float32)
    level = torch.einsum('btlsd,tls->bd', standardized_windows, level_weights)
    neighbors = standardized_windows[:, :-1] * standardized_windows[:, 1:]
    transition = torch.einsum('btlsd,tls->bd', neighbors, transition_weights)
    return (level + transition) * coordinates / math.sqrt(standardized_windows.shape[-1])


def _validate_inputs(states, vectors, ids, actual, logp, checkpoint, start, variant):
    """Check interpretation-critical contracts once at the scoring boundary."""
    weights = checkpoint['state_dict']
    window, layers, sites = weights['level'].shape
    width = weights['coordinate'].numel()
    count, candidates, candidate_width = vectors.shape
    if variant not in VARIANTS or variant != checkpoint['variant']:
        raise ValueError('Variant must match a supported frozen checkpoint')
    if tuple(states.shape[1:]) != (layers, sites, width) or candidate_width != width:
        raise ValueError('State and candidate coordinates must match the checkpoint')
    if ids.shape != (count, candidates) or logp.shape != ids.shape or actual.shape != (count,):
        raise ValueError('Candidate IDs/logp and actual IDs must align with embeddings')
    if count < 1 or start < 0 or start + count + window - 1 > len(states):
        raise ValueError('Requested token windows are outside the captured answer')
    if not torch.equal(ids[:, -1], actual):
        raise ValueError('The last candidate must be the actual observed token')
    if torch.any((ids == actual[:, None]).all(dim=1)):
        raise ValueError('Every target needs a rival distinct from the actual token')
    if torch.any((ids[:, :-1] == ids[:, :1]).all(dim=1)):
        raise ValueError('Native top-K needs a rival distinct from its greedy token')
    if variant == 'linear' and torch.count_nonzero(weights['transition']) != 0:
        raise ValueError('Linear checkpoint must have a frozen zero transition')
    finite = [states, vectors, logp, checkpoint['mean'], checkpoint['scale'], *weights.values()]
    if any(not torch.isfinite(value).all() for value in finite):
        raise ValueError('Scoring inputs and frozen checkpoint must be finite')
    if torch.any(checkpoint['scale'] <= 0):
        raise ValueError('Frozen standardization scale must be positive')
    return window


def _candidate_gaps(scores, candidate_ids, actual_ids, candidate_logp):
    observed_mask = candidate_ids == actual_ids[:, None]
    rival_scores, rival_columns = scores.masked_fill(observed_mask, -torch.inf).max(dim=1)
    combined = scores + candidate_logp
    logp_rival_scores, logp_columns = combined.masked_fill(observed_mask, -torch.inf).max(dim=1)

    # Index 0 is native greedy. Its pre-choice diagnostic may only compete
    # against native top-K; the appended observed token is not yet known.
    greedy_ids = candidate_ids[:, 0]
    native_mask = candidate_ids == greedy_ids[:, None]
    native_mask[:, -1] = True
    native_rival_scores, native_columns = scores.masked_fill(native_mask, -torch.inf).max(dim=1)
    gather_ids = lambda columns: candidate_ids.gather(1, columns[:, None]).squeeze(1)
    return dict(gap=rival_scores - scores[:, -1],
                logp_gap=logp_rival_scores - combined[:, -1],
                native_gap=native_rival_scores - scores[:, 0],
                rival_ids=gather_ids(rival_columns),
                logp_rival_ids=gather_ids(logp_columns),
                native_rival_ids=gather_ids(native_columns),
                candidate_scores=scores)


@torch.no_grad()
def score_windows(states, candidate_vectors, candidate_ids, actual_ids,
                  candidate_logp, checkpoint, variant=None, start=0, permutations=None):
    """Score a token chunk against its independently supplied candidates.

    states is raw full-answer [T+W-1,L,S,D]. Candidate vectors [B,C,D] are
    normalized lm_head rows, IDs/logp are [B,C], actual IDs are [B]. start is
    the global answer-token offset. Output gap/logp_gap/native_gap are signed
    and unclipped: max distinct rival minus actual score. native_gap compares
    native greedy only with the original top-K, excluding the appended row.

    For shuffled, pass the [B,W] slice of one per-answer permutation tensor.
    If omitted, deterministic seed73 permutations are generated for all T
    before slicing, so changing chunk size or readout seed cannot change them.
    """
    variant = checkpoint['variant'] if variant is None else variant
    states = torch.as_tensor(states, dtype=torch.float32)
    device = states.device
    vectors = torch.as_tensor(candidate_vectors, dtype=torch.float32, device=device)
    ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=device)
    actual = torch.as_tensor(actual_ids, dtype=torch.long, device=device)
    logp = torch.as_tensor(candidate_logp, dtype=torch.float32, device=device)
    window = _validate_inputs(states, vectors, ids, actual, logp, checkpoint, start, variant)
    count = len(vectors)
    mean = checkpoint['mean'].to(device=device, dtype=torch.float32)
    scale = checkpoint['scale'].to(device=device, dtype=torch.float32)
    selected = (states[start:start + count + window - 1] - mean) / scale
    windows = selected.unfold(0, window, 1).movedim(-1, 1)

    if variant == 'shuffled':
        if permutations is None:
            total = len(states) - window + 1
            permutations = row_permutations(total, window, seed=73)[start:start + count]
        permutations = torch.as_tensor(permutations, dtype=torch.long, device=device)
        expected = torch.arange(window, device=device).expand(count, window)
        if permutations.shape != (count, window) or not torch.equal(permutations.sort(dim=1).values, expected):
            raise ValueError('Shuffled permutations must contain each window position once')
        if torch.any(permutations[:, -1] != window - 1):
            raise ValueError('Shuffled view must preserve the current prediction row')
    windows = transform_view(windows, variant, permutations)
    vector = candidate_vector(windows, checkpoint)
    bias = checkpoint['state_dict']['bias'].to(device=device, dtype=torch.float32)
    scores = torch.einsum('bd,bcd->bc', vector, vectors) + bias
    if not torch.isfinite(scores).all():
        raise ValueError('Frozen scoring produced nonfinite candidate scores')
    return _candidate_gaps(scores, ids, actual, logp)
