"""All first-token probabilities share one prefix forward and one normalizer."""

import torch

from .operators import vocabulary_logits


def read_prefix(model, probe, interventions=(), restore=None):
    from .native import forward
    native_logits, run = forward(model, probe['prefix_ids'], probe, interventions, restore)
    with torch.inference_mode():
        logits = vocabulary_logits(model, run.final_normalized[-1])
    log_prob = logits.double().log_softmax(-1)
    candidates = [tokens[0] for tokens in probe['candidates']]
    correct, wrong = [float(log_prob[token]) for token in candidates]
    record = dict(next_margin=correct - wrong, correct_first_logp=correct, wrong_first_logp=wrong,
        next_top1=int(logits.argmax()), candidate_first_ids=candidates,
        raw_native_next_margin=float(native_logits[-1, candidates[0]] - native_logits[-1, candidates[1]]),
        lm_head_rounding_error=float((native_logits[-1] - logits).abs().max()),
        first_metric='same_prefix_fp32_head_float64_log_softmax')
    return record, run, log_prob


def continuation_logps(model, probe, candidate, first_logp, interventions, restore, branch):
    from .native import forward
    # The candidate is forced only after the intervention query. No later cut is added.
    _, run = forward(model, probe['prefix_ids'] + candidate[:-1], probe,
                     interventions, restore, branch=branch)
    start = len(probe['prefix_ids']) - 1
    states = run.final_normalized[start:start + len(candidate)]
    with torch.inference_mode():
        log_prob = vocabulary_logits(model, states).double().log_softmax(-1)
    target = torch.tensor(candidate, device=log_prob.device)
    selected = log_prob[torch.arange(len(candidate), device=target.device), target]
    branch_difference = float(selected[0]) - first_logp
    selected = selected.clone()
    selected[0] = first_logp
    return selected.cpu().tolist(), branch_difference, run.message_writes


def evaluate_candidates(model, probe, interventions=(), restore=None, sequence=True, reference_logp=None):
    record, run, log_prob = read_prefix(model, probe, interventions, restore)
    run.branch_message_writes = {'prefix': run.message_writes}
    if reference_logp is not None:
        reference = reference_logp.to(log_prob)
        record['vocabulary_kl_from_full'] = float((reference.exp() * (reference - log_prob)).sum())
    if sequence:
        for name, candidate in zip(('correct', 'wrong'), probe['candidates']):
            first = record[name + '_first_logp']
            logs, disagreement, writes = continuation_logps(
                model, probe, candidate, first, interventions, restore, name)
            run.branch_message_writes[name] = writes
            record[name + '_token_logps'] = logs
            record[name + '_logp'] = sum(logs)
            record[name + '_mean_logp'] = sum(logs) / len(logs)
            record[name + '_tokens'] = len(logs)
            record[name + '_branch_first_error'] = disagreement
        record['sequence_margin'] = record['correct_logp'] - record['wrong_logp']
        record['mean_margin'] = record['correct_mean_logp'] - record['wrong_mean_logp']
        record['tail_margin'] = record['sequence_margin'] - record['next_margin']
    run.prefix_log_prob = log_prob.detach().cpu()
    return record, run
