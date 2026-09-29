"""Score every original token under full and reset source conditions."""
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL
from .native import batches, full_states, reset_states
from .readout import distribution_scores

METHODS = tuple(f'{mode}_{name}' for mode in ('full', 'reset') for name in (
    'cad_tail', 'cad_nll', 'source_tail', 'source_ratio', 'source_nll', 'confidence_tail'))


def append_arrays(destination, arrays, prefix):
    for name, value in arrays.items():
        destination.setdefault(prefix + name, []).append(value.cpu().numpy())


@torch.no_grad()
def capture_record(model, row, directory, window, batch_size):
    prompt, answer = row['prompt'], row['response']['answer_ids']
    masks = [[1] * len(prompt), [int(not source) for source in row['source']['source_mask']]]
    assert masks[1][0], 'source exclusion must retain the BOS key'
    states, caches = [], []
    for mask in masks:
        hidden, cache = full_states(model, prompt, answer, mask)
        states.append(hidden)
        caches.append(cache)
    scores, details, error = {}, {}, 0.
    for targets in batches(len(answer), window, batch_size):
        target_ids = torch.tensor([answer[target] for target in targets], device=model.device)
        for mode in ('full', 'reset'):
            logits = []
            for condition in range(2):
                hidden = states[condition][targets].to(model.device) if mode == 'full' else reset_states(
                    model, prompt, answer, masks[condition], caches[condition], targets, window)
                value = model.lm_head(hidden).float()
                assert torch.isfinite(value).all(), ('nonfinite logits', row['key'], mode, targets)
                checked = [target for target in targets if target <= window]
                if mode == 'reset' and checked:
                    original = model.lm_head(states[condition][checked].to(model.device)).float()
                    difference = (value[:len(checked)] - original).abs()
                    assert torch.isfinite(difference).all(), ('nonfinite replay', row['key'])
                    error = max(error, float(difference.max()))
                logits.append(value)
            measured, candidates = distribution_scores(*logits, target_ids)
            append_arrays(scores, measured, mode + '_')
            append_arrays(details, candidates, mode + '_')
    assert error < .005, ('prefix/cache mismatch', row['key'], error)
    scores = {name: np.concatenate(value) for name, value in scores.items()}
    details = {name: np.concatenate(value) for name, value in details.items()}
    np.savez_compressed(directory / 'scores.npz', token_ids=answer, **scores)
    np.savez_compressed(directory / 'candidates.npz', **details)
    calls = 2 + 2 * len(list(batches(len(answer), window, batch_size)))
    write_json(directory / 'capture.json', dict(tokens=len(answer), max_prefix_error=error,
        model_calls=calls, labels_used=False, status='complete'))
    return dict(model_calls=calls, max_prefix_error=error)


def capture(output):
    plan = read_json(output / 'manifest.json')
    torch.manual_seed(42)
    model = load_model(MODEL)
    started = perf_counter()
    audits = []
    for index, row in enumerate(plan['records']):
        audit = capture_record(model, row, output / row['key'], plan['window'], plan['batch_size'])
        audits.append(audit)
        print('capture', index + 1, len(plan['records']), row['key'],
              round(perf_counter() - started, 1), flush=True)
    write_json(output / 'capture_complete.json', dict(status='complete', answers=len(audits),
        tokens=sum(len(row['response']['answer_ids']) for row in plan['records']),
        model_calls=sum(audit['model_calls'] for audit in audits),
        max_prefix_error=max(audit['max_prefix_error'] for audit in audits),
        seconds=perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        methods=METHODS, primary=plan['primary'], labels_used=False))
