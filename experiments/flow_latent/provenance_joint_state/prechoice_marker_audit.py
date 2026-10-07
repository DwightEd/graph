"""Forecasting audit of old post-token attention markers shifted to past states.

This does not test residual/MLP states and does not train a forecasting model.
First-error-clean cohorts remove the easy signal from already wrong history.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.token_backtrace.repetition_teacher import BASE, SelfReadout
from .matrix_audit import OUTPUT, score_metrics, sha256, write_json


def previous_scores(row, teachers):
    values = [teacher.standardized(row) for teacher in teachers]
    nonlinear = [torch.sigmoid(teacher.logits(state)).detach().numpy()
                 for teacher, state in zip(teachers, values)]
    signed = [state.numpy() @ teacher.tangent() for teacher, state in zip(teachers, values)]
    return dict(teacher=np.mean(nonlinear, axis=0), signed=np.mean(signed, axis=0))


def first_error_cohort(row, horizon):
    labels = row['labels'].astype(bool)
    errors = np.flatnonzero(labels)
    if len(errors):
        positions = np.arange(horizon, errors[0] + 1)
    else:
        positions = np.arange(horizon, len(labels))
    return positions, labels[positions]


def horizon_report(rows, features, horizon):
    all_truth, clean_truth = [], []
    all_scores = {name: [] for name in ('teacher', 'signed')}
    clean_scores = {name: [] for name in all_scores}
    first_errors_missing_prompt = 0
    for row in rows:
        labels = row['labels']
        all_truth.extend(labels[horizon:])
        positions, truth = first_error_cohort(row, horizon)
        clean_truth.extend(truth)
        errors = np.flatnonzero(labels)
        first_errors_missing_prompt += bool(len(errors) and errors[0] < horizon)
        for name in all_scores:
            all_scores[name].extend(features[row['id']][name][:-horizon])
            clean_scores[name].extend(features[row['id']][name][positions - horizon])
    all_truth, clean_truth = np.asarray(all_truth), np.asarray(clean_truth)
    result = dict(horizon_tokens=horizon, all_valid_tokens=len(all_truth),
                  all_wrong=int(all_truth.sum()), first_error_clean_tokens=len(clean_truth),
                  first_errors=int(clean_truth.sum()), excluded_first_errors=first_errors_missing_prompt,
                  metrics={})
    for name in all_scores:
        result['metrics'][name] = dict(all_tokens=score_metrics(all_truth, np.asarray(all_scores[name])),
            clean_prefix_first_error=score_metrics(clean_truth, np.asarray(clean_scores[name])))
    return result


def main():
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    rows = torch.load(BASE / 'prepared.pt', weights_only=False)['test']
    features = {row['id']: previous_scores(row, teachers) for row in rows}
    result = dict(scope='frozen naturally supervised attention readouts, forecasting diagnostic only',
        model='Llama3.1 observer on Llama2 generated answers; not same-generator prospective validation',
        position='token t-h observed state to forecast token t; future candidate identity is not an input',
        cohort='clean until first gold token; annotations may precede actual semantic decision',
        first_prompt_state_missing=True, new_fits=0, new_llm_forwards=0,
        horizons=[horizon_report(rows, features, horizon) for horizon in (1, 2, 4, 8)],
        input_hash=sha256(BASE / 'prepared.pt'), code_hash=sha256(Path(__file__)))
    write_json(OUTPUT / 'PRECHOICE_MARKER_AUDIT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
