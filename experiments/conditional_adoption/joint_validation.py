"""Finite replay check for frozen joint source/local site derivatives."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_risk_flow.run import load_model
from .mechanism import finite_query_group, prefill_past
from .mechanism_run import write_json


CHECKS = (('12219', 223), ('12297', 106))


def validate(model, record, directory, target):
    with np.load(directory / record['id'] / 'arrays.npz') as arrays:
        actual, rival = int(arrays['actual_id'][target]), int(arrays['rival_id'][target])
        reference = torch.tensor([arrays['actual_logp'][target], arrays['margin'][target]])
    with np.load(directory / record['id'] / 'nodes' / f'{target:06d}.npz') as node:
        messages = torch.from_numpy(node['messages'])
        response = torch.from_numpy(node['response'])
    tokens = record['prompt'] + record['response']['answer_ids']
    position = record['prompt_length'] + target - 1
    past = prefill_past(model, tokens[:position])
    results = []
    for group in (0, 1):
        directions = messages[:, group]
        arguments = (model, past, tokens[position], actual, rival, directions)
        identity = finite_query_group(*arguments, dose=0.)
        assert (identity - reference).abs().max() <= 1e-3
        predicted = response[:, :, group].sum((1, 2))
        for dose in (.01, .001):
            positive = finite_query_group(*arguments, dose=dose)
            negative = finite_query_group(*arguments, dose=-dose)
            measured = (positive - negative) / (2 * dose)
            absolute = (predicted - measured).abs()
            relative = absolute / predicted.abs().clamp_min(1e-6)
            passed = ((absolute <= .02) | (relative <= .02)).all()
            results.append(dict(group=group, dose=dose, predicted=predicted.tolist(),
                measured=measured.tolist(), absolute_error=absolute.tolist(), relative_error=relative.tolist(),
                identity_max_error=float((identity - reference).abs().max()), passed=bool(passed)))
            assert passed, f'Joint current-source directional derivative failed: {results[-1]}'
    return dict(id=record['id'], target=target, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text())
    records = {record['id']: record for record in document['records']}
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(document['model'])
    checks = [validate(model, records[identity], args.capture, target) for identity, target in CHECKS]
    write_json(args.capture / 'JOINT_SITE_FIDELITY.json', dict(status='PASS',
        scope='native finite current-query joint source/local directions, fixed original past KV',
        checks=checks, new_backwards=0, finite_forwards=20))


if __name__ == '__main__':
    main()
