"""Validate and measure source events' downstream influence, without fixed past KV."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from .measure import MODEL
from .native import intervene, observe


def measure_probe(model, row, probe, baseline, directory, index, dose_size=.05):
    prompt = len(row['prompt'])
    answer = row['response']['answer_ids']
    tokens = row['prompt'] + answer[:-1]
    probe = dict(probe, receiver=prompt - 1 + probe['target'])
    arrays = {}
    audits = []
    for treatment in probe['treatments']:
        runs = {}
        masses = {}
        for dose in (0., -dose_size, dose_size):
            with intervene(model, probe, treatment, dose) as patch:
                runs[dose] = observe(model, tokens, prompt, answer, baseline['alternatives'])
                masses[dose] = patch.mass
        zero = runs[0.]
        reconstruction = float(np.max(np.abs(zero['margin'] - baseline['margin'])))
        before = slice(0, probe['target'])
        causal = max(float(np.max(np.abs(runs[dose]['margin'][before] - zero['margin'][before]), initial=0))
                     for dose in (-dose_size, dose_size))
        assert reconstruction < .005, ('reconstruction', row['key'], reconstruction)
        assert causal < 1e-5, ('future intervention changed past', row['key'], causal)
        name = treatment['kind']
        for field in ('margin', 'logp'):
            arrays[f'{name}_{field}_slope'] = (runs[dose_size][field] - runs[-dose_size][field]) / (2 * dose_size)
            arrays[f'{name}_{field}_plus'] = runs[dose_size][field] - zero[field]
            arrays[f'{name}_{field}_minus'] = runs[-dose_size][field] - zero[field]
        delta = runs[dose_size]['hidden'] - runs[-dose_size]['hidden']
        arrays[f'{name}_hidden_slope'] = np.linalg.norm(delta, axis=-1) / (2 * dose_size) / np.linalg.norm(zero['hidden'], axis=-1)
        finite = float(arrays[f'{name}_margin_slope'][probe['target']])
        audits.append(dict(**treatment, finite_slope=finite, absolute_error=abs(finite - treatment['slope']),
            reconstruction=reconstruction, earlier_effect=causal, mass_by_dose=masses))
    np.savez_compressed(directory / f'propagation_{index}.npz', **arrays)
    write_json(directory / f'propagation_{index}.json', dict(probe=probe, audits=audits, dose=dose_size))
    return audits


def run_record(model, row, directory):
    answer = row['response']['answer_ids']
    baseline = observe(model, row['prompt'] + answer[:-1], len(row['prompt']), answer)
    np.savez_compressed(directory / 'baseline.npz', **baseline)
    audits = []
    for index, probe in enumerate(read_json(directory / 'probes.json')):
        audits.extend(measure_probe(model, row, probe, baseline, directory, index))
        print('propagation', row['key'], index, flush=True)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'measured.json')
    torch.manual_seed(42)
    model = load_model(MODEL)
    started = perf_counter()
    audits = []
    records = read_json(args.output / 'manifest.json')['records']
    for row in records:
        audits.extend(run_record(model, row, args.output / row['key']))
    errors = [audit['absolute_error'] for audit in audits]
    write_json(args.output / 'finite_complete.json', dict(status='complete', answers=len(records),
        probes=2 * len(records), forward_passes=len(records) * 19, seconds=perf_counter() - started,
        max_reconstruction=max(a['reconstruction'] for a in audits),
        max_earlier_effect=max(a['earlier_effect'] for a in audits),
        max_slope_error=max(errors), median_slope_error=float(np.median(errors)),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), labels_used=False,
        scope='whole-prefix QK/MLP/KV recomputed; generated tokens clamped; single source event'))


if __name__ == '__main__':
    main()
