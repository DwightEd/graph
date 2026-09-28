"""Posthoc exact head-identity randomization, conditional on per-layer counts."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from experiments.decision_risk_flow.data import write_json
from .score import FIELDS


def identity_overlap(first, second):
    probability = np.ones(1)
    expectation = 0.
    for left, right in zip(first, second):
        count_left, count_right = int(left.sum()), int(right.sum())
        outcomes = np.arange(min(count_left, count_right) + 1)
        probability = np.convolve(probability, hypergeom.pmf(outcomes, len(left), count_left, count_right))
        expectation += count_left * count_right / len(left)
    observed = int((first & second).sum())
    return dict(observed=observed, expected_under_within_layer_identity_shuffle=expectation,
                upper_tail_probability=float(probability[observed:].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output / 'diagnostics'
    cases = ['14315_headwear_scope', '14375_onion_stage']
    maps = [np.load(directory / (case + '.npz'))['head_auc'] for case in cases]
    results = {}
    for index, field in enumerate(FIELDS):
        for name, select in [('above_075', lambda x: x > .75), ('below_025', lambda x: x < .25)]:
            result = identity_overlap(select(maps[0][..., index]), select(maps[1][..., index]))
            result['bonferroni_12_upper_tail'] = min(1., 12 * result['upper_tail_probability'])
            results[field + '_' + name] = result
    write_json(directory / 'head_identity_null.json', dict(posthoc=True, results=results,
        hypothesis='Does physical head overlap exceed exchanging head identities independently within each layer?',
        caveat='Descriptive conditional randomization on two already exposed sources. Not token-independent statistics or proof of universal hallucination heads.'))


if __name__ == '__main__':
    main()
