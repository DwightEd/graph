"""Reproduce the four frozen local rounds, preserving completed stages."""

import argparse
from pathlib import Path
import subprocess
import sys


def stage(module, output, marker, extra=()):
    if (output/marker).exists():
        print('Preserving completed stage:', module, output, flush=True)
        return
    subprocess.run([sys.executable, '-m', 'experiments.'+module,
                    '--output', str(output), *extra], check=True)


def evaluate(output):
    for module, marker in [('message_js.evaluate', 'evaluation.json'),
                           ('message_js.score_verify', 'score_verification.json'),
                           ('head_state_readout.verify', 'verification.json'),
                           ('head_state_readout.audit', 'TOKEN_AUDIT.html')]:
        stage(module, output, marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='outputs/head_state_readout_new')
    args = parser.parse_args()
    outputs = [Path(args.output_prefix+'_v'+str(version)) for version in range(1, 5)]
    first, second, third, fourth = outputs
    stage('head_state_readout.features', first, 'features_complete.json')
    stage('head_state_readout.score', first, 'scores_frozen.json')
    evaluate(first)
    stage('head_state_readout.pair_diagnostic', first, 'pair_diagnostic.json')
    stage('head_state_readout.budget', first, 'budget_diagnostic.json')
    stage('head_state_readout.association', second, 'scores_frozen.json', ('--first', str(first)))
    evaluate(second)
    stage('head_state_readout.association', third, 'scores_frozen.json', ('--first', str(first), '--head-only'))
    evaluate(third)
    stage('head_state_readout.budget', third, 'budget_diagnostic.json')
    stage('head_state_readout.plot', third, 'natural_token_scores.json')
    stage('head_state_readout.addresses', third, 'address_audit.json')
    stage('head_state_readout.boundary', fourth, 'scores_frozen.json', ('--previous', str(third)))
    evaluate(fourth)


if __name__ == '__main__':
    main()
