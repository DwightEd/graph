"""Explicit, label-free measurement/scoring stages and separate evaluation."""

import argparse
from pathlib import Path
import runpy
import sys


# Reuse the existing native adapter package when running from a source checkout.
sys.path.insert(0, str(Path(__file__).parent / 'teaching/state_audit/src'))

COMMANDS = {
    'graph': {'lineage': 'token_backtrace.pipeline', 'score': 'token_backtrace.pipeline'},
    'token': {'trace': 'token_backtrace.trace', 'validate': 'token_backtrace.validate',
              'baseline': 'token_backtrace.benchmark', 'evaluate': 'token_backtrace.diagnose',
              'relations': 'token_backtrace.logic_benchmark'},
    'evidence': {'prepare': 'automatic_evidence.prepare', 'capture': 'automatic_evidence.capture',
                 'score': 'automatic_evidence.score', 'evaluate': 'automatic_evidence.evaluate'},
    'baseline': {'fixed': 'unsupervised_graph.fixed', 'source-capture': 'source_relation.capture',
                 'js': 'source_relation.measure', 'mmd': 'source_relation.kernel',
                 'score': 'source_relation.score', 'evaluate': 'message_js.evaluate'},
    'history': {'capture': 'choice_feedback.run', 'verify': 'choice_feedback.verify',
                'evaluate': 'choice_feedback.analyze'},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Use GROUP STAGE --help for inputs. Graph scoring consumes frozen measured '
        'events; semantic proposal generation and real-model validation remain research work.')
    groups = parser.add_subparsers(dest='group', required=True)
    for group, stages in COMMANDS.items():
        command = groups.add_parser(group, help=', '.join(stages))
        command.add_argument('stage', choices=stages)
        command.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments
    if args.group == 'graph':
        arguments = [args.stage, *arguments]
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        runpy.run_module('experiments.' + COMMANDS[args.group][args.stage], run_name='__main__')
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
