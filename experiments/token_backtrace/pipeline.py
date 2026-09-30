"""CPU stages for frozen global-graph measurements; no automatic truth oracle."""

import argparse
import json
from pathlib import Path

import numpy as np

from .global_graph import score_events


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_new(path, value):
    encoded = json.dumps(value, default=json_value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')


def score_document(document):
    """A measured-event contract, distinct from an end-to-end text detector."""
    if document['schema'] != 'measured_token_graph_v1':
        raise ValueError('expected measured_token_graph_v1')
    if document['labels_used'] is not False:
        raise ValueError('the scoring contract excludes natural truth-label inputs')
    if not isinstance(document['proposals_complete'], bool):
        raise ValueError('proposals_complete must be a boolean')
    tokens, offsets = document['token_ids'], document['character_offsets']
    if not tokens or len(offsets) != len(tokens):
        raise ValueError('every original token requires its character offset')
    identities = [event['event_id'] for event in document['events']]
    if len(identities) != len(set(identities)):
        raise ValueError('one event per frozen proposal/comparison pair is required')
    thresholds = document['thresholds']
    if not thresholds['reference_id']:
        raise ValueError('independent channel threshold reference must be identified')
    scored = score_events(document['events'], len(tokens), thresholds['direct'], thresholds['graph'])
    complete = scored['complete'] and document['proposals_complete']
    return dict(schema='scored_token_graph_v1', key=document['key'], source_id=document['source_id'],
        token_ids=tokens, character_offsets=offsets, thresholds=thresholds,
        scores=scored, status='complete' if complete else 'incomplete',
        labels_used=False, semantic_proposals_generated_here=False,
        interpretation='reference-scaled compatibility, not factuality probability')


def lineage_document(path, prompt_length):
    from .readout import attribution_lineage
    with np.load(path, allow_pickle=False) as trace:
        result = attribution_lineage(trace['root_effect'], prompt_length)
        tokens = trace['token_ids']
    return dict(token_ids=tokens, **result,
        interpretation='normalized total-root-response lineage, not native Jacobian edges')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    score = stages.add_parser('score', help='score frozen, calibrated measured events')
    score.add_argument('--input', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    lineage = stages.add_parser('lineage', help='triangular lineage from existing full-answer root gradients')
    lineage.add_argument('--trace', type=Path, required=True)
    lineage.add_argument('--prompt-length', type=int, required=True)
    lineage.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'score':
        result = score_document(json.loads(args.input.read_text()))
    else:
        result = lineage_document(args.trace, args.prompt_length)
    write_new(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
