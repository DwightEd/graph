"""Audit existing top-8 native routes; local reviewed claims are NOT corpus labels."""

import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from ..channels import ChannelGraph
from .onset_detection import save_detection, scan_channels
from .onset_windows import preceding_state


def archive_channels(graph, layers, heads):
    count_layers, count_heads, length, slots = graph['source'].shape
    query_rows = np.repeat(np.arange(length), slots)
    for layer in range(count_layers):
        if layers and layer not in layers:
            continue
        for head in range(count_heads):
            if heads and head not in heads:
                continue
            source = graph['source'][layer, head].reshape(-1)
            weight = graph['attention'][layer, head].reshape(-1).astype(float)
            if np.any((source > query_rows) & (weight > 0)) or np.any(weight < 0):
                raise ValueError('Archive contains noncausal or negative attention')
            matrix = csr_matrix((weight, (query_rows, source)), shape=(length, length))
            matrix.eliminate_zeros()
            matrix.sort_indices()
            yield ChannelGraph(layer, head, np.arange(length), matrix, int(graph['prompt_length']))


def archived_special_mask(context):
    """For THIS archive, enumerate control-token IDs from saved native decodings.

    Full-cache mode instead uses the original tokenizer's all_special_ids.
    Never assume a numerical vocabulary cutoff or silently call missing IDs normal.
    """
    ids = np.asarray(context['prefix_ids'])
    special_ids = sorted({int(token) for token, text in zip(ids, context['token_text'])
                          if re.fullmatch(r'<\|[^<>]+\|>', text)})
    if not special_ids:
        raise ValueError('Archive has no identifiable saved control-token decodings; supply original cache/tokenizer')
    return np.isin(ids, special_ids), special_ids


def save_claim(context, arrays, nodes, destination, args):
    target = len(context['prefix_ids'])-context['prompt_length']
    rows = []
    for threshold, states in zip(arrays['thresholds'], arrays['node_states']):
        prior = np.flatnonzero(states[max(0, target-args.window):target] == 1) + max(0, target-args.window)
        rows.append(dict(threshold=threshold, target=target, side=context['side'],
            unsupported=context['side'] == 'unsupported', prior_state=preceding_state(states, target, args.window),
            decision_state=int(states[target]), prior_node_targets=json.dumps(prior.tolist()),
            claim=context['reviewed_case'][context['side']]['target']))
    pd.DataFrame(rows).to_csv(destination / 'claim_association.csv', index=False)
    nodes['known_claim_target'] = target
    nodes['lag_to_known_claim'] = target-nodes.target
    nodes['claim_in_next_window'] = nodes.lag_to_known_claim.between(1, args.window)
    nodes['claim_at_decision'] = nodes.lag_to_known_claim.eq(0)
    nodes['known_claim_unsupported'] = context['side'] == 'unsupported'
    nodes['other_followup_annotation'] = 'unknown'
    nodes['future_hallucination_known'] = [True if upcoming and context['side'] == 'unsupported' else None
                                          for upcoming in nodes.claim_in_next_window]
    nodes.to_csv(destination / 'nodes.csv', index=False)


def annotate_source_roles(destination, context):
    sources = pd.read_csv(destination / 'source_reads.csv')
    sources['reviewed_roles'] = ['|'.join(role for role, positions in context['roles'].items()
                                        if source in positions) or 'outside_reviewed_claim_sources'
                                for source in sources.source]
    sources.to_csv(destination / 'source_reads.csv', index=False)


def run_archive(args):
    paths = sorted((args.review_archive / 'cases').glob('*/*/context.json'))
    if not paths:
        raise ValueError('No native route case contexts found')
    for path in tqdm(paths, desc='existing native prefixes'):
        context = json.loads(path.read_text())
        identity = context['case_id'] + '__' + context['side']
        destination = args.output / 'samples' / identity
        if args.resume and (destination / 'metadata.json').exists():
            continue
        with np.load(path.parent / 'routes.npz', allow_pickle=False) as saved:
            graph = {key: saved[key] for key in ('source', 'attention', 'prefix_ids', 'prompt_length')}
        if graph['prefix_ids'].tolist() != context['prefix_ids']:
            raise ValueError('Native routes/context token IDs disagree')
        special, special_ids = archived_special_mask(context)
        channels = archive_channels(graph, args.layers, args.heads)
        arrays, events, sources = scan_channels(channels, graph['prefix_ids'], context['prompt_length'], special, args)
        nodes = save_detection(destination, arrays, events, sources, graph['prefix_ids'],
                               context['token_text'], context['prompt_length'])
        annotate_source_roles(destination, context)
        save_claim(context, arrays, nodes, destination, args)
        metadata = dict(id=identity, source_id=context['source_id'], task='QA', generator='native_llama31_8b',
            split='resampled_review', annotation_origin='reviewed_local_claim_only_prefix_history_unreviewed',
            special_ids=special_ids, special_origin='explicit_control_spellings_in_saved_native_token_text',
            response_tokens=len(graph['prefix_ids'])-context['prompt_length'])
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
