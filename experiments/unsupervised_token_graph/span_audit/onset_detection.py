"""Scan full answers before labels; keep physical heads and source token identities."""

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .onset_changes import MEASURES, aggregate_heads, anchor_sources, classify, event_starts, measure_head


def scan_channels(channels, token_ids, prompt_length, special, args):
    head_ids, all_states, raw_states, events, sources = [], [], [], [], []
    thresholds = sorted(set([.05, .1, .2, args.minimum_shift]))
    for channel in tqdm(channels, desc='physical heads', leave=False):
        measures, rows = measure_head(channel, token_ids, prompt_length, special,
                                      args.baseline_steps, args.local_window)
        raw = np.stack([classify(measures, threshold) for threshold in thresholds])
        states = np.stack([event_starts(state) for state in raw])
        head_ids.append([channel.layer, channel.head])
        all_states.append(states)
        raw_states.append(raw)
        for target in np.flatnonzero(states[thresholds.index(args.minimum_shift)] == 1):
            query = prompt_length+int(target)-1
            identity = dict(target=int(target), query=query, layer=channel.layer, head=channel.head)
            events.append(dict(identity, **dict(zip(MEASURES, measures[target]))))
            anchors = anchor_sources(rows, query, prompt_length, args.local_window, args.baseline_steps)
            sources.extend(dict(identity, **source) for source in anchors)
    if not head_ids:
        raise ValueError('No requested physical attention heads found')
    head_states = np.stack(all_states, axis=1)
    node_states = np.stack([aggregate_heads(states) for states in head_states])
    arrays = dict(thresholds=np.array(thresholds), head_ids=np.array(head_ids),
                  head_states=head_states, node_states=node_states,
                  qualifying_states=np.stack(raw_states, axis=1),
                  target_positions=np.arange(node_states.shape[-1]), special_mask=special,
                  token_ids=token_ids, prompt_length=np.array(prompt_length))
    events = pd.DataFrame(events, columns=['target', 'query', 'layer', 'head', *MEASURES])
    sources = pd.DataFrame(sources, columns=['target', 'query', 'layer', 'head', 'source',
        'attention_now', 'attention_before', 'source_gain', 'source_gain_low'])
    return arrays, events, sources


def token_context(text, position, radius=5):
    return ''.join(text[max(0, position-radius):position]) + ' ⟦' + text[position] + '⟧ ' + ''.join(text[position+1:position+radius+1])


def describe_sources(sources, token_ids, text, prompt_length, event_queries):
    table = sources.copy()
    table['source_token_id'] = [int(token_ids[position]) for position in table.source]
    table['source_text'] = [text[position] for position in table.source]
    table['source_context'] = [token_context(text, position) for position in table.source]
    table['region'] = np.where(table.source < prompt_length, 'prompt', 'history')
    table['source_is_prior_event'] = table.source.isin(event_queries)
    table['same_token_as_node'] = [bool(token_ids[source] == token_ids[query])
                                   for source, query in zip(table.source, table['query'])]
    return table


def describe_nodes(events, token_ids, text, prompt_length):
    rows = []
    for target, group in events.groupby('target'):
        query = prompt_length+int(target)-1
        heads = [[int(row.layer), int(row.head)] for row in group.itertuples()]
        rows.append(dict(target=int(target), query=query, node_token=int(target)-1,
            node_token_id=int(token_ids[query]), node_text=text[query], context=token_context(text, query),
            predicted_token=text[query+1] if query+1 < len(text) else None,
            heads=len(group), head_ids=json.dumps(heads), max_shift_low=group.shift_low.max()))
    return pd.DataFrame(rows, columns=['target', 'query', 'node_token', 'node_token_id', 'node_text',
        'context', 'predicted_token', 'heads', 'head_ids', 'max_shift_low'])


def save_detection(destination, arrays, events, sources, token_ids, text, prompt_length):
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'detection.npz', **arrays)
    coverage = [dict(threshold=threshold, nodes=int((states == 1).sum()),
                     absent_positions=int((states == 0).sum()), unknown_positions=int((states == -1).sum()),
                     physical_heads=len(arrays['head_ids']))
                for threshold, states in zip(arrays['thresholds'], arrays['node_states'])]
    pd.DataFrame(coverage).to_csv(destination / 'coverage.csv', index=False)
    (destination / 'token_text.json').write_text(json.dumps(dict(tokens=text, prompt_length=prompt_length),
                                                          ensure_ascii=False) + '\n', encoding='utf-8')
    events.to_csv(destination / 'head_events.csv', index=False)
    sources = describe_sources(sources, token_ids, text, prompt_length, set(events['query']))
    sources.to_csv(destination / 'source_reads.csv', index=False)
    nodes = describe_nodes(events, token_ids, text, prompt_length)
    nodes.to_csv(destination / 'nodes.csv', index=False)
    return nodes
