"""Test local and gold-onset reading against distance and lexical controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import torch
from transformers import AutoTokenizer

from experiments.native_support.ragtruth_benchmark.data import encode_source
from .dominance_data import ATTENTION, transfer_source_mask
from .grounded_projection_data import DATASET, MODEL, write_json
from .repetition_analysis import source_bootstrap
from .repetition_teacher import BASE


def edge_features(attention, tokens, prompt, target, anchor, source_mask):
    """Predictor query P+t-1: anchor expectation preserves distance and BPE copy."""
    keys = np.arange(len(tokens))
    history = (keys >= prompt) & (keys < prompt + target)
    distance = prompt + target - keys
    source = np.zeros(len(tokens), dtype=bool)
    source[:prompt] = source_mask
    anchor_key = prompt + anchor
    anchor_distance = target - anchor
    groups = np.ceil(np.log2(np.maximum(distance, 1))).astype(int)
    group = int(np.ceil(np.log2(anchor_distance)))
    same_word = tokens == tokens[prompt + target]
    eligible = history & (groups == group) & (same_word == same_word[anchor_key])
    expected = attention[:, eligible].sum(axis=1) / eligible.sum()
    return dict(source=attention[:, source].sum(axis=1),
        local=attention[:, history & (distance <= 8)].sum(axis=1),
        remote=attention[:, history & (distance > 32)].sum(axis=1),
        onset=attention[:, anchor_key], onset_excess=attention[:, anchor_key]-expected,
        retained=attention.sum(axis=1), exchangeable=np.full(len(attention), eligible.sum() > 1))


def answer_pairs(row, pairs, saved, source_mask, important):
    tokens = saved['token_ids']
    prompt = int(saved['response_idx'])
    length = len(row['labels'])
    matrix = csr_matrix((saved['response_values'].astype(np.float32),
        saved['response_column_indices'], saved['response_row_ptr']), shape=(1024*length, len(tokens)))
    records = []
    for target, control in pairs:
        onset = target
        while onset and row['labels'][onset-1]:
            onset -= 1
        age = target-onset
        normal_anchor = control-age
        if not age or normal_anchor < 0 or row['labels'][normal_anchor:control+1].any():
            continue
        values = []
        for position, anchor in ((target, onset), (control, normal_anchor)):
            attention = matrix[np.arange(1024)*length+position-1].toarray()
            measured = edge_features(attention, tokens, prompt, position, anchor, source_mask)
            values.append(measured)
        record = dict(id=row['id'], source_id=row['source_id'], target=target, control=control,
                      onset=onset, normal_anchor=normal_anchor, age=age)
        for name in values[0]:
            for subset, heads in (('all', np.arange(1024)), ('important', important)):
                record[subset+'_'+name+'_error'] = float(values[0][name][heads].mean())
                record[subset+'_'+name+'_normal'] = float(values[1][name][heads].mean())
        records.append(record)
    return records


def matched_statistics(records, normalize=False):
    statistics = {}
    names = [name[:-6] for name in records[0] if name.endswith('_error')]
    for prefix in names:
        if normalize and prefix.split('_', 1)[1] not in ('local', 'remote', 'source', 'onset', 'onset_excess'):
            continue
        errors = np.array([r[prefix+'_error'] for r in records])
        normal = np.array([r[prefix+'_normal'] for r in records])
        if normalize:
            subset = prefix.split('_', 1)[0]
            errors /= [r[subset+'_retained_error'] for r in records]
            normal /= [r[subset+'_retained_normal'] for r in records]
        sources = [r['source_id'] for r in records]
        statistics[prefix] = dict(error=source_bootstrap(sources, errors),
            normal=source_bootstrap(sources, normal), difference=source_bootstrap(sources, errors-normal))
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(BASE / 'prepared.pt', weights_only=False)
    pairs = json.loads((args.output / 'matched_tokens.json').read_text())['strict']
    important = json.loads((args.output / 'head_selection.json').read_text())['important']
    index = {str(r['sample_id']): r for r in map(json.loads, (ATTENTION / 'test/index.jsonl').open())}
    sources = {str(r['source_id']): r for r in map(json.loads, (DATASET / 'source_info.jsonl').open())}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records = []
    for row in data['test']:
        if not pairs[row['id']]:
            continue
        with np.load(ATTENTION / 'test' / index[row['id']]['path']) as saved:
            prompt = int(saved['response_idx'])
            canonical = encode_source(sources[row['source_id']], tokenizer)
            mask = transfer_source_mask(canonical, saved['token_ids'][:prompt].tolist())
            records.extend(answer_pairs(row, pairs[row['id']], saved, mask, important))
    statistics = matched_statistics(records)
    write_json(args.output / 'edge_pairs.json', records)
    write_json(args.output / 'edge_results.json', dict(pairs=len(records), statistics=statistics,
        scope='retained CSR floor .01, original predictor rows, gold onset diagnosis'))
    normalized = matched_statistics(records, normalize=True)
    write_json(args.output / 'edge_normalized_results.json',
        {name: values['difference'] for name, values in normalized.items()})
    print(json.dumps(statistics, indent=2))


if __name__ == '__main__':
    main()
