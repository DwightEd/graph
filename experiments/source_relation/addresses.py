"""Expose source addresses for every official token and reviewed natural token."""
import argparse
import csv
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans


def top_addresses(values,positions,pieces):
    total = values.sum((0,1))
    total = total/max(total.sum(),1e-30)
    selected = np.argsort(total,kind='stable')[-5:][::-1]
    return [dict(prompt_position=int(positions[i]),text=pieces[positions[i]],mass=float(total[i]),
        context=''.join(pieces[max(0,positions[i]-4):positions[i]+5])) for i in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'scores_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'],local_files_only=True)
    with (args.output/'tokens.csv').open() as stream:
        official = list(csv.DictReader(stream))
    natural = read_local_spans(Path('outputs/role_free_flow_20260928'),Path('experiments/path_conflict/paired_cases.json'))
    all_rows = []
    for row in manifest['records']:
        key = row['key']
        if row['role']=='regression':
            selected = [dict(position=int(r['token']),label=int(r['label']),text=r['text']) for r in official if r['key']==key]
        elif row['role']=='natural':
            chosen = natural[natural.trace==row['trace']]
            if chosen.empty:
                continue
            ids = np.load(Path(manifest['samples'])/row['trace'])['token_ids']
            length = len(np.load(args.output/'sources'/str(row['source_id'])/'input.npz')['token_ids'])
            selected = [dict(position=int(r.position),label=int(r.local_label),text=tokenizer.decode([int(ids[length+r.position])]))
                        for r in chosen.itertuples()]
        else:
            continue
        inp = np.load(args.output/'sources'/str(row['source_id'])/'input.npz')
        ids = inp['token_ids']
        positions = np.flatnonzero(inp['source_mask'])
        pieces = tokenizer.batch_decode([[int(i)] for i in ids])
        attention = np.load(Path(manifest['base'])/key/'attention.npy',mmap_mode='r')
        gradient = np.load(Path(manifest['base'])/key/'derivative.npy',mmap_mode='r')
        readout = np.load(Path(manifest['base'])/key/'readouts.npz')
        for item in selected:
            token = item['position']
            read = np.asarray(attention[:,:,token,:len(ids)])[...,positions]
            effect = np.asarray(gradient[:,:,token,:len(ids)])[...,positions]
            all_rows.append(dict(key=key,role=row['role'],**item,
                query_input_position=len(ids)-1+token,
                alternative_token=tokenizer.decode([int(readout['alternative'][token])]),
                native_margin=float(readout['confidence'][token,2]),
                entropy_nats=float(readout['confidence'][token,0]),
                source_attention=float(read.sum(-1).mean()),
                reading=top_addresses(read,positions,pieces),
                positive_effect=top_addresses(np.maximum(effect,0),positions,pieces),
                negative_effect=top_addresses(-np.minimum(effect,0),positions,pieces)))
    write_json(args.output/'address_token_ledger.json',dict(rows=all_rows,
        scope='official token labels and reviewed natural positions only; all other natural positions unknown',
        interpretation='head-summed addresses for display only; native all-head/all-key arrays retained; no semantic evidence labels',
        labels_used_for_scoring=False))
    print('source addresses',len(all_rows),flush=True)


if __name__ == '__main__':
    main()
