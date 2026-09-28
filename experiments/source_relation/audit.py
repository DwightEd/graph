"""Post-freeze, exhaustive token ledger of reading, effect and persistence."""
import argparse
import csv
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json


def finite_mean(value, axis):
    valid = np.isfinite(value)
    return np.divide(np.where(valid,value,0).sum(axis),valid.sum(axis),
                     out=np.zeros_like(value.sum(axis)),where=valid.sum(axis)>0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'evaluation.json')
    manifest = read_json(args.output/'manifest.json')
    with (args.output/'token_audit.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ledger = []
    for record in manifest['records']:
        if record['role'] != 'regression':
            continue
        key = record['key']
        original = Path(manifest['base'])/key
        readout = np.load(original/'readouts.npz')
        confidence = readout['confidence']
        length = int(readout['prompt_length'])
        source = np.load(Path(manifest['route_base'])/key/'measurements.npz')['source_mask'].astype(bool)
        attn = np.load(original/'attention.npy',mmap_mode='r')
        derivative = np.load(original/'derivative.npy',mmap_mode='r')
        source_attention = np.asarray(attn[...,:length])[...,source].sum(-1).mean((0,1))
        history_attention = attn[...,length:].sum(-1).mean((0,1))
        local = np.zeros(len(confidence))
        positive,negative = [],[]
        for token in range(len(confidence)):
            local[token] = attn[:,:,token,length+max(0,token-16):length+token].sum(-1).mean()
            effect = np.asarray(derivative[:,:,token,:length])[...,source]
            positive.append(np.maximum(effect,0).sum(-1).mean())
            negative.append(-np.minimum(effect,0).sum(-1).mean())
        feature = np.load(args.output/key/'relations.npz')['values']
        means = finite_mean(feature,(0,1))
        fields = np.load(args.output/key/'relations.npz')['fields'].tolist()
        selected = [row for row in rows if row['key']==key]
        for row in selected:
            token = int(row['position'])
            ledger.append(dict(row, source_attention=float(source_attention[token]),
                history_attention=float(history_attention[token]), local16_attention=float(local[token]),
                local_given_history=float(local[token]/max(history_attention[token],1e-20)),
                source_positive_margin_effect=positive[token], source_negative_margin_effect=negative[token],
                native_margin=float(confidence[token,2]),
                **{name+'_head_mean':float(means[token,i]) for i,name in enumerate(fields)},
                relation_defined=bool(np.isfinite(feature[:,:,token,1]).any())))
    with (args.output/'mechanism_token_ledger.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=ledger[0])
        writer.writeheader()
        writer.writerows(ledger)
    methods = read_json(args.output/'scores_frozen.json')['methods']
    summary = []
    distance = 'mmd2' if manifest.get('measurement')=='native_message_kernel_mmd2' else 'js'
    fields = ['source_attention','history_attention','local_given_history','source_positive_margin_effect',
              'source_negative_margin_effect','native_margin',f'read_relation_{distance}_head_mean',f'use_relation_{distance}_head_mean']
    for method in methods:
        for status in ('TP','FP','FN','TN'):
            chosen = [row for row in ledger if row['method']==method and row['status']==status]
            summary.append(dict(method=method,status=status,count=len(chosen),
                means={name:float(np.mean([row[name] for row in chosen])) if chosen else None for name in fields}))
    write_json(args.output/'mechanism_summary.json',dict(rows=summary,
        interpretation='effects are native margin gate derivatives, not truth support; means for audit only, scoring preserves heads',
        scope='every eligible official regression token; natural labels evaluated separately'))
    print('mechanism ledger',len(ledger),flush=True)


if __name__ == '__main__':
    main()
