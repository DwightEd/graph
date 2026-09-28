"""Post-freeze response magnitudes, finite effects and exact FP/FN locations."""
import argparse
import csv
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json


def finite_summary(output):
    rows = read_json(output/'finite_effects.json')['rows']
    result = []
    for dose in (0., .01, .05, .25):
        selected = [r for r in rows if r['dose']==dose]
        predicted = np.array([r['predicted'] for r in selected])
        measured = np.array([r['measured'] for r in selected])
        error = np.abs(predicted-measured)
        result.append(dict(dose=dose, interventions=len(selected), max_absolute_error=float(error.max()),
            mean_absolute_error=float(error.mean()), sign_agreement=float((np.sign(predicted)==np.sign(measured)).mean()),
            median_relative_error=float(np.median(error/np.maximum(np.abs(predicted), 1e-8)))))
    write_json(output/'finite_summary.json', result)
    return result


def locations(output):
    with (output/'token_audit.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r['method']=='sparse_joint_fused']
    manifest = read_json(output/'manifest.json')
    data = {r['key']: np.load(output/r['key']/'responses.npz')['values']
            for r in manifest['records'] if r['role']=='regression'}
    enriched = []
    for row in selected:
        values = data[row['key']][:, :, int(row['position'])]
        item = dict(row)
        for index, name in ((0, 'backward'), (1, 'forward')):
            item[name+'_response_min'] = float(values[..., index].min())
            item[name+'_response_max'] = float(values[..., index].max())
            item[name+'_negative_heads'] = int((values[..., index]<0).sum())
        item['source_attention_mean'] = float(values[..., 10].mean())
        enriched.append(item)
    with (output/'mechanism_token_audit.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=enriched[0])
        writer.writeheader()
        writer.writerows(enriched)
    summaries = []
    for key in dict.fromkeys(r['key'] for r in selected):
        for status in ('TP', 'FP', 'FN', 'TN'):
            tokens = [r for r in enriched if r['key']==key and r['status']==status]
            summaries.append(dict(key=key, status=status, tokens=len(tokens),
                positions=[int(r['position']) for r in tokens],
                source_attention_mean=float(np.mean([r['source_attention_mean'] for r in tokens])) if tokens else None))
    write_json(output/'mechanism_summary.json', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    args = parser.parse_args()
    finite_summary(args.first)
    locations(args.second)
    for path in (args.first, args.second):
        manifest = read_json(path/'manifest.json')
        fields = {}
        for row in manifest['records']:
            values = np.load(path/row['key']/'responses.npz')['values']
            assert values.shape[:2]==(32, 32) and values.shape[-1]==12
            assert np.isfinite(values).all(), row['key']
            fields.setdefault(row['role'], set()).add(row['source_id'])
        for first, second in (('fit', 'dev'), ('fit', 'regression'), ('dev', 'regression'), ('fit', 'natural'), ('dev', 'natural')):
            assert not fields[first] & fields[second], (first, second)
        write_json(path/'feature_verification.json', dict(status='passed', answers=len(manifest['records']),
            axes='32 layers x 32 physical heads x all answer tokens x 12 fields',
            source_partition_disjoint=True, finite_features=True, same_agent=True))
    print('finite effect summaries, all token locations, feature axes and source partitions checked')


if __name__ == '__main__':
    main()
