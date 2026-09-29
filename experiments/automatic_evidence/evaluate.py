"""Evaluation-only labels; exact token alignment and separate attribution fidelity."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.decision_risk_flow.data import labels, read_json, write_json
from .score import METHODS


def metrics(gold, score, alarm):
    result = dict(tokens=len(gold), positive=int(gold.sum()),
        tp=int(np.sum(alarm & (gold == 1))), fp=int(np.sum(alarm & (gold == 0))),
        fn=int(np.sum(~alarm & (gold == 1))))
    if len(np.unique(gold)) == 2:
        result.update(auroc=float(roc_auc_score(gold, score)),
                      ap=float(average_precision_score(gold, score)))
    return result


def historical():
    result = {}
    for name, directory in [('historical_js', 'source_relation_20260928_v1'),
                            ('historical_mmd', 'source_relation_20260928_v3')]:
        with (Path('outputs') / directory / 'token_audit.csv').open() as stream:
            for item in csv.DictReader(stream):
                if item['method'] != 'relation_conditional_fused':
                    continue
                index = (item['key'], int(item['position']))
                result.setdefault(index, {})[name] = (float(item['score']), item['status'] in ('TP', 'FP'))
    with Path('outputs/token_backtrace_readout_20260930_v2/pilot_tokens.csv').open() as stream:
        for item in csv.DictReader(stream):
            index = (item['key'], int(item['target']))
            for name in ('base', 'odds_full', 'logic_full'):
                result.setdefault(index, {})[name] = (float(item[name]), item[name+'_alarm'] == 'True')
    return result


def token_rows(output, records):
    truth = labels([row['original'] for row in records])
    thresholds = read_json(output / 'thresholds.json')
    old = historical()
    tokens = []
    for row in records:
        groups = read_json(output / row['key'] / 'sources.json')
        with np.load(output / row['key'] / 'scores.npz') as saved:
            np.testing.assert_array_equal(saved['token_ids'], row['response']['answer_ids'])
            for target, (token_id, (start, stop)) in enumerate(zip(saved['token_ids'], row['response']['offsets'])):
                item = dict(key=row['key'], task=row['task'], target=target,
                    text=row['response']['token_text'][target], start=start, stop=stop,
                    gold=int(truth[row['key']][target]), valid=bool(stop > start and token_id not in row['response']['special_ids']))
                for name in METHODS:
                    item[name] = float(saved[name][target])
                    item[name+'_alarm'] = bool(saved[name][target] > thresholds[row['key']][name])
                for name, (value, alarm) in old[(row['key'], target)].items():
                    item[name], item[name+'_alarm'] = value, alarm
                item['source_index'] = int(saved['selected'][target])
                item['source_text'] = groups[item['source_index']]['text']
                item['verified_source_index'] = int(saved['confirmed'][target])
                item['verified_source_text'] = groups[item['verified_source_index']]['text']
                tokens.append(item)
    return tokens


def summarize(rows, methods):
    valid = [item for item in rows if item['valid']]
    gold = np.array([item['gold'] for item in valid])
    return {name: metrics(gold, np.array([item[name] for item in valid]),
                          np.array([item[name+'_alarm'] for item in valid])) for name in methods}


def equal_budget(rows, methods):
    """Each answer's top 5% uses score ranks only; labels remain evaluation-only."""
    selected = {name: [] for name in methods}
    gold = []
    for key in dict.fromkeys(item['key'] for item in rows):
        subset = [item for item in rows if item['key'] == key and item['valid']]
        gold.extend(item['gold'] for item in subset)
        for name in methods:
            value = np.array([item[name] for item in subset])
            alarm = np.zeros(len(value), dtype=bool)
            alarm[np.argsort(-value, kind='stable')[:int(np.ceil(.05*len(value)))]] = True
            selected[name].extend(alarm)
    gold = np.asarray(gold)
    return {name: dict(tp=int(np.sum(np.asarray(alarm) & (gold == 1))),
                       fp=int(np.sum(np.asarray(alarm) & (gold == 0)))) for name, alarm in selected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'scores_frozen.json')
    records = read_json(args.output / 'manifest.json')['records']
    rows = token_rows(args.output, records)
    methods = (*METHODS, 'historical_js', 'historical_mmd', 'base', 'odds_full', 'logic_full')
    with (args.output / 'tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(scope='eight exposed development answers; all 1487 tokens; no blind-test claim',
        pooled=summarize(rows, methods), equal_alarm_budget=equal_budget(rows, methods),
        answers={row['key']: summarize([item for item in rows if item['key'] == row['key']], methods)
                 for row in records})
    write_json(args.output / 'evaluation.json', result)
    print(result['pooled'], flush=True)


if __name__ == '__main__':
    main()
