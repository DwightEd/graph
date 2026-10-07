"""Post-score error diagnosis and exact-token comparison with frozen baselines."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .grounded_projection_data import write_json
from .grounded_projection_evaluate import calibrated_scores


def lexical_diagnosis(cases, scores, annotations):
    grouped = {0: {'exact': [], 'node': [], 'effect': []}, 1: {'exact': [], 'node': [], 'effect': []}}
    examples = {}
    for case in cases:
        if case['cohort'] == 'reference':
            continue
        annotation = annotations[case['id']]
        truth, valid = np.asarray(annotation['labels']), np.asarray(annotation['valid_tokens'], bool)
        source_ids = {i for i, selected in zip(case['source']['prompt_with_source'], case['source']['source_mask'])
                      if selected}
        exact = np.array([i in source_ids for i in case['response']['answer_ids']])
        for label, group in grouped.items():
            mask = valid & (truth == label)
            group['exact'].extend(exact[mask])
            group['node'].extend(scores[case['id']]['node'][mask])
            group['effect'].extend(scores[case['id']]['graph'][mask])
        examples[case['id']] = dict(wrong=int((truth & valid).sum()),
            wrong_word_also_in_source=int((truth & valid & exact).sum()))
    summary = {str(label): dict(tokens=len(group['exact']), exact_rate=float(np.mean(group['exact'])),
        node_mean=float(np.mean(group['node'])), response_mean=float(np.mean(group['effect'])),
        positive_response_rate=float(np.mean(np.asarray(group['effect']) > 0))) for label, group in grouped.items()}
    return dict(groups=summary, cases=examples, scope='posthoc mechanism diagnosis, not a selected detector')


def frozen_baselines(cases, scores, annotations, baseline):
    truth, values, identities = [], {n: [] for n in ('historical_fixed', 'fixed_reference', 'node', 'joint_graph')}, []
    for case in cases:
        path = baseline / case['task'] / (case['id'] + '_predictions.npz')
        if case['cohort'] == 'reference' or not path.exists():
            continue
        previous, current = np.load(path), scores[case['id']]
        if not np.array_equal(previous['token_id'], current['token_id']):
            raise ValueError(f'{case["id"]}: frozen baseline token mismatch')
        valid = np.asarray(annotations[case['id']]['valid_tokens'], bool)
        truth.append(np.asarray(annotations[case['id']]['labels'])[valid])
        identities.append(case['id'])
        for name in values:
            values[name].append((previous if name in previous.files else current)[name][valid])
    joined = np.concatenate(truth)
    return dict(ids=identities, tokens=len(joined), wrong=int(joined.sum()),
        metrics={name: dict(auroc=float(roc_auc_score(joined, np.concatenate(rows))),
                           ap=float(average_precision_score(joined, np.concatenate(rows))))
                 for name, rows in values.items()},
        limitation='same answer tokens, older measurement template/reference; no baseline re-fit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/source_sequence_20261006_v2_raw'))
    args = parser.parse_args()
    cases = json.loads((args.output / 'inputs.json').read_text())['cases']
    annotations = json.loads((args.output / 'evaluation_annotations.json').read_text())
    scores = {c['id']: dict(np.load(args.output / c['id'] / 'scores.npz')) for c in cases}
    calibrated_scores(cases, scores)
    write_json(args.output / 'lexical_diagnosis.json', lexical_diagnosis(cases, scores, annotations))
    write_json(args.output / 'frozen_baseline_comparison.json', frozen_baselines(cases, scores, annotations, args.baseline))


if __name__ == '__main__':
    main()
