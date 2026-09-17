"""Compare completed fixed_graph novelty scores with saved supervised scores.

Same TARGET tokens, but different observers/inputs: fixed_graph uses pre-token
queries, CHARM uses post-token queries. This is not an isolated loss ablation.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import read_json, write_json
from .score_groups import score_tails


METHODS = ('nodes', 'graph', 'shuffled', 'one_hop', 'no_future', 'node_smooth')


def read_fixed_scores(root, prepared, table):
    root = Path(root)
    frozen = read_json(root/'predictions/freeze.json')
    if not frozen['complete']:
        raise ValueError('Use a completed fixed_graph prediction set')
    expected = {str(k): g for k, g in table.groupby('id')}
    blocks, ignored = [], 0
    for record in frozen['records']:
        identity = str(record['id'])
        if identity not in expected:
            ignored += 1
            continue
        group = expected[identity]
        with np.load(root/'predictions'/record['file'], allow_pickle=False) as saved:
            actual = json.loads(str(saved['record_json']))
            ids = saved['token_ids']
            prompt = int(saved['prompt_length'])
            coverage = saved['coverage'].astype(bool)
            scores = {name: saved[name] for name in METHODS}
        with np.load(Path(prepared)/'graphs/test'/(identity+'.npz'), allow_pickle=False) as graph:
            if not np.array_equal(ids, graph['token_ids']) or prompt != int(graph['prompt_length']):
                raise ValueError('fixed_graph and CHARM do not share exactly the same token IDs')
            if not np.array_equal(group.gold, graph['gold']):
                raise ValueError('Score table and shared token labels differ')
            response, offsets = str(graph['response']), graph['offsets']
        if group.text.tolist() != [response[a:b] for a, b in offsets]:
            raise ValueError('Prepared text differs from score table')
        if str(actual['id']) != identity or set(group.source_id) != {str(actual['source_id'])}:
            raise ValueError('fixed_graph answer/source identity mismatch')
        if len(coverage) != len(group) or any(len(s) != len(group) for s in scores.values()):
            raise ValueError('fixed_graph scores use a different target-token axis')
        blocks.append(pd.DataFrame(dict(id=identity, token=np.arange(len(group)), fixed_coverage=coverage, **scores)))
    if not blocks:
        raise ValueError('No shared fixed_graph answers; do not compare unmatched samples')
    scores = pd.concat(blocks, ignore_index=True)
    merged = table.merge(scores, on=['id', 'token'], how='left', validate='one_to_one')
    return merged, ignored


def rank_metrics(labels, scores):
    labels = np.asarray(labels, bool)
    return dict(auroc=float(roc_auc_score(labels, scores)) if labels.any() and (~labels).any() else None,
                ap=float(average_precision_score(labels, scores)) if labels.any() else None)


def compare_fixed_scores(root, prepared, table, output, fraction):
    joined, ignored = read_fixed_scores(root, prepared, table)
    common = joined.fixed_coverage.eq(True) & np.isfinite(joined[list(METHODS)]).all(axis=1)
    subset = joined.loc[common].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError('No common observable target tokens')
    baseline = score_tails(subset, fraction)
    rows, cells = [], []
    rows.append(dict(method='supervised_on_common', **rank_metrics(subset.gold, subset.score)))
    for method in METHODS:
        other = score_tails(subset.assign(score=subset[method]), fraction)
        other['supervised_tail'] = baseline.score_tail
        for (gold, highlow, predicted), group in other.groupby(['gold', 'supervised_tail', 'score_tail']):
            cells.append(dict(method=method, gold=int(gold), supervised_tail=highlow,
                fixed_tail=predicted, tokens=len(group), answers=group.id.nunique()))
        correlations = []
        for identity, group in subset.groupby('id'):
            if group.score.nunique() > 1 and group[method].nunique() > 1:
                correlations.append(float(spearmanr(group.score, group[method]).statistic))
        rows.append(dict(method=method, **rank_metrics(subset.gold, subset[method]),
            within_answer_score_spearman=float(np.mean(correlations)) if correlations else None))
        joined['tail_'+method] = pd.Series(other.score_tail.to_numpy(), index=joined.index[common])
    output = Path(output)
    pd.DataFrame(rows).to_csv(output/'fixed_score_comparison.csv', index=False)
    pd.DataFrame(cells).to_csv(output/'fixed_score_tail_crosses.csv', index=False)
    joined.to_csv(output/'fixed_score_tokens.csv.gz', index=False)
    write_json(output/'fixed_comparison_protocol.json', dict(source=str(root),
        all_tokens=len(table), common_tokens=len(subset), common_errors=int(subset.gold.sum()),
        common_answers=subset.id.nunique(), ignored_fixed_answers=ignored,
        coverage='Intersection of all six fixed scores; no missing-score fill, label shift or sign flip.',
        interpretation='Score alignment diagnostic only: observer position, attributes, compression, splits and scoring differ.',
        target_alignment='Exact full token IDs and original text checked. CHARM post-token versus fixed pre-token is NOT removed.'))
