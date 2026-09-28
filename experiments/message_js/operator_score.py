"""Conditional unlabeled mechanism tests, calibrated on a separate mixture."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .score import fit_cdf, percentile, valid_tokens

BASELINES = ('propagated', 'relative_propagated', 'attention_js', 'raw_route',
             'raw_route_offline_mean', 'entropy', 'surprisal')
METHODS = ('mechanism', 'mechanism_unconditional', 'operator_route', 'prompt_cancel',
           'full_fisher_history', 'full_output_js_coefficient') + BASELINES


def measurements(output, base, row):
    data = np.load(base / row['key'] / 'readouts.npz')
    prompt = int(data['prompt_length'])
    count = len(data['token_ids'])
    attention = np.load(base / row['key'] / 'attention.npy', mmap_mode='r')
    reading = attention[..., :prompt].sum(-1).reshape(1024, count)
    relative_js = np.load(base / row['key'] / 'branch.npz')['relative_js']
    operator = np.load(output / row['key'] / 'operator.npz')
    raw = data['measured'].reshape(1024, count, -1)
    values = np.stack((reading, raw[:, :, 3], relative_js,
                      operator['prompt_cancel'].reshape(1024, count), raw[:, :, 2]), -1)
    gram = operator['gram']
    history_share = gram[:, 1, 1] / np.maximum(gram[:, 0, 0] + gram[:, 1, 1], 1e-30)
    # JS(p(g+eps*e_P),p(g+eps*e_H))/eps^2 -> this exact local coefficient, in nats.
    output_js = np.maximum(gram[:, 0, 0] + gram[:, 1, 1] - 2 * gram[:, 0, 1], 0) / 8
    history = np.stack((history_share, output_js), -1)
    context = np.stack((data['confidence'][:, 0], np.arange(count) / max(count - 1, 1)), -1)
    return values, history, context


def context_codes(context, entropy_cuts):
    return np.searchsorted(entropy_cuts, context[:, 0]) * 3 + np.minimum((context[:, 1] * 3).astype(int), 2)


def fit_conditional(values, weights, codes):
    fitted = dict(global_cdf=fit_cdf(values, weights), local={})
    for code in range(9):
        selected = (codes == code) & np.isfinite(values)
        count = int(selected.sum())
        fitted['local'][code] = (fit_cdf(values[selected], weights[selected]), count / (count + 32.))
    return fitted


def conditional_percentile(values, codes, fitted):
    global_rank = percentile(values, fitted['global_cdf'])
    result = global_rank.copy()
    for code in np.unique(codes):
        selected = codes == code
        local, weight = fitted['local'][code]
        if weight > 0:
            result[selected] = weight * percentile(values[selected], local) + (1 - weight) * global_rank[selected]
    return result, global_rank


def pattern_score(ranks, history):
    read, deficit, mismatch, cancel, read_use = [ranks[:, :, field] for field in range(5)]
    absent = np.minimum(1 - read, deficit)
    integration = np.minimum(np.minimum(read, deficit), np.maximum(cancel, read_use))
    conflict = np.minimum(mismatch, np.minimum(history[:, 0], history[:, 1])[None])
    # A missing relay excludes the conflict test; it is not evidence of consistency.
    return np.nanmax(np.stack((absent, integration, conflict)), axis=0)


def score_task(output, base, records, task, old):
    rows = [row for row in records if row['task'] == task]
    observed = {row['key']: measurements(output, base, row) for row in rows}
    reference, weights, histories, contexts = [], [], [], []
    for row in rows:
        if row['role'] != 'fit':
            continue
        valid = valid_tokens(row, output, old)
        value, history, context = observed[row['key']]
        reference.append(value[:, valid])
        histories.append(history[valid])
        contexts.append(context[valid])
        weights.append(np.full(valid.sum(), 1 / valid.sum()))
    reference = np.concatenate(reference, 1)
    weights, histories, contexts = np.concatenate(weights), np.concatenate(histories), np.concatenate(contexts)
    ordered, cumulative = fit_cdf(contexts[:, 0], weights)
    cuts = ordered[np.searchsorted(cumulative[1:], [1/3, 2/3])]
    codes = context_codes(contexts, cuts)
    fitted = [[fit_conditional(reference[head, :, field], weights, codes) for field in range(5)]
              for head in range(1024)]
    fitted_history = [fit_conditional(histories[:, field], weights, codes) for field in range(2)]
    for row in rows:
        value, history, context = observed[row['key']]
        codes = context_codes(context, cuts)
        ranks, unconditional = np.empty_like(value), np.empty_like(value)
        for head in range(1024):
            for field in range(5):
                ranks[head, :, field], unconditional[head, :, field] = conditional_percentile(
                    value[head, :, field], codes, fitted[head][field])
        history_rank, history_global = np.empty_like(history), np.empty_like(history)
        for field in range(2):
            history_rank[:, field], history_global[:, field] = conditional_percentile(history[:, field], codes, fitted_history[field])
        patterns = pattern_score(ranks, history_rank)
        mechanism = np.quantile(patterns, .9, axis=0)
        global_score = np.quantile(pattern_score(unconditional, history_global), .9, axis=0)
        previous = np.load(base / row['key'] / 'scores.npz')
        route = previous['route_combined'] * 2 - previous['propagated']
        # Natural trajectories have no old-route cache; the extra fusion is missing there.
        score = dict(mechanism=mechanism, mechanism_unconditional=global_score,
            operator_route=np.maximum(mechanism, route),
            prompt_cancel=np.quantile(value[:, :, 3], .9, axis=0), full_fisher_history=history[:, 0],
            full_output_js_coefficient=history[:, 1])
        for name in BASELINES:
            score[name] = previous[name]
        np.savez_compressed(output / row['key'] / 'scores.npz', **score,
            head_mechanism=patterns, head_conditional_percentiles=ranks)
        print('operator scored', row['key'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    args = parser.parse_args()
    read_json(args.output / 'capture_complete.json')
    manifest = read_json(args.output / 'manifest.json')
    records, base = manifest['records'], Path(manifest['base'])
    for task in ('QA', 'Summary', 'Data2txt'):
        score_task(args.output, base, records, task, args.old)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        values = {name: [] for name in METHODS}
        weights = []
        for row in records:
            if row['task'] != task or row['role'] != 'dev':
                continue
            valid = valid_tokens(row, args.output, args.old)
            weights.append(np.full(valid.sum(), 1 / valid.sum()))
            saved = np.load(args.output / row['key'] / 'scores.npz')
            for name in METHODS:
                values[name].append(saved[name][valid])
        thresholds[task] = {}
        for name in METHODS:
            ordered, cumulative = fit_cdf(np.concatenate(values[name]), np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'scores_frozen.json', dict(status='all scores and thresholds frozen',
        methods=METHODS, main='mechanism', labels_used=False, reference_sources_per_task=4,
        threshold='source-equal unlabeled dev mixture .95 quantile; not normal FPR guarantee',
        context='fit entropy tertiles x relative position thirds; empirical CDF shrunk to global by n/(n+32)',
        patterns='max of absent=min(low_read,deficit), integration=min(read,deficit,max(cancel,read_use)), conflict=min(relative_JS,history_Fisher,output_JS_coefficient)',
        propagation='native current-query depth transport; no recursive risk labels'))


if __name__ == '__main__':
    main()
