"""One controlled LDA audit: covariance -> head identity -> history -> prompt reading."""

from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import read_json, write_json, save_scores, prepare_output
from .lda_data import read_audit_inputs
from .lda_math import (moments, coefficients, covariance_terms, common_and_contrast,
                       head_features, balance_context, past_mean, fit_linear_prediction)


def add_scores(name, model, inputs, scores, output):
    save_scores(output / 'parameters' / (name + '.npz'), **model)
    for split in scores:
        scores[split][name] = inputs[split] @ model['weight'] + model['intercept']
    print('LDA:', name, flush=True)


def fit_node_controls(data, scores, args, output):
    inputs = {split: value[0] for split, value in data.items()}
    labels = data['fit'][1].gold.to_numpy()
    heads = data['fit'][3][1]
    stats = moments(inputs['fit'], labels, args.lda_ridge)
    save_scores(output / 'parameters/full_moments.npz', **stats)
    for structure in ('raw_mean', 'diagonal', 'layer_block', 'full'):
        model = coefficients(stats, structure, heads)
        add_scores(structure, model, inputs, scores, output)
    for view in ('layer_mean', 'head_contrast', 'unordered_heads'):
        if view == 'unordered_heads':
            transformed = {s: np.sort(x.reshape(len(x), -1, heads), axis=2).reshape(x.shape)
                           for s, x in inputs.items()}
        else:
            transformed = {s: head_features(x, heads, view) for s, x in inputs.items()}
        transformed_stats = moments(transformed['fit'], labels, args.lda_ridge)
        add_scores(view, coefficients(transformed_stats), transformed, scores, output)
    return stats, coefficients(stats)


def fit_context_controls(data, scores, args, output):
    inputs = {split: value[0] for split, value in data.items()}
    labels = data['fit'][1].gold.to_numpy()
    weights, coverage = balance_context(data['fit'][1])
    pd.DataFrame(coverage).to_csv(output / 'fit_context_balance.csv', index=False)
    if not weights.any():
        print('No mixed FIT context strata: matched_support/balanced_context unavailable.', flush=True)
        return
    for name, weight in (('matched_support', (weights > 0).astype(float)), ('balanced_context', weights)):
        stats = moments(inputs['fit'], labels, args.lda_ridge, weight)
        add_scores(name, coefficients(stats), inputs, scores, output)


def decompose_scores(data, scores, stats, full, args, output):
    heads = data['fit'][3][1]
    terms = covariance_terms(stats, full['weight'], heads)
    common, contrast = common_and_contrast(full['weight'], heads)
    np.testing.assert_allclose(sum(terms.values()), full['weight'], rtol=1e-7, atol=1e-8)
    for split, table in scores.items():
        values = data[split][0]
        for name, weight in terms.items():
            table['part_' + name] = (values - stats['center']) @ weight
        table['part_layer_common'] = (values - stats['center']) @ common
        table['part_head_contrast'] = (values - stats['center']) @ contrast
        table['history_mean'] = past_mean(table.full, table.id.to_numpy(), args.lda_window)
        table['current_increment'] = table.full - table.history_mean
    save_scores(output / 'parameters/score_parts.npz', **terms, common=common, contrast=contrast)
    return terms


def fit_prompt_controls(data, scores, stats, full, args, output):
    """Use the original retained per-head prompt mass; keep self-node models untouched."""
    prompts = {split: value[2] for split, value in data.items()}
    labels = data['fit'][1].gold.to_numpy()
    prompt_stats = moments(prompts['fit'], labels, args.lda_ridge)
    add_scores('prompt_only', coefficients(prompt_stats), prompts, scores, output)
    prediction = fit_linear_prediction(prompts['fit'], data['fit'][0], args.lda_ridge)
    save_scores(output / 'parameters/prompt_to_self.npz', **prediction)
    residual, joint, surrogate_rows = {}, {}, []
    target_mean = float((data['fit'][0] @ full['weight'] + full['intercept']).mean())
    for split, (values, meta, prompt, _) in data.items():
        expected = prompt @ prediction['weight'] + prediction['intercept']
        residual[split] = values - expected
        joint[split] = np.concatenate((values, prompt), axis=1)
        predicted_score = expected @ full['weight'] + full['intercept']
        actual_score = values @ full['weight'] + full['intercept']
        if split in scores:
            scores[split]['prompt_predicted_full'] = predicted_score
            scores[split]['full_unexplained_by_prompt'] = actual_score - predicted_score
        squared_error = np.square(actual_score - predicted_score).sum()
        denominator = np.square(actual_score - target_mean).sum()
        surrogate_rows.append(dict(split=split, tokens=len(meta), score_r2=1 - squared_error / denominator))
    for name, values in (('self_after_prompt_regression', residual), ('self_plus_prompt', joint)):
        fitted = moments(values['fit'], labels, args.lda_ridge)
        add_scores(name, coefficients(fitted), values, scores, output)
    pd.DataFrame(surrogate_rows).to_csv(output / 'prompt_explains_score.csv', index=False)


def compare_old_lda(args, full, data, output):
    path = Path(args.root) / 'audit_mixture/supervised_lda_diagnostic.npz'
    result = dict(reference_exists=path.is_file())
    if path.is_file():
        with np.load(path, allow_pickle=False) as reference:
            differences = data['test'][0] @ (full['weight'] - reference['coefficient'])
            differences += full['intercept'] - reference['intercept']
        result['max_abs_test_logit_difference'] = float(abs(differences).max())
    write_json(output / 'baseline_replay.json', result)


def run_lda_audit(args):
    from .lda_report import report, save_channel_rules, report_saved_scores

    default = 'audit_lda_prompt' if args.lda_prompt else 'audit_lda_nodes'
    if args.lda_stage == 'scores':
        default = 'audit_lda_scores'
    output = Path(args.output) if args.output else Path(args.root) / default
    if args.lda_stage == 'scores':
        from .lda_data import read_saved_scores
        output.mkdir(parents=True, exist_ok=True)
        pair_path = Path(args.pairs) if args.pairs else Path(args.root) / 'charm_in/test/cluster_audit/pairs.json'
        pairs = read_json(pair_path) if pair_path.exists() else []
        pairs = [pair for pair in pairs if pair['tier'] == 'cluster']
        report_saved_scores(read_saved_scores(args.root, args.lda_window), pairs, output, args.bootstrap)
        return
    if args.lda_stage == 'report':
        report(output, args.bootstrap)
        return
    config = dict(prepared=str(Path(args.prepared).resolve()), root=str(Path(args.root).resolve()),
                  ridge=args.lda_ridge, window=args.lda_window, prompt=args.lda_prompt, pairs=args.pairs)
    prepare_output(output, config)
    with threadpool_limits(limits=4):
        data, parts = read_audit_inputs(args)
        scores = {name: data[name][1].copy() for name in ('calibration', 'test')}
        metadata_columns = list(scores['test'].columns)
        stats, full = fit_node_controls(data, scores, args, output)
        fit_context_controls(data, scores, args, output)
        terms = decompose_scores(data, scores, stats, full, args, output)
        if args.lda_prompt:
            fit_prompt_controls(data, scores, stats, full, args, output)
        compare_old_lda(args, full, data, output)
        pair_path = Path(args.pairs) if args.pairs else Path(args.root) / 'charm_in/test/cluster_audit/pairs.json'
        pairs = read_json(pair_path) if pair_path.exists() else []
        pairs = [pair for pair in pairs if pair['tier'] == 'cluster']
        write_json(output / 'pairs.json', pairs)
        save_channel_rules(stats, full, terms, data['test'], pairs, output)
        for name, table in scores.items():
            table.to_csv(output / (name + '_scores.csv.gz'), index=False)
        write_json(output / 'protocol.json', dict(metadata_columns=metadata_columns,
            score_columns=[c for c in scores['test'] if c not in metadata_columns],
            layers=data['fit'][3][0], heads=data['fit'][3][1], parts=parts,
            fit_tokens=len(data['fit'][1]), test_tokens=len(data['test'][1]),
            matched_pairs=len(pairs), prompt=args.lda_prompt,
            scope='Supervised diagnostic; post-token raw x; no CHARM training or message passing.'))
    report(output, args.bootstrap)
