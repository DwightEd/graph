"""Diagnose three questions on frozen supervised 1024-head models and saved data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .grounded_projection_data import MODEL, write_json
from .repetition_analysis import (match_tokens, pair_statistics, ranking,
    select_native_pairs, token_category)
from .repetition_teacher import (BASE, TEACHERS, SelfReadout, different_word_persistence,
    different_word_prototype, literal_repetition, prototype_banks)


DENSITY = Path('outputs/unsupervised_head_graph_20261007_v2_covariance')


def teacher_features(row, teachers):
    result, standardized = {}, None
    for seed, teacher in teachers.items():
        standardized = teacher.standardized(row)
        with torch.no_grad():
            logits = teacher.logits(standardized).numpy()
        probability = torch.tensor(logits).sigmoid().numpy()
        result[f'teacher_{seed}'] = probability
        result[f'logit_{seed}'] = logits
        result[f'tangent_{seed}'] = standardized.numpy() @ teacher.tangent()
    result['teacher'] = (result['teacher_42'] + result['teacher_123']) / 2
    result['tangent'] = (result['tangent_42'] + result['tangent_123']) / 2
    result['absolute_tangent'] = abs(result['tangent'])
    result['squared_tangent'] = result['tangent'] ** 2
    result['norm'] = np.linalg.norm(standardized.numpy(), axis=1)
    result['previous_teacher'] = np.r_[np.nan, result['teacher'][:-1]]
    return result, standardized.numpy()


def freeze_heads(teachers, output):
    directions = {seed: teacher.tangent() for seed, teacher in teachers.items()}
    magnitude = (abs(directions[42]) + abs(directions[123])) / 2
    layers = np.argsort(magnitude.reshape(32, 32).sum(axis=1))[::-1][:4].tolist()
    important = np.argsort(magnitude)[::-1][:32].tolist()
    heads = {str(layer): np.argsort(magnitude.reshape(32, 32)[layer])[::-1][:8].tolist() for layer in layers}
    top = {seed: set(np.argsort(abs(direction))[::-1][:32]) for seed, direction in directions.items()}
    metadata = dict(layers=layers, heads=heads, important=important,
        tangent_cosine=float(np.dot(directions[42], directions[123]) /
            (np.linalg.norm(directions[42]) * np.linalg.norm(directions[123]))),
        top32_shared=len(top[42] & top[123]), selection='frozen teacher derivative at fitting mean',
        final_detector=False, source='supervised model, mechanism discovery only')
    write_json(output / 'head_selection.json', metadata)
    np.savez_compressed(output / 'teacher_tangents.npz', seed42=directions[42], seed123=directions[123])
    return metadata


def cached_features(row, result, standardized, selection, density, banks):
    counts, lag = literal_repetition(row['token_ids'])
    for column, size in enumerate((1, 2, 4)):
        result[f'repeat_{size}'] = counts[:, column].astype(float)
    result['repeat_lag'] = lag.astype(float)
    result.update(different_word_persistence(standardized, row['token_ids']))
    result['cross_source_pattern'] = different_word_prototype(standardized, row['token_ids'], banks)
    channels = row['node'].reshape(-1, 1024, 4)
    result['self_mean'] = channels[..., 0].mean(axis=1)
    result['important_self'] = channels[:, selection['important'], 0].mean(axis=1)
    result['important_prompt'] = channels[:, selection['important'], 1].mean(axis=1)
    result['important_history'] = channels[:, selection['important'], 2].mean(axis=1)
    result['important_innovation'] = (channels[:, selection['important'], 0]
        - row['memories']['native'].reshape(-1, 1024, 2)[:, selection['important'], 0]).mean(axis=1)
    result['source_risk'] = density['source:' + row['id']]
    result['joint_novelty'] = density['node_joint_novelty:' + row['id']]
    result['position'] = (np.arange(len(channels)) + .5) / len(channels)
    return counts


def compute_test(rows, features):
    labels = np.concatenate([row['labels'] for row in rows])
    result = {name: ranking(labels, np.concatenate([features[row['id']][name] for row in rows]))
              for name in next(iter(features.values()))}
    onset = np.concatenate([row['labels'].astype(bool) & ~np.r_[False, row['labels'][:-1].astype(bool)]
                            for row in rows])
    for subset, allowed in (('onset_vs_normal', onset | (labels == 0)),
                             ('continuation_vs_normal', ~onset)):
        result[subset] = {name: ranking(labels[allowed],
            np.concatenate([features[row['id']][name] for row in rows])[allowed])
            for name in ('teacher', 'previous_teacher', 'repeat_1', 'recent_pattern', 'tangent')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    data = torch.load(BASE / 'prepared.pt', weights_only=False)
    teachers = {seed: SelfReadout(seed) for seed in (42, 123)}
    selection = freeze_heads(teachers, args.output)
    banks = prototype_banks(data['fit'], teachers[42])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    density = np.load(DENSITY / 'test_scores.npz')
    saved = {seed: np.load(TEACHERS / f'self_only_seed{seed}/test_scores.npz') for seed in teachers}
    features, strict, relaxed, reproduction = {}, {}, {}, []
    for row in data['test']:
        result, standardized = teacher_features(row, teachers)
        for seed in teachers:
            reproduction.append(float(np.max(abs(result[f'teacher_{seed}'] - saved[seed][row['id']]))))
        counts = cached_features(row, result, standardized, selection, density, banks)
        categories = np.array([token_category(tokenizer.decode([int(token)])) for token in row['token_ids']])
        strict[row['id']] = match_tokens(row, counts)
        relaxed[row['id']] = match_tokens(row, counts, categories)
        features[row['id']] = result
    assert max(reproduction) < 2e-5, f'frozen teacher reconstruction: {max(reproduction)}'
    np.savez_compressed(args.output / 'features.npz',
        **{name + ':' + identity: values for identity, row in features.items() for name, values in row.items()})
    write_json(args.output / 'matched_tokens.json', dict(strict=strict, relaxed=relaxed))
    names = ('teacher', 'previous_teacher', 'tangent', 'absolute_tangent', 'repeat_1', 'repeat_4',
             'recent_pattern', 'remote_pattern', 'cross_source_pattern', 'norm', 'joint_novelty',
             'important_self', 'important_prompt', 'important_history', 'important_innovation', 'source_risk')
    result = dict(metrics=compute_test(data['test'], features),
        strict=pair_statistics(data['test'], features, strict, names),
        relaxed=pair_statistics(data['test'], features, relaxed, names),
        teacher_reproduction_max=max(reproduction), prototype_bank_per_class=len(banks[0]['values']))
    write_json(args.output / 'results.json', result)
    pairs = select_native_pairs(data['test'], features, strict, relaxed)
    write_json(args.output / 'native_pairs.json', pairs)
    write_json(args.output / 'execution.json', dict(status='DONE', seconds=time.time()-started,
        test_answers=len(data['test']), natural_labels_used=True, fitted_new_detector=False,
        new_llm_forwards=0, strict_pairs=sum(map(len, strict.values())),
        relaxed_pairs=sum(map(len, relaxed.values()))))
    print('METRICS', {n: result['metrics'][n] for n in names}, flush=True)
    print('NATIVE PAIRS', pairs, flush=True)


if __name__ == '__main__':
    main()
