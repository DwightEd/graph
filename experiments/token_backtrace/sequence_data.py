"""Unlabelled source-isolated reference data for the sequence experiment."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .sequence import FEATURES


PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')
CONTROLS = (Path('outputs/token_grounding_20261006_witness_v1/manifest.json'),
            Path('outputs/token_grounding_20261006_extension_witness_v1/manifest.json'))
ARRAY_KEYS = ('context', 'observations', 'target', 'token_id', 'unit_index')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def controls(paths=CONTROLS):
    return [dict(record['original'], cohort='original8' if index == 0 else 'extension6')
            for index, path in enumerate(paths) for record in read_json(path)['records']]


def features(pack, unit_support=False):
    context, observed = pack['context'], pack['observations']
    if unit_support:
        return np.column_stack((context[:, 1], context[:, 0], observed[:, 2:7]))
    return np.column_stack((observed[:, 1] + context[:, 1], observed[:, 0] + context[:, 0],
                            observed[:, 2:7]))


def load_pack(task, split, packs=PACKS):
    path = packs / f'{task}_{split}'
    metadata = read_json(path.with_suffix('.json'))
    # The historical NPZ also contains gold: never access those members here.
    with np.load(path.with_suffix('.npz')) as saved:
        pack = {name: saved[name] for name in ARRAY_KEYS}
    return pack, metadata


def answer(pack, row, root):
    selected = slice(row['packed_start'], row['packed_stop'])
    arrays = {name: value[selected] for name, value in pack.items()}
    return dict(record=row, root=str(root), pack=arrays, values=features(arrays))


def select_reference(pack, metadata, excluded, fit_sources=32, dev_sources=16, seed=42):
    selected = {}
    for partition, count in (('fit', fit_sources), ('dev', dev_sources)):
        available = {row['source_id'] for row in metadata['records']
                     if row['partition'] == partition and row['source_id'] not in excluded}
        ordered = sorted(available, key=lambda source: hashlib.sha256(f'{seed}:{source}'.encode()).digest())
        if len(ordered) < count:
            raise ValueError(f'{partition}: fewer than {count} isolated sources')
        chosen = set(ordered[:count])
        selected[partition] = [answer(pack, row, metadata['source_cache']) for row in metadata['records']
                               if row['partition'] == partition and row['source_id'] in chosen]
    fit = {row['record']['source_id'] for row in selected['fit']}
    dev = {row['record']['source_id'] for row in selected['dev']}
    assert not (fit & dev or (fit | dev) & excluded)
    return selected['fit'], selected['dev']


def answer_weights(answers):
    counts = Counter(row['record']['source_id'] for row in answers)
    return np.array([1 / (len(counts) * counts[row['record']['source_id']] * len(row['values']))
                     for row in answers])


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, quantile * cumulative[-1])
    return float(values[order[min(index, len(order) - 1)]])


def nuisance_design(answer_, generators):
    positions = answer_['pack']['target'] / max(answer_['record']['tokens'] - 1, 1)
    length = np.full(len(positions), np.log1p(answer_['record']['tokens']))
    generator = np.array([answer_['record']['generator'] == value for value in generators[1:]], dtype=float)
    return np.column_stack((np.ones(len(positions)), positions, positions ** 2, length,
                            np.broadcast_to(generator, (len(positions), len(generator)))))


def fit_reference(answers, conditioned=True):
    generators = sorted({row['record']['generator'] for row in answers})
    values = np.concatenate([row['values'] for row in answers])
    design = np.concatenate([nuisance_design(row, generators) for row in answers])
    if not conditioned:
        design = design[:, :1]
    weights = np.concatenate([np.full(len(row['values']), weight)
                              for row, weight in zip(answers, answer_weights(answers))])
    regularizer = 1e-6 * np.eye(design.shape[1])
    regularizer[0, 0] = 0.
    coefficients = np.linalg.solve((design * weights[:, None]).T @ design + regularizer,
                                   (design * weights[:, None]).T @ values)
    residual = values - design @ coefficients
    scale = np.sqrt(np.sum(weights[:, None] * residual ** 2, axis=0))
    if (scale < 1e-8).any():
        raise ValueError('Degenerate reference feature')
    return dict(features=list(FEATURES), generators=generators, conditioned=conditioned,
                coefficients=coefficients.tolist(), scale=scale.tolist())


def transform_answer(answer_, reference):
    if answer_['record']['generator'] not in reference['generators']:
        raise ValueError('Generator absent from fitting reference')
    design = nuisance_design(answer_, reference['generators'])
    if not reference['conditioned']:
        design = design[:, :1]
    return (answer_['values'] - design @ np.asarray(reference['coefficients'])) / reference['scale']


def selected_controls(task, rows, packs=PACKS):
    result = []
    for split in ('train', 'test'):
        wanted = {row['id']: row for row in rows if row['task'] == task and row['split'] == split}
        if not wanted:
            continue
        pack, metadata = load_pack(task, split, packs)
        for row in metadata['records']:
            if row['id'] in wanted:
                merged = dict(row, cohort=wanted[row['id']]['cohort'])
                result.append(answer(pack, merged, metadata['source_cache']))
    return result
