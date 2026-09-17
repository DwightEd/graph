"""Read existing graphs and scores. Labels stay outside model inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


GRAPH_KEYS = ('x', 'edge_index', 'edge_attr', 'edge_mark', 'prompt_length', 'layers', 'heads')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_scores(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def load_graph(record, prepared):
    path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    identity = json.loads(str(arrays['record_json']))
    if str(identity['id']) != str(record['id']) or str(identity['source_id']) != str(record['source_id']):
        raise ValueError('Prepared graph identity differs from record: ' + str(path))
    graph = {key: arrays[key] for key in GRAPH_KEYS}
    return graph, arrays


def load_predictions(directory):
    """Original CHARM sample format. Load scores, not saved embeddings."""
    directory = Path(directory)
    paths = read_json(directory / 'predictions.json')
    samples = []
    for old_path in paths:
        with np.load(directory / 'samples' / Path(old_path).name, allow_pickle=False) as saved:
            record = json.loads(str(saved['record_json']))
            sample = {key: saved[key] for key in ('score', 'gold', 'onset', 'spans', 'offsets', 'response')}
        samples.append(dict(record, **sample))
    if len({str(s['id']) for s in samples}) != len(samples) or not samples:
        raise ValueError('Predictions must contain unique, nonempty response IDs')
    return samples


def read_tables(directory):
    """CSV exports are sufficient for report mode; no torch or graph load."""
    directory = Path(directory)
    tokens = pd.read_csv(directory / 'tokens.csv', keep_default_na=False,
                         dtype={'id': str, 'source_id': str})
    spans = pd.read_csv(directory / 'spans.csv', dtype={'id': str, 'source_id': str})
    if tokens.duplicated(['id', 'token']).any() or not np.isfinite(tokens.score).all():
        raise ValueError('Duplicate coordinates or nonfinite baseline scores')
    return tokens, spans


def token_frame(sample, score, threshold):
    offsets = sample['offsets']
    text = str(sample['response'])
    count = len(score)
    if count != len(sample['gold']):
        raise ValueError('Score/label token alignment differs')
    return pd.DataFrame(dict(id=str(sample['id']), source_id=str(sample['source_id']),
        token=np.arange(count), text=[text[a:b] for a, b in offsets],
        gold=sample['gold'].astype(int), score=score, predicted=(score > threshold).astype(int)))


def original_parts(prepared, recipe):
    """Reuse the saved fit/select/calibration/test partition. Never split tokens."""
    index = {str(r['id']): r for r in read_json(Path(prepared) / 'index.json')}
    parts = {name: [index[str(rid)] for rid in ids] for name, ids in recipe['partitions'].items()}
    source_sets = [set(str(r['source_id']) for r in rows) for rows in parts.values()]
    for i, left in enumerate(source_sets):
        if any(left & right for right in source_sets[i + 1:]):
            raise ValueError('Original source partitions overlap')
    for name, rows in parts.items():
        split = 'test' if name == 'test' else 'train'
        if any(r['split'] != split for r in rows):
            raise ValueError('Original partition conflicts with official split')
    return parts


def prepare_output(output, config):
    """One configuration check at the run boundary; no per-feature identities."""
    output = Path(output)
    path = output / 'config.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Output already belongs to another run; use another --output')
    write_json(path, config)
    return output
