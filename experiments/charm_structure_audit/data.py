"""Reuse existing canonical NPZ/identity readers; build once, then audit CHARM."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.unsupervised_token_graph.cache_index import CacheIndex, file_stamp
from experiments.unsupervised_token_graph.data import ResponseCache
from experiments.unsupervised_token_graph.evaluation_data import EvaluationBinding, prepare_record, read_sources

from .graph import build_graph


SCHEMA = "charm-structure-audit-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_graph(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def label_arrays(offsets, spans):
    """Same half-open character-overlap rule as the existing label_views.

    Retain each original annotation interval AND its union of error tokens.
    Adjacent annotations are not silently treated as one fact choice.
    """
    y, onset = np.zeros(len(offsets), bool), np.zeros(len(offsets), bool)
    intervals = []
    for span in spans:
        hit = (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start']) & (offsets[:, 1] > offsets[:, 0])
        ids = np.flatnonzero(hit)
        if not len(ids):
            raise ValueError('gold span has no token overlap; verify original offsets')
        y |= hit
        onset[ids[0]] = True
        intervals.append([int(ids[0]), int(ids[-1]) + 1])
    return y, onset, np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def prepare(attention_root, annotations, output, tokenizer=None, tau=.05,
            tasks=("QA", "Summary", "Data2txt"), generator="llama-2-7b-chat", limit=0, splits=("train", "test")):
    """Precompute supervised audit data; never modify original caches/results.

    limit is per task/split and takes deterministic file order, not a random
    benchmark subset. Training on a limit is recorded as a pilot.
    """
    root, output = Path(attention_root).resolve(), Path(output)
    with Path(annotations).open(encoding='utf-8') as stream:
        gold = {str(row['id']): row for row in (json.loads(line) for line in stream if line.strip())}
    sources, source_path = read_sources(annotations)
    config = dict(schema=SCHEMA, attention_root=str(root), annotations=file_stamp(annotations),
                  sources=file_stamp(source_path) if source_path else None,
                  tokenizer=str(tokenizer) if tokenizer else None, tau=tau, tasks=list(tasks),
                  generator=generator, limit=limit, splits=list(splits), alignment='post_token_i_to_label_i')
    settings = output / 'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != config:
        raise ValueError('audit data settings changed; choose another prepared output')
    write_json(settings, config)
    records, used_ids = [], set()
    for split in splits:
        cache_root = root / split
        index = CacheIndex(cache_root)
        reader = ResponseCache(index=index)
        binding = EvaluationBinding(dict(cache=str(cache_root)), tokenizer)
        counts = {task: 0 for task in tasks}
        files = sorted(cache_root.rglob('*.npz'))
        if not files:
            raise FileNotFoundError('no canonical NPZs in ' + str(cache_root))
        for path in tqdm(files, desc='prepare ' + split, unit='sample'):
            # Existing canonical naming permits filtering before loading edges.
            rid = path.stem.removeprefix('attention_')
            annotation = gold.get(rid)
            if annotation is None:
                raise ValueError('attention filename is not a RAGTruth response ID: ' + str(path))
            task = sources.get(str(annotation['source_id']), {}).get('task_type', '')
            if task not in tasks or (generator and annotation['model'] != generator):
                continue
            if annotation['split'] != split:
                raise ValueError('official split conflicts with cache directory: ' + str(path))
            if limit and counts[task] >= limit:
                continue
            if rid in used_ids:
                raise ValueError('duplicate response cache: ' + rid)
            used_ids.add(rid); counts[task] += 1
            destination = output / 'graphs' / split / (rid + '.npz')
            stamp = file_stamp(path)
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    row = json.loads(str(saved['record_json']))
                if row['cache_stamp'] != stamp:
                    raise ValueError('source cache changed: ' + str(path))
                records.append(row)
                continue
            record = reader.load(path)
            floor = None
            for parent in (path.parent, *path.parents):
                manifest = parent / 'manifest.json'
                if manifest.is_file():
                    floor = json.loads(manifest.read_text()).get('attention_floor')
                    if floor is not None:
                        break
            if floor is not None and float(floor) > tau:
                raise ValueError('cache floor exceeds graph tau; missing edges cannot be recovered')
            graph = build_graph(record, tau)
            # The graph above cannot inspect these labels.
            relative = path.relative_to(cache_root).as_posix()
            identity = dict(record.metadata or {}, id=record.response_id, source_id=record.source_id,
                            cache=relative, file='samples/' + relative)
            identity = prepare_record(dict(cache=str(cache_root)), identity)
            arrays = dict(token_ids=record.token_ids, prompt_length=record.prompt_length)
            if record.offsets is not None:
                arrays['offsets'] = record.offsets
            identity, offsets = binding.bind(identity, annotation, arrays, sources)
            if len(offsets) != len(graph['x']) - record.response_idx:
                raise ValueError('response offsets and graph token count differ: ' + rid)
            y, onset, spans = label_arrays(offsets, annotation['labels'])
            row = dict(id=rid, source_id=identity['source_id'], split=split, task=task,
                       generator=annotation['model'], graph=str(destination.resolve()),
                       cache_stamp=stamp, attention_floor=floor, response_tokens=len(y), positives=int(y.sum()),
                       channels=graph['x'].shape[1], edges=graph['edge_index'].shape[1])
            save_npz(destination, **graph, gold=y, onset=onset, spans=spans, offsets=offsets,
                     token_ids=record.token_ids, response=np.asarray(annotation['response']),
                     record_json=np.asarray(json.dumps(row)))
            records.append(row)
    if not records:
        raise ValueError('no samples match tasks/generator')
    write_json(output / 'index.json', records)
    return records


def partitions(records, seed=42):
    """Source-disjoint fit/select/calibrate within official train; test untouched."""
    train_sources = {r['source_id'] for r in records if r['split'] == 'train'}
    test_sources = {r['source_id'] for r in records if r['split'] == 'test'}
    if train_sources & test_sources:
        raise ValueError('source overlap between official train and test')
    order = sorted(train_sources, key=lambda sid: hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest())
    if len(order) < 4:
        raise ValueError('need >=4 train sources for fit/select/calibrate; raise --limit')
    n = max(1, int(.1 * len(order)))
    select, calibration = set(order[:n]), set(order[n:2*n])
    result = dict(fit=[], select=[], calibration=[], test=[])
    for row in records:
        part = 'test' if row['split'] == 'test' else (
            'select' if row['source_id'] in select else 'calibration' if row['source_id'] in calibration else 'fit')
        result[part].append(row)
    return result
