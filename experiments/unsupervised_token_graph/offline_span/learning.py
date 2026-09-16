"""来源划分、困难配对学习；不导入评价模块，不读取 response.jsonl。"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from .controls import build_contrastive_pairs
from .data import load_graph, write_json
from .detection import fit_unlabeled_reference, score_spans
from .model import EvidenceSpanScorer
from .regions import build_views


def partition_sources(records, seed=17, calibration_fraction=.2):
    """同一source的所有回答一起划分，官方test不参加词表、训练或校准。"""
    training, testing = {}, set()
    for record in records:
        source = record['source_id']
        if not source:
            raise ValueError('source_id is required before fit; use --index with the existing inputs/records metadata')
        if record['split'] == 'test':
            testing.add(source)
        elif record['split'] == 'train':
            training[source] = record['task']
        else:
            raise ValueError('fit needs explicit train/test metadata or --train-cache/--test-cache partitions')
    if testing.intersection(training):
        raise ValueError('the same source occurs in official train and test')
    random = np.random.default_rng(seed)
    calibration = set()
    for task in sorted(set(training.values())):
        sources = sorted(source for source, value in training.items() if value == task)
        if len(sources) < 2:
            raise ValueError(f'{task}: need at least two train sources for label-free calibration')
        random.shuffle(sources)
        count = min(len(sources) - 1, max(1, round(len(sources) * calibration_fraction)))
        calibration.update(sources[:count])
    return {source: ('calibration' if source in calibration else 'fit') for source in training}


def contrastive_loss(observed_logits, reconnected_logits):
    return (functional.softplus(-observed_logits) + functional.softplus(reconnected_logits)).mean()


def create_model(graph_dir, records, hidden_size=32, relation='real'):
    vocabulary, channels, node_sizes, modes = set(), set(), set(), set()
    for record in tqdm(records, desc='fit vocabulary / channels'):
        graph = load_graph(Path(graph_dir) / record['file'])
        vocabulary.update(graph.sample.token_ids.tolist())
        channels.update(map(tuple, graph.channels.tolist()))
        node_sizes.add(graph.node_features.shape[1])
        modes.add(graph.feature_mode)
    if len(node_sizes) != 1 or len(modes) != 1:
        raise ValueError('do not mix token-only and hidden layouts in one detector')
    config = dict(vocabulary=sorted(vocabulary), channels=sorted(channels), node_size=node_sizes.pop(),
                  hidden_size=hidden_size, relation=relation)
    return EvidenceSpanScorer(**config), config


def train_scorer(model, graph_dir, records, epochs=5, learning_rate=.001, max_length=32,
                 future_budget=8, pair_budget=128, seed=17, checkpoint=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    random = np.random.default_rng(seed)
    by_source = {}
    for record in records:
        by_source.setdefault(record['source_id'], []).append(record)
    history, first_epoch = [], 0
    if checkpoint is not None and Path(checkpoint).is_file():
        saved = torch.load(checkpoint, map_location=model.device, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        random.bit_generator.state = saved['random_state']
        history = saved['history']
        first_epoch = saved['epoch'] + 1
    for epoch in range(first_epoch, epochs):
        # One answer per source per epoch: repeated generators cannot dominate training.
        selected = [rows[int(random.integers(len(rows)))] for rows in by_source.values()]
        random.shuffle(selected)
        losses, pairs_used, skipped = [], 0, 0
        model.train()
        for record in tqdm(selected, desc=f'epoch {epoch + 1}/{epochs}'):
            graph = load_graph(Path(graph_dir) / record['file'])
            spans, index = build_views(graph, max_length, future_budget)
            pairs = build_contrastive_pairs(graph, spans, index, pair_budget, int(random.integers(2**31)))
            if not pairs:
                skipped += 1
                continue
            optimizer.zero_grad()
            encoded = model.encode_graph(graph)
            kind_losses = []
            for kind in ('evidence', 'reuse'):
                subset = [pair for pair in pairs if pair.control_kind == kind]
                if subset:
                    observed = model.score_encoded(encoded, [pair.observed for pair in subset])
                    reconnected = model.score_encoded(encoded, [pair.reconnected for pair in subset])
                    kind_losses.append(contrastive_loss(observed, reconnected))
            loss = torch.stack(kind_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            pairs_used += len(pairs)
        if not losses:
            raise ValueError('No valid structural contrast pairs; inspect source groups/context width before training')
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), pairs=pairs_used, skipped_sources=skipped)
        history.append(row)
        if checkpoint is not None:
            partial = Path(checkpoint).with_suffix('.partial.pt')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            random_state=random.bit_generator.state, epoch=epoch, history=history), partial)
            partial.replace(checkpoint)
        print(json.dumps(row), flush=True)
    return history


def fit_detector(graph_dir, output, device='cpu', epochs=5, hidden_size=32, max_length=32,
                 future_budget=8, pair_budget=128, seed=17, relation='real', learning_rate=.001,
                 calibration_fraction=.2, tail_quantile=.95, boundary_penalty=2., resume=False):
    root, output = Path(graph_dir), Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    records = json.loads((root / 'manifest.json').read_text())['records']
    split = partition_sources(records, seed, calibration_fraction)
    fit_rows = [row for row in records if row['split'] == 'train' and split[row['source_id']] == 'fit']
    cal_rows = [row for row in records if row['split'] == 'train' and split[row['source_id']] == 'calibration']
    torch.manual_seed(seed)
    model, model_config = create_model(root, fit_rows, hidden_size, relation)
    model.to(device)
    history = train_scorer(model, root, fit_rows, epochs, learning_rate, max_length, future_budget, pair_budget, seed,
                           output / 'checkpoint.pt')

    calibration_records = []
    for record in tqdm(cal_rows, desc='unlabeled calibration'):
        graph = load_graph(root / record['file'])
        spans, _ = build_views(graph, max_length, future_budget)
        if spans:
            scores = score_spans(model, graph, spans)
            lengths = np.asarray([span.end - span.start for span in spans])
            group = graph.sample.task + '|' + graph.sample.generator
            calibration_records.append((group, record['source_id'], lengths, scores))
    reference = fit_unlabeled_reference(calibration_records, tail_quantile)
    torch.save(model.cpu().state_dict(), output / 'weights.pt')
    settings = dict(model=model_config, max_length=max_length, future_budget=future_budget,
                    pair_budget=pair_budget, boundary_penalty=boundary_penalty, seed=seed,
                    source_partition=split, fit_answers=len(fit_rows), calibration_answers=len(cal_rows),
                    feature_mode='hidden' if model_config['node_size'] else 'tokens',
                    labels_used=False, objective='observed vs matched reconnected pairs, not truth labels')
    write_json(output / 'model.json', settings)
    write_json(output / 'reference.json', reference)
    write_json(output / 'training.json', history)
    write_json(output / 'complete.json', dict(complete=True, labels_used=False))


def load_scorer(model_dir, device='cpu'):
    root = Path(model_dir)
    settings = json.loads((root / 'model.json').read_text())
    model = EvidenceSpanScorer(**settings['model']).to(device)
    model.load_state_dict(torch.load(root / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    return model, settings, json.loads((root / 'reference.json').read_text())
