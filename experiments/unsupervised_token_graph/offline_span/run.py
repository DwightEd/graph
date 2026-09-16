"""一个入口完成 prepare → fit → score → evaluate；前3阶段不打开标注。"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import load_graph, load_observations, load_samples, save_graph, write_json
from .detection import decode_spans, score_spans
from .graph import build_token_graph
from .learning import fit_detector, load_scorer
from .regions import build_views


DATA_ROOT = Path('/share/home/tm902089733300000/a903202310/lys/data/RAGTruth')
CACHE_ROOT = DATA_ROOT / 'attention/llama31_8b'


def prepare_graphs(args, output):
    output = Path(output)
    settings = {key: getattr(args, key) for key in (
        'train_cache', 'test_cache', 'index', 'tasks', 'generators', 'feature_mode',
        'feature_root', 'hidden_layer', 'edges_per_partition', 'context_width', 'layers', 'heads', 'limit')}
    settings['version'] = 'offline-span-graph-v1'
    if output.exists():
        if not args.resume or json.loads((output / 'settings.json').read_text()) != settings:
            raise ValueError('choose a new output, or --resume with identical graph settings')
    else:
        output.mkdir(parents=True)
        write_json(output / 'settings.json', settings)
    records = []
    tasks = [] if args.tasks == ['all'] else args.tasks
    generators = [] if args.generators == ['all'] else args.generators

    for split, cache in [('train', args.train_cache), ('test', args.test_cache)]:
        if cache is None:
            continue
        samples, index = load_samples(cache, args.index, split, tasks, generators)
        if args.limit:
            samples = samples[:args.limit]
        (output / split).mkdir(exist_ok=True)
        for number, sample in enumerate(tqdm(samples, desc=f'prepare {split}')):
            relative = f'{split}/{number:06d}.npz'
            path = output / relative
            if args.resume and path.is_file():
                graph = load_graph(path)
                if graph.sample.metadata() != sample.metadata() or not np.array_equal(graph.sample.token_ids, sample.token_ids):
                    raise ValueError('resumed graph roster changed; choose a new output')
            else:
                observations = load_observations(sample, index, args.feature_mode, args.hidden_layer, args.feature_root)
                graph = build_token_graph(sample, observations, args.edges_per_partition,
                                          args.context_width, args.layers, args.heads)
                save_graph(graph, path)
            pruned = graph.masses[:, :, 3]
            observed_pruned = pruned[np.isfinite(pruned)]
            report = dict(file=relative, **sample.metadata(), tokens=sample.response_length,
                          coverage=float(graph.coverage.mean()), edges=len(graph.edges),
                          channels=len(graph.channels), hidden_size=graph.node_features.shape[1],
                          entropy_present=bool(np.isfinite(graph.entropy).any()),
                          offsets_present=bool(len(sample.offsets)),
                          mean_pruned_mass=float(observed_pruned.mean()) if len(observed_pruned) else None)
            records.append(report)
    write_json(output / 'manifest.json', dict(records=records, labels_used=False, complete=True))
    print(f'Prepared {len(records)} answers: {output}', flush=True)


def score_answers(graph_dir, model_dir, output, device='cpu', split='test', resume=False, batch_size=256):
    graph_dir, output = Path(graph_dir), Path(output)
    model, settings, reference = load_scorer(model_dir, device)
    manifest = json.loads((graph_dir / 'manifest.json').read_text())
    records = [record for record in manifest['records'] if record['split'] == split]
    if not records:
        raise ValueError(f'No {split} graphs in {graph_dir}')
    output.mkdir(parents=True, exist_ok=resume)
    (output / 'samples').mkdir(exist_ok=True)
    config = dict(graph_dir=str(graph_dir.resolve()), model_dir=str(Path(model_dir).resolve()),
                  split=split, labels_used=False, inference='complete-answer offline', model_settings=settings)
    config_path = output / 'settings.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('saved prediction configuration differs; use a new output')
    write_json(config_path, config)
    saved = []
    for number, record in enumerate(tqdm(records, desc=f'score {split}')):
        path = output / 'samples' / f'{number:06d}.npz'
        if resume and path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                row = json.loads(str(archive['record_json']))
            if row['id'] != record['response_id']:
                raise ValueError('prediction roster changed')
            saved.append(row)
            continue
        graph = load_graph(graph_dir / record['file'])
        spans, _ = build_views(graph, settings['max_length'], settings['future_budget'])
        scores = score_spans(model, graph, spans, batch_size)
        result, potentials = decode_spans(graph, spans, scores, reference, settings['boundary_penalty'])
        singleton = np.full(graph.sample.response_length, np.nan)
        for span, value in zip(spans, scores):
            if span.end == span.start + 1:
                singleton[span.start] = value
        valid_mass = np.isfinite(graph.masses[:, :, 0])
        difference = graph.masses[:, :, 1] - graph.masses[:, :, 0]
        history = np.divide(np.nansum(difference, axis=0), valid_mass.sum(axis=0),
                            out=np.full(graph.sample.response_length, np.nan), where=valid_mass.sum(axis=0) > 0)
        row = dict(id=graph.sample.response_id, source_id=graph.sample.source_id, task=graph.sample.task,
                   generator=graph.sample.generator, split=graph.sample.split,
                   response_sha256=graph.sample.response_sha256, cache=graph.sample.cache_files[0],
                   file=path.relative_to(output).as_posix(), tokens=graph.sample.response_length)
        arrays = dict(record_json=json.dumps(row), token_ids=graph.sample.token_ids,
                      prompt_length=graph.sample.prompt_length, total_tokens=len(graph.sample.token_ids),
                      offline_span=result.token_scores, singleton=singleton, history_minus_prompt=history,
                      entropy=graph.entropy, position=np.log1p(np.arange(graph.sample.response_length)),
                      coverage=result.covered_tokens, span_bounds=result.span_bounds, span_scores=result.span_scores,
                      candidate_bounds=np.asarray([(span.start, span.end) for span in spans], int).reshape(-1, 2),
                      candidate_scores=scores, candidate_potentials=potentials)
        if len(graph.sample.offsets):
            arrays['offsets'] = graph.sample.offsets
        partial = path.with_suffix('.partial.npz')
        np.savez_compressed(partial, **arrays)
        partial.replace(path)
        saved.append(row)
    write_json(output / 'prediction_freeze.json', dict(complete=True, labels_used=False, records=saved))
    print(f'Frozen {len(saved)} answers: {output}', flush=True)


def inspect_inputs(args):
    for split, cache in [('train', args.train_cache), ('test', args.test_cache)]:
        if cache is None:
            continue
        samples, index = load_samples(cache, args.index, split)
        sample = samples[0]
        observations = load_observations(sample, index, args.feature_mode, args.hidden_layer, args.feature_root)
        graph = build_token_graph(sample, observations, args.edges_per_partition, args.context_width, args.layers, args.heads)
        print(json.dumps(dict(split=split, answers=len(samples), first_id=sample.response_id,
                              missing_source_ids=sum(not item.source_id for item in samples),
                              unknown_tasks=sum(item.task == 'unknown' for item in samples),
                              channel_count=len(graph.channels), tokens=sample.response_length,
                              coverage=float(graph.coverage.mean()), hidden=graph.node_features.shape,
                              saved_offsets=bool(len(sample.offsets)), feature_mode=graph.feature_mode), ensure_ascii=False))


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--phase', choices=['all', 'inspect', 'prepare', 'fit', 'score', 'evaluate'], default='all')
    command.add_argument('--train-cache', default=str(CACHE_ROOT / 'train'))
    command.add_argument('--test-cache', default=str(CACHE_ROOT / 'test'))
    command.add_argument('--index', help='existing inputs.jsonl/records.jsonl, or its directory; never rebuild metadata')
    command.add_argument('--output', default='outputs/offline_span_v1')
    command.add_argument('--tasks', nargs='+', default=['all'])
    command.add_argument('--generators', nargs='+', default=['all'])
    command.add_argument('--feature-mode', choices=['tokens', 'hidden'], default='tokens')
    command.add_argument('--feature-root', help='optional existing <response_id>.npz hidden cache; token IDs must match')
    command.add_argument('--hidden-layer', type=int, help='saved array index when hidden_states is [layers,N,D]')
    command.add_argument('--layers', type=int, nargs='+')
    command.add_argument('--heads', type=int, nargs='+')
    command.add_argument('--edges-per-partition', type=int, default=2, help='top prompt/history edges per head-row; 0 retains all')
    command.add_argument('--context-width', type=int, default=32)
    command.add_argument('--max-length', type=int, default=32)
    command.add_argument('--future-budget', type=int, default=8)
    command.add_argument('--pair-budget', type=int, default=128)
    command.add_argument('--hidden-size', type=int, default=32)
    command.add_argument('--relation', choices=['real', 'none', 'permuted'], default='real')
    command.add_argument('--epochs', type=int, default=5)
    command.add_argument('--learning-rate', type=float, default=.001)
    command.add_argument('--calibration-fraction', type=float, default=.2)
    command.add_argument('--tail-quantile', type=float, default=.95)
    command.add_argument('--boundary-penalty', type=float, default=2.)
    command.add_argument('--batch-size', type=int, default=256, help='scoring spans per batch, graph is encoded once per answer')
    command.add_argument('--device', default='cpu')
    command.add_argument('--threads', type=int, default=4)
    command.add_argument('--seed', type=int, default=17)
    command.add_argument('--limit', type=int, default=0, help='explicit execution-order subset per split, not a full benchmark')
    command.add_argument('--resume', action='store_true')
    command.add_argument('--annotations', default=str(DATA_ROOT / 'dataset/response.jsonl'))
    command.add_argument('--tokenizer', help='original local observer tokenizer for evaluation offset verification only')
    command.add_argument('--source-info')
    command.add_argument('--bootstrap', type=int, default=200)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    if min(args.context_width, args.max_length, args.pair_budget, args.hidden_size, args.epochs, args.batch_size) < 1:
        raise ValueError('length, width, pair/batch size and epochs must be positive')
    if not 0 < args.calibration_fraction < 1 or not 0 < args.tail_quantile < 1:
        raise ValueError('calibration fraction and quantile must lie in (0,1)')
    if min(args.edges_per_partition, args.future_budget, args.limit, args.bootstrap) < 0:
        raise ValueError('budgets cannot be negative')
    root = Path(args.output)
    if args.phase == 'inspect':
        inspect_inputs(args)
        return
    if args.phase in ('all', 'prepare'):
        prepare_graphs(args, root / 'graphs')
    if args.phase in ('all', 'fit'):
        fit_options = {key: getattr(args, key) for key in (
            'epochs', 'hidden_size', 'max_length', 'future_budget', 'pair_budget', 'seed',
            'relation', 'learning_rate', 'calibration_fraction', 'tail_quantile', 'boundary_penalty')}
        options_path = root / 'fit_options.json'
        if options_path.is_file() and json.loads(options_path.read_text()) != fit_options:
            raise ValueError('training settings changed; select a new --output rather than reuse old weights')
        write_json(options_path, fit_options)
        complete = root / 'model/complete.json'
        if not (args.resume and complete.is_file()):
            fit_detector(root / 'graphs', root / 'model', args.device, args.epochs, args.hidden_size,
                         args.max_length, args.future_budget, args.pair_budget, args.seed, args.relation,
                         args.learning_rate, args.calibration_fraction, args.tail_quantile, args.boundary_penalty, args.resume)
    if args.phase in ('all', 'score'):
        score_answers(root / 'graphs', root / 'model', root / 'predictions', args.device, resume=args.resume,
                      batch_size=args.batch_size)
    if args.phase in ('all', 'evaluate'):
        if Path(args.annotations).is_file():
            from .evaluation import evaluate_saved_predictions
            evaluate_saved_predictions(root / 'predictions', args.annotations, args.tokenizer, args.source_info, args.bootstrap)
        else:
            print(f'Evaluation skipped: annotation file absent: {args.annotations}. Frozen scores are preserved.', flush=True)


if __name__ == '__main__':
    main()
