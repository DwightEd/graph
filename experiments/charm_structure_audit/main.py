"""One entry for CHARM experiments. New position/head modes: LOCALIZATION.md."""

import argparse
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import (read_json, write_json, load_graph, load_predictions, read_tables,
                   token_frame, save_scores, original_parts, prepare_output)
from .evaluate import analyze, compare_pairs
from .ablations import MODEL_ABLATIONS, GRAPH_ABLATIONS, change_graph, topology_change


# Change this list or use --ablations. These are independent deletions, not new features.
ABLATIONS = ['no_graph', 'no_node', 'no_edge', 'no_source', 'no_mark',
             'no_prompt', 'no_history', 'no_residual', 'no_relay', 'head_mean']
MODELS = ['charm_in', 'charm_out', 'node_only', 'local_in', 'rewire_in']
DEFAULT_ROOT = 'outputs/charm_structure_audit_qa/QA/seed_0'
DEFAULT_PREPARED = 'outputs/charm_structure_audit_qa/data'


def read_pairs(args):
    path = Path(args.pairs) if args.pairs else Path(args.root)/'charm_in/test/cluster_audit/pairs.json'
    if args.pairs and not path.exists():
        raise FileNotFoundError(path)
    if not path.exists():
        print('没有已保存的配对文件；本次不报告配对成绩。', flush=True)
        return []
    return read_json(path)


def run_report(args, output, pairs):
    """Only tables: old results can be analyzed without raw graphs or torch."""
    summaries, matched = [], {}
    reference = None
    for name in args.models:
        directory = Path(args.root)/name/'test'
        if not (directory/'tokens.csv').exists():
            print('未发现已完成模型：', name, flush=True)
            continue
        table, spans = read_tables(directory)
        identity = table[['id', 'source_id', 'token', 'gold', 'text']].reset_index(drop=True)
        if reference is not None and not reference.equals(identity):
            raise ValueError('Model CSVs do not have identical token identities/order')
        reference = identity
        threshold = read_json(Path(args.root)/name/'threshold.json')['value']
        summary, matched[name] = analyze(table, spans, threshold, pairs, output/name, args.bootstrap)
        summaries.append(dict(model=name, **summary))
        if name == 'node_only' and (directory/'ewma/tokens.csv').exists():
            smooth, smooth_spans = read_tables(directory/'ewma')
            if not identity.equals(smooth[identity.columns].reset_index(drop=True)):
                raise ValueError('EWMA token identities differ')
            value = read_json(Path(args.root)/name/'threshold.json')['ewma']['value']
            summary, matched['node_ewma'] = analyze(smooth, smooth_spans, value, pairs, output/'node_ewma', args.bootstrap)
            summaries.append(dict(model='node_ewma', **summary))
    if not summaries:
        raise ValueError('No completed model CSVs found under --root')
    pd.DataFrame(summaries).to_csv(output/'models.csv', index=False)
    compare_pairs(matched, output, args.bootstrap, reference='charm_in' if 'charm_in' in matched else 'full')
    print(pd.DataFrame(summaries)[['model', 'auroc', 'ap', 'recall', 'fpr']].to_string(index=False))


def run_frozen(args, output, pairs):
    """Same checkpoint, original divisor. Save only scores, never embeddings."""
    import torch
    from .model import load_checkpoint, degree

    directory = Path(args.root)/'charm_in'
    threshold = read_json(directory/'threshold.json')['value']
    samples = load_predictions(directory/'test')
    checkpoint = Path(args.checkpoint) if args.checkpoint else directory/'checkpoint.pt'
    model, _ = load_checkpoint(checkpoint, args.device, 'in', args.edge_chunk)
    names = list(dict.fromkeys(['full', *args.ablations]))
    tables = {name: [] for name in names}
    changes = []
    for sample in tqdm(samples, desc='frozen module ablations', unit='answer'):
        graph, original = load_graph(sample, args.prepared)
        if any(not np.array_equal(sample[k], original[k]) for k in ('gold', 'offsets', 'spans')):
            raise ValueError('Prediction and prepared graph token alignment differs')
        path = output/'scores'/(str(sample['id'])+'.npz')
        seed = zlib.crc32(str(sample['id']).encode()) + args.seed
        counts = degree(graph)
        with torch.no_grad():
            replay = torch.sigmoid(model(graph)[int(graph['prompt_length']):]).cpu().numpy()
        if not np.allclose(replay, sample['score'], atol=2e-5, rtol=0):
            raise ValueError('Original checkpoint does not replay saved scores; stopping attribution')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                scores = {name: saved[name] for name in names}
            if not np.allclose(scores['full'], replay, atol=2e-5, rtol=0):
                raise ValueError('Previously saved ablation scores have a different baseline')
        else:
            scores = {'full': replay}
            for name in names[1:]:
                view = change_graph(graph, name, seed)
                with torch.no_grad():
                    logits = model(view, ablation=name, divisor=counts)
                    scores[name] = torch.sigmoid(logits[int(graph['prompt_length']):]).cpu().numpy()
            save_scores(path, **scores)
        for name in names:
            tables[name].append(token_frame(sample, scores[name], threshold))
            if name in GRAPH_ABLATIONS:
                view = change_graph(graph, name, seed)
                changes.append(dict(id=str(sample['id']), ablation=name, **topology_change(graph, view)))
    _, spans = read_tables(directory/'test')
    save_experiments(tables, spans, threshold, pairs, output, args.bootstrap)
    pd.DataFrame(changes).to_csv(output/'graph_changes.csv', index=False)


def save_experiments(tables, spans, threshold, pairs, output, bootstrap):
    summaries, matched = [], {}
    for name, frames in tables.items():
        summary, matched[name] = analyze(pd.concat(frames, ignore_index=True), spans, threshold,
                                         pairs, output/name, bootstrap)
        summaries.append(dict(ablation=name, **summary))
    pd.DataFrame(summaries).to_csv(output/'models.csv', index=False)
    compare_pairs(matched, output, bootstrap, reference='full')
    print(pd.DataFrame(summaries)[['ablation', 'auroc', 'ap', 'recall', 'fpr']].to_string(index=False))


def run_training(args, output, pairs):
    """Only this explicit mode starts independent supervised training."""
    from .model import load_checkpoint
    from .train import fit, predict, calibrate

    recipe = read_json(Path(args.root)/'charm_in/training.json')
    parts = original_parts(args.prepared, recipe)
    if args.epochs is not None:
        recipe['epochs'] = args.epochs
    recipe['seed'] = args.seed
    _, spans = read_tables(Path(args.root)/'charm_in/test')
    summaries, matched = [], {}
    for name in ['full', *[n for n in args.ablations if n != 'full']]:
        directory = prepare_output(output/name, dict(recipe=recipe, ablation=name, prepared=str(Path(args.prepared).resolve())))
        checkpoint = fit(parts, args.prepared, name, recipe, directory, args.device, args.edge_chunk)
        model, _ = load_checkpoint(checkpoint, args.device, 'in', args.edge_chunk)
        calibration = predict(model, parts['calibration'], args.prepared, name, recipe['seed'])
        threshold = calibrate(calibration, recipe['fpr'])
        write_json(directory/'threshold.json', threshold)
        samples = predict(model, parts['test'], args.prepared, name, recipe['seed'])
        table = pd.concat([token_frame(s, s['score'], threshold['value']) for s in samples], ignore_index=True)
        summary, matched[name] = analyze(table, spans, threshold['value'], pairs, directory/'test', args.bootstrap)
        table.to_csv(directory/'test/tokens.csv', index=False)
        spans.to_csv(directory/'test/spans.csv', index=False)
        summaries.append(dict(ablation=name, **summary))
    pd.DataFrame(summaries).to_csv(output/'models.csv', index=False)
    compare_pairs(matched, output, args.bootstrap, reference='full')


def run_matching(args, output):
    from .matching import match_answer, NAMES

    samples = load_predictions(Path(args.root)/'charm_in/test')
    pairs, status, skipped = [], [], []
    for record in tqdm(samples, desc='score-blind span matching', unit='answer'):
        graph, sample = load_graph(record, args.prepared)
        # The matcher receives graph/labels/text/token IDs, never detector scores.
        sample = dict(sample, id=record['id'], source_id=record['source_id'])
        answer_pairs, answer_status, answer_skipped = match_answer(graph, sample)
        pairs.extend(answer_pairs)
        status.extend(dict(row, id=record['id']) for row in answer_status)
        skipped.extend(dict(row, id=record['id']) for row in answer_skipped)
    write_json(output/'pairs.json', pairs)
    pd.DataFrame(status).to_csv(output/'matching_status.csv', index=False)
    pd.DataFrame(skipped).to_csv(output/'skipped.csv', index=False)
    balance = [dict(id=p['id'], tier=p['tier'], error_start=p['error_start'], feature=name,
        error=p['error_structure'][i], normal=p['normal_structure'][i])
        for p in pairs for i,name in enumerate(NAMES)]
    pd.DataFrame(balance).to_csv(output/'balance.csv', index=False)
    table, spans = read_tables(Path(args.root)/'charm_in/test')
    threshold = read_json(Path(args.root)/'charm_in/threshold.json')['value']
    analyze(table, spans, threshold, pairs, output/'charm_in', args.bootstrap)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['report', 'ablate', 'train', 'match', 'locate', 'routes', 'heads'], default='report')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--prepared', default=DEFAULT_PREPARED)
    parser.add_argument('--output', help='New output directory; default <root>/audit_<mode>')
    parser.add_argument('--pairs', help='Reuse one locked pairs.json for every model')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--ablations', nargs='+', choices=MODEL_ABLATIONS+GRAPH_ABLATIONS, default=ABLATIONS)
    parser.add_argument('--checkpoint', help='Original charm_in checkpoint, only for frozen ablations')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--edge-chunk', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, help='Explicit training override; otherwise original recipe')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--pair-tier', choices=['context', 'cluster', 'cluster_heads'], default='cluster')
    parser.add_argument('--window', type=int, default=5, help='Common normal context for route pre/post observations')
    parser.add_argument('--channel-unit', choices=['layer', 'head'], default='layer')
    parser.add_argument('--llm-layers', nargs='+', type=int, help='Original LLM layer indices for channel masks')
    parser.add_argument('--channels', nargs='+', help='Exact original LLM coordinates, e.g. 0:0 1:3 (not selected heads)')
    parser.add_argument('--channel-operations', nargs='+', choices=['zero', 'coupled', 'independent'], default=['zero'])
    parser.add_argument('--channel-sites', nargs='+', choices=['node', 'edge', 'both'], default=['node', 'edge'])
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else Path(args.root)/('audit_'+args.mode)
    protected = [Path(args.root), Path(args.prepared)]
    protected += [Path(args.root)/name/'test' for name in MODELS]
    if output.resolve() in [p.resolve() for p in protected] or (output/'prediction_settings.json').exists():
        raise ValueError('Use a separate output directory, not original data/results')
    config = vars(args).copy()
    if args.mode in ('report', 'ablate', 'train', 'match'):
        # Preserve actual saved configurations of the four existing workflows.
        for key in ('pair_tier', 'window', 'channel_unit', 'llm_layers', 'channels', 'channel_sites', 'channel_operations'):
            config.pop(key)
    output = prepare_output(output, config)
    if args.mode in ('ablate', 'train', 'heads'):
        import torch
        torch.set_num_threads(1)
    pairs = read_pairs(args)
    if args.mode == 'report':
        run_report(args, output, pairs)
    elif args.mode == 'ablate':
        run_frozen(args, output, pairs)
    elif args.mode == 'train':
        run_training(args, output, pairs)
    elif args.mode == 'match':
        run_matching(args, output)
    elif args.mode == 'locate':
        from .localization import run_localization
        run_localization(args, output, pairs)
    elif args.mode == 'routes':
        from .routes import run_routes
        run_routes(args, output, pairs)
    else:
        from .head_audit import run_heads
        run_heads(args, output, pairs)
    print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
