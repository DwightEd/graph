"""Explicit prepare / fit / audit / report stages for CHARM attribution."""

import argparse
import json
from pathlib import Path

import torch

from .data import partitions, prepare, write_json
from .experiment import audit, compare_models, fit, load_predictions
from .graph import VARIANTS
from .metrics import report


BASE = '/share/home/tm902089733300000/a903202310/lys'


def read_records(prepared, task, split=None):
    records = json.loads((Path(prepared) / 'index.json').read_text())
    return [r for r in records if r['task'] == task and (split is None or r['split'] == split)]


def print_report(value):
    for group, row in value['groups'].items():
        metrics = row['token_scopes']
        first = metrics['first_error_vs_normal']
        print(json.dumps(dict(group=group, auroc=metrics['all_error']['auroc'],
                              ap=metrics['all_error']['ap'], first_error_recall=first['recall'],
                              continuation_auroc=metrics['continuation_vs_normal']['auroc'],
                              first_span=row['boundaries']['first_error_spans']), ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--attention-root', default=BASE + '/data/RAGTruth/attention/llama31_8b')
    p.add_argument('--annotations', default=BASE + '/data/RAGTruth/dataset/response.jsonl')
    p.add_argument('--tokenizer', default=BASE + '/models/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--output', required=True)
    p.add_argument('--tau', type=float, default=.05)
    p.add_argument('--tasks', nargs='+', default=['QA', 'Summary', 'Data2txt'])
    p.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    p.add_argument('--generator', default='llama-2-7b-chat')
    p.add_argument('--limit', type=int, default=0)
    for command in ('fit', 'audit'):
        p = sub.add_parser(command)
        p.add_argument('--prepared', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--tasks', nargs='+', default=['QA', 'Summary', 'Data2txt'])
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--edge-chunk', type=int, default=4096)
        p.add_argument('--bootstrap', type=int, default=200)
        p.add_argument('--threads', type=int, default=1)
        p.add_argument('--prefix-sites', type=int, default=8)
        p.add_argument('--no-perturbations', action='store_true')
        if command == 'fit':
            p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
            p.add_argument('--seeds', type=int, nargs='+', default=[0])
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--patience', type=int, default=5)
            p.add_argument('--hidden-dim', type=int, default=128)
            p.add_argument('--layers', type=int, default=3)
            p.add_argument('--batch-size', type=int, default=32)
            p.add_argument('--fpr', type=float, default=.05)
        else:
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--variant', choices=VARIANTS, default='charm_out')
            p.add_argument('--threshold', type=float, default=.5,
                           help='predeclared threshold; never optimized on test labels')
            p.add_argument('--split', choices=['train', 'test'], default='test')
    p = sub.add_parser('report')
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--completed-only', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        rows = prepare(args.attention_root, args.annotations, args.output, args.tokenizer,
                       args.tau, args.tasks, args.generator, args.limit, args.splits)
        print(json.dumps(dict(prepared=len(rows), scope='pilot' if args.limit else 'selected_cohort')))
        return
    if args.command == 'report':
        samples = load_predictions(args.predictions, args.completed_only)
        protocol = json.loads((Path(args.predictions) / 'prediction_settings.json').read_text())
        protocol['evaluation_scope'] = 'completed_preview' if args.completed_only else 'complete_prediction_set'
        print_report(report(samples, args.output, protocol['threshold'], protocol, args.bootstrap))
        return
    torch.set_num_threads(args.threads)
    root = Path(args.output)
    for task in args.tasks:
        records = read_records(args.prepared, task)
        if not records:
            raise ValueError('no prepared records for task=' + task)
        if args.command == 'audit':
            cohort = [r for r in records if r['split'] == args.split]
            if not cohort:
                raise ValueError('no records in requested split for ' + task)
            threshold = dict(value=args.threshold, rule='score > threshold', origin='predeclared_external_checkpoint_threshold')
            out = root / task / 'external_checkpoint'
            samples = audit(args.checkpoint, cohort, out, args.variant, device=args.device,
                            edge_chunk=args.edge_chunk, threshold=threshold,
                            perturbations=not args.no_perturbations, prefix_sites=args.prefix_sites)
            protocol = json.loads((out / 'prediction_settings.json').read_text())
            protocol.update(experiment='frozen_existing_checkpoint',
                            compatibility='strict state_dict and channel dimensions; original observer/cohort still must match')
            print_report(report(samples, out, threshold, protocol, args.bootstrap))
            continue
        parts = partitions(records)
        write_json(root / task / 'partitions.json', {k: [r['id'] for r in v] for k, v in parts.items()})
        for seed in args.seeds:
            predictions = {}
            for variant in args.variants:
                out = root / task / f'seed_{seed}' / variant
                checkpoint = fit(parts, out, variant, seed, args.epochs, args.patience,
                                 args.hidden_dim, args.layers, args.batch_size, device=args.device,
                                 edge_chunk=args.edge_chunk, fpr=args.fpr)
                threshold = json.loads((out / 'threshold.json').read_text())
                pred = out / 'test'
                samples = audit(checkpoint, parts['test'], pred, variant, seed, args.device,
                                args.edge_chunk, threshold, not args.no_perturbations, args.prefix_sites)
                protocol = dict(experiment='independently_retrained_ablation', variant=variant, seed=seed,
                                checkpoint=str(checkpoint), supervised=True, selection='heldout_source_selection_AP',
                                calibration='separate_heldout_sources', alignment='post_token_i_to_label_i')
                print_report(report(samples, pred, threshold, protocol, args.bootstrap))
                if variant == 'node_only':
                    smooth = [dict(s, score=s['score_ewma']) for s in samples]
                    report(smooth, pred / 'ewma', threshold['ewma'], dict(protocol, baseline='fixed_EWMA_beta_0.5'), 0)
                predictions[variant] = pred
            compare_models(predictions, root / task / f'seed_{seed}' / 'comparison.json', args.bootstrap)


if __name__ == '__main__':
    main()
