"""固定图、无标签参照、全部对照一次运行；没有神经网络训练。"""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ..offline_span.data import write_json
from .pipeline import fit, inspect, prepare, read_json, score


ROOT = Path('/share/home/tm902089733300000/a903202310/lys/data/RAGTruth')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'inspect', 'prepare', 'fit', 'score', 'evaluate'], default='all')
    parser.add_argument('--train-cache', type=Path, default=ROOT / 'attention/llama31_8b/train')
    parser.add_argument('--test-cache', type=Path, default=ROOT / 'attention/llama31_8b/test')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--source-info', type=Path)
    parser.add_argument('--tokenizer', help='original observer tokenizer; used only for evaluation alignment')
    parser.add_argument('--output', type=Path, default=Path('outputs/fixed_graph_v1'))
    parser.add_argument('--tasks', nargs='+', default=['all'])
    parser.add_argument('--generators', nargs='+', default=['all'])
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--heads', nargs='+', type=int)
    parser.add_argument('--local-window', type=int, default=16)
    parser.add_argument('--dimensions', type=int, default=128)
    parser.add_argument('--bank-size', type=int, default=4096)
    parser.add_argument('--tokens-per-source', type=int, default=16)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--limit', type=int, default=0, help='execution-order pilot per split, not the full benchmark')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def record_settings(args):
    current = {key: str(value.resolve()) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    for key in ('phase', 'resume', 'tokenizer', 'bootstrap', 'threads'):
        current.pop(key)
    current['version'] = 'fixed-relational-graph-v1'
    path = args.output / 'settings.json'
    if path.exists():
        if not args.resume or read_json(path) != current:
            raise ValueError('Use a new output or --resume with identical graph/reference settings')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(path, current)


def main(argv=None):
    args = arguments(argv)
    positive = (args.local_window, args.dimensions, args.bank_size, args.tokens_per_source, args.neighbors, args.threads)
    if min(positive) < 1 or args.limit < 0 or not 0 < args.quantile < 1:
        raise ValueError('Budgets must be positive; quantile must lie in (0,1)')
    with threadpool_limits(limits=args.threads):
        if args.phase == 'inspect':
            inspect(args)
            return
        if args.phase != 'evaluate':
            record_settings(args)
        if args.phase in ('all', 'prepare'):
            prepare(args)
        if args.phase in ('all', 'fit'):
            complete = args.output / 'reference/complete.json'
            if not (args.resume and complete.is_file()):
                fit(args)
        if args.phase in ('all', 'score'):
            score(args)
        if args.phase in ('all', 'evaluate'):
            from .evaluation import evaluate
            evaluate(args)


if __name__ == '__main__':
    main()
