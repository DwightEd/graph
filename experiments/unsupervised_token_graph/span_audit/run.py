"""标签辅助机制比较。没有fit/train模式，不加载LLM权重。"""

import argparse
import json
from pathlib import Path

from .experiment import analyze_dataset
from .inputs import AuditInputs
from .report import summarize, write_json


ROOT = Path('/share/home/tm902089733300000/a903202310/lys/data/RAGTruth')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['analyze', 'summarize'], default='analyze')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--tokenizer', help='original local observer tokenizer, not model weights')
    parser.add_argument('--feature-root', type=Path, help='optional existing entropy NPZs with matching token_ids')
    parser.add_argument('--output', type=Path, default=Path('outputs/span_mechanism_train_v1'))
    parser.add_argument('--tasks', nargs='+', choices=['QA', 'Summary', 'Data2txt'])
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--heads', nargs='+', type=int)
    parser.add_argument('--window', type=int, default=8, help='boundary observation radius, NOT candidate span length')
    parser.add_argument('--position-gap', type=float, default=.25)
    parser.add_argument('--repetition-gap', type=float, default=.15)
    parser.add_argument('--entropy-gap', type=float, default=.5)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--limit', type=int, default=0, help='explicit execution-order pilot; 0 uses every answer')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def prepare_output(args):
    settings = vars(args).copy()
    for name, value in settings.items():
        if isinstance(value, Path):
            settings[name] = str(value.resolve())
    for name in ('phase', 'resume', 'output'):
        settings.pop(name)
    settings['version'] = 'labelled-span-mechanism-v1'
    annotation = args.dataset / 'response.jsonl'
    settings['annotation_size'] = annotation.stat().st_size
    settings['annotation_modified_ns'] = annotation.stat().st_mtime_ns

    path = args.output / 'settings.json'
    if args.output.exists():
        if not args.resume or json.loads(path.read_text()) != settings:
            raise ValueError('Use a new output or resume the same experiment settings')
    else:
        (args.output / 'samples').mkdir(parents=True)
        write_json(path, settings)
    return settings


def main(argv=None):
    args = arguments(argv)
    if args.phase == 'summarize':
        print(json.dumps(summarize(args.output, args.bootstrap), ensure_ascii=False))
        return
    if args.window < 1 or min(args.limit, args.bootstrap) < 0:
        raise ValueError('window must be positive; limit and bootstrap cannot be negative')
    args.cache = args.cache or ROOT / 'attention/llama31_8b' / args.split
    settings = prepare_output(args)
    inputs = AuditInputs(args.cache, args.dataset, args.index, args.tokenizer, args.feature_root)
    result = analyze_dataset(inputs, args.output, settings, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
