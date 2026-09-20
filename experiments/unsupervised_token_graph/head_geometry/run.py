"""One-command, label-free physical-head geometry detection and frozen evaluation."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ..offline_span.data import write_json
from .inputs import SIGNALS, inspect, prepare, special_ids
from .pipeline import fit, read_json, score


ROOT = Path("/share/home/tm902089733300000/a903202310/lys/data/RAGTruth")
TOKENIZER = ROOT.parent.parent / "models/Meta-Llama-3.1-8B-Instruct"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "inspect", "prepare", "fit", "score", "evaluate"], default="all")
    parser.add_argument("--train-cache", type=Path, default=ROOT / "attention/llama31_8b/train")
    parser.add_argument("--test-cache", type=Path, default=ROOT / "attention/llama31_8b/test")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--tokenizer", default=str(TOKENIZER))
    parser.add_argument("--special-token-ids", nargs="+", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/head_geometry_v1"))
    parser.add_argument("--tasks", nargs="+", default=["QA"])
    parser.add_argument("--generators", nargs="+", default=["all"])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=["self"])
    parser.add_argument("--layer-bands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=.1)
    parser.add_argument("--dimensions", type=int, default=256,
                        help="relation sketch width; 0 keeps every head-pair entry; current heads are always exact")
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--tokens-per-source", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def record_settings(args, excluded):
    current = {key: str(value.resolve()) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    for key in ("phase", "resume", "bootstrap", "threads"):
        current.pop(key)
    current.update(version="head-geometry-v1", excluded_token_ids=excluded,
                   natural_labels_used_for_fit=False)
    path = args.output / "settings.json"
    if path.exists():
        if not args.resume or read_json(path) != current:
            raise ValueError("Use a new output or --resume with identical geometry settings")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(path, current)


def main(argv=None):
    args = arguments(argv)
    args.signal_indices = [SIGNALS.index(name) for name in args.signals]
    budgets = (args.window, args.bank_size, args.tokens_per_source,
               args.neighbors, args.threads)
    if min(budgets) < 1 or args.ridge <= 0 or not 0 < args.quantile < 1 or args.limit < 0 or args.dimensions < 0:
        raise ValueError("Budgets/ridge must be positive; quantile must lie in (0,1)")
    with threadpool_limits(limits=args.threads):
        if args.phase == "evaluate":
            from .evaluation import evaluate
            return evaluate(args)
        excluded = special_ids(args)
        if args.phase == "inspect":
            return inspect(args, excluded)
        record_settings(args, excluded)
        if args.phase in ("all", "prepare"):
            prepare(args, excluded)
        if args.phase in ("all", "fit"):
            if not (args.resume and (args.output / "reference/complete.json").exists()):
                fit(args)
        if args.phase in ("all", "score"):
            score(args)
        if args.phase == "all":
            from .evaluation import evaluate
            return evaluate(args)
