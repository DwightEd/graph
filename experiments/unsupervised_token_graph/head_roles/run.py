"""Paper-inspired head profiling and label-free denoising ablations."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ..head_geometry.inputs import special_ids
from ..head_geometry.pipeline import read_json
from ..head_geometry.run import ROOT, TOKENIZER
from ..offline_span.data import write_json
from .inputs import FEATURES, prepare
from .pipeline import fit, score
from .profile import profile


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "prepare", "profile", "fit", "score", "evaluate", "binding", "diagnostic"], default="all")
    parser.add_argument("--train-cache", type=Path, default=ROOT / "attention/llama31_8b/train")
    parser.add_argument("--test-cache", type=Path, default=ROOT / "attention/llama31_8b/test")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--model", default=str(TOKENIZER))
    parser.add_argument("--tokenizer", default=str(TOKENIZER))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--special-token-ids", nargs="+", type=int)
    parser.add_argument("--tasks", nargs="+", default=["QA"])
    parser.add_argument("--generators", nargs="+", default=["all"])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--output", type=Path, default=Path("outputs/head_roles_v1"))
    parser.add_argument("--probe-sources", type=int, default=8)
    parser.add_argument("--probe-queries", type=int, default=4)
    parser.add_argument("--block-width", type=int, default=8)
    parser.add_argument("--swaps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--drop-fraction", type=float, default=.25)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--reconstruction-atol", type=float, default=.02)
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--bank-size", type=int, default=2048)
    parser.add_argument("--tokens-per-source", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--binding-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-lda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def settings(args, excluded):
    values = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    for key in ("phase", "resume", "bootstrap", "threads", "binding_check", "diagnostic_lda"):
        values.pop(key)
    values.update(version="head-roles-v1", window=1, special_ids=excluded, labels_used_for_fit=False)
    path = args.output / "settings.json"
    if path.exists():
        if not args.resume or read_json(path) != values:
            raise ValueError("Use new output or --resume with identical head-role settings")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(path, values)


def main(argv=None):
    args = arguments(argv)
    budgets = (args.probe_sources, args.probe_queries, args.block_width, args.swaps,
               args.random_controls, args.bank_size, args.tokens_per_source, args.neighbors, args.threads)
    if min(budgets) < 1 or args.temperature <= 0 or not 0 < args.drop_fraction < 1 or not 0 < args.quantile < 1:
        raise ValueError("Positive budgets/temperature and fractions in (0,1) required")
    with threadpool_limits(limits=args.threads):
        if args.phase == "evaluate":
            from .evaluation import evaluate
            return evaluate(args)
        if args.phase == "diagnostic":
            from .diagnostic import diagnostic
            return diagnostic(args)
        excluded = special_ids(args)
        settings(args, excluded)
        if args.phase == "binding":
            from .binding import run_binding
            return run_binding(args, excluded)
        if args.phase in ("all", "prepare"):
            prepare(args, excluded)
        if args.phase in ("all", "profile"):
            profile(args, excluded)
        if args.phase in ("all", "fit"):
            fit(args)
        if args.phase in ("all", "score"):
            score(args)
        if args.phase == "all":
            from .evaluation import evaluate
            report = evaluate(args)
            if args.diagnostic_lda:
                from .diagnostic import diagnostic
                diagnostic(args)
            if args.binding_check:
                from .binding import run_binding
                run_binding(args, excluded)
            return report
