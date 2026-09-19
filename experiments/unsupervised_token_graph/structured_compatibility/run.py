"""One self-supervised compatibility experiment; labels are evaluation-only."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ..offline_span.data import write_json
from .pipeline import fit, inspect, score


ROOT = Path("/share/home/tm902089733300000/a903202310/lys/data/RAGTruth")
TOKENIZER = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/"
    "Meta-Llama-3.1-8B-Instruct"
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "inspect", "fit", "score", "evaluate"],
        default="all",
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=ROOT / "attention/llama31_8b/train",
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=ROOT / "attention/llama31_8b/test",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--tokenizer", default=str(TOKENIZER))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/structured_compatibility_v1"),
    )
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--generators", nargs="+", default=["all"])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--train-budget", type=int, default=2048)
    parser.add_argument("--tokens-per-source", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def saved_settings(args):
    values = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    for key in ("phase", "resume", "bootstrap", "threads"):
        values.pop(key)
    values["version"] = "structured-compatibility-v1"
    values["natural_labels_used_for_fit"] = False
    return values


def record_settings(args):
    path = args.output / "settings.json"
    current = saved_settings(args)

    if path.exists():
        saved = json.loads(path.read_text())
        if not args.resume or saved != current:
            raise ValueError(
                "Use a new output, or --resume with identical compatibility settings"
            )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(path, current)


def main(argv=None):
    args = arguments(argv)
    with threadpool_limits(limits=args.threads):
        if args.phase == "inspect":
            inspect(args)
            return

        if args.phase != "evaluate":
            record_settings(args)

        if args.phase in ("all", "fit"):
            complete = args.output / "fit/complete.json"
            if not (args.resume and complete.exists()):
                fit(args)

        if args.phase in ("all", "score"):
            score(args)

        if args.phase in ("all", "evaluate"):
            from .evaluation import evaluate
            evaluate(args)


if __name__ == "__main__":
    main()
