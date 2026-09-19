"""RAGTruth evidence-flow audit: full cache screen, held-out functional confirmation."""

import argparse
from pathlib import Path

from .screen import screen_dataset
from .confirm import confirm_dataset
from .report import report


ROOT = Path("/share/home/tm902089733300000/a903202310/lys/data/RAGTruth")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["screen", "confirm", "report", "grounding", "all"],
        default="all",
    )
    parser.add_argument("--train-cache", type=Path, default=ROOT / "attention/llama31_8b/train")
    parser.add_argument("--test-cache", type=Path, default=ROOT / "attention/llama31_8b/test")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--tokenizer", default="/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model", default="/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/ragtruth_evidence_flow_v1"))
    parser.add_argument("--tasks", nargs="+", choices=["QA", "Summary", "Data2txt"], default=["QA", "Summary", "Data2txt"])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--position-gap", type=float, default=.25)
    parser.add_argument("--repetition-gap", type=float, default=.15)
    parser.add_argument("--entropy-gap", type=float, default=.5)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--screen-heads", type=int, default=12)
    parser.add_argument("--confirm-heads", type=int, default=4)
    parser.add_argument("--confirm-pairs", type=int, default=8)
    parser.add_argument("--grounding-ridge", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("screen", "all"):
        screen_dataset(args)
    if args.phase in ("confirm", "all"):
        confirm_dataset(args)
    if args.phase in ("grounding", "all"):
        from .grounding_dynamics import run_grounding_dynamics
        run_grounding_dynamics(args)
    if args.phase in ("report", "all"):
        report(args.output)


if __name__ == "__main__":
    main()
