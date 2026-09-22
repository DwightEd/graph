"""Evaluate short hallucination spans from frozen scores, without model inference."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .inputs import load_input, short_cohort
from .metrics import evaluate
from .reanchor import reanchor_rows
from .reporting import save_report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/head_cross_terms_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/short_span_audit_v1"))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--tasks", nargs="+", choices=("QA", "Summary", "Data2txt"))
    parser.add_argument("--fpr", nargs="+", type=float, default=[.01, .03, .05])
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--reanchor", type=Path, help="optional old reanchor_nodes_v2 directory")
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.threads < 1 or any(value < 0 or value > 1 for value in args.fpr):
        parser.error("bootstrap >= 0, threads >= 1, and FPR budgets in [0, 1] are required")
    return args


def main(argv=None):
    args = arguments(argv)
    with threadpool_limits(limits=args.threads), tqdm(total=3, desc="short span audit") as progress:
        blocks, methods, matching, settings, provenance = load_input(args)
        cohort = short_cohort(blocks)
        progress.update()
        report = evaluate(blocks, methods, args.fpr, args.bootstrap, args.seed)
        report.update(purpose="frozen_score_short_span_audit_not_new_detector",
                      natural_labels_used_for_fit=False, methods=methods, provenance=provenance,
                      statistical_status="exploratory_source_bootstrap_uncorrected",
                      normal_history_steps=15, bootstrap=args.bootstrap, seed=args.seed)
        reanchors = reanchor_rows(args.reanchor, cohort) if args.reanchor else []
        progress.update()
        archive = save_report(args.output, report, cohort, matching, settings, reanchors)
        progress.update()
    print(json.dumps(dict(answers=len(blocks), short_span_answers=len(cohort), methods=methods,
                          report=str(args.output / "summary.md"), archive=str(archive))), flush=True)
    return report


if __name__ == "__main__":
    main()
