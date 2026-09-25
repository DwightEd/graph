"""CPU refinement of full source-first caches; all test tasks, train-only selection."""

import argparse
import json
from contextlib import closing
from pathlib import Path

from state_audit.storage import read_json, start_stage

from ..choice_cache import CaptureReader
from ..ragtruth_benchmark.data import TASKS
from ..ragtruth_benchmark.report import pack
from .data import score_all, select_records
from .scoring import ANCHORS, CHANNELS, PRIMARY, SELECTION_POOL, WEIGHTS


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed source-first directory or exported TASK_cache.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", choices=TASKS, nargs="+")
    parser.add_argument("--window", type=int, help="Offline token window; default 16, frozen before selection")
    parser.add_argument("--select-on-train", action="store_true", default=None)
    parser.add_argument("--stage", choices=("run", "score", "select", "evaluate", "pack", "export"), default="run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    saved_path = args.output / "protocol.json"
    saved = read_json(saved_path) if saved_path.exists() else {}
    args.input = args.input or (Path(saved["input"]) if saved else None)
    args.tasks = args.tasks or saved.get("tasks", list(TASKS))
    args.window = args.window if args.window is not None else saved.get("window", 16)
    if args.select_on_train is None:
        args.select_on_train = saved.get("select_on_train", False)
    if args.input is None:
        parser.error("A new output requires --input with completed scalar measurements")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a new output directory; source measurements are immutable")
    if args.window < 1:
        parser.error("window must be positive")
    if args.stage == "select" and not args.select_on_train:
        parser.error("Selection requires --select-on-train from the initial scoring stage")
    return args


def protocol(args):
    return dict(version="ragtruth-token-refine-v2", input=str(args.input.resolve()),
        tasks=args.tasks, window=args.window, select_on_train=args.select_on_train,
        fixed_candidate=PRIMARY, anchors=ANCHORS, channels=CHANNELS, residual_weights=WEIGHTS,
        selection_pool=SELECTION_POOL, labels_used_for_fixed_scores=False, classifier_training=False,
        calibration="source_balanced_midCDF_separately_on_train_development_and_test; transductive",
        tie_readout="integer_anchor_level + .25 * within_unit_observation_rank_residual",
        residual_readout="anchor_midCDF + lambda * within_unit_observation_rank_residual",
        future_tokens_used=True, score_scope="offline_answer_and_selected_task_split_cohort",
        test_status="prior_test_results_informed_design; exploratory_re_evaluation",
        native_model_forwards=0, original_results_immutable=True)


def prepare(args, reader):
    original = reader.json("manifest.json")
    if reader.json("coverage.json")["status"] != "complete":
        raise ValueError("Input source-first capture/scoring must be complete")
    records, development = select_records(original, args.tasks, args.select_on_train)
    first = records[0]["directory"]
    if not reader.exists(first + "/observations.npz"):
        raise ValueError("Input contains summaries only. Use the original source-first directory or TASK_cache.zip; no model rerun is needed.")
    previous = original["previous_selected"] if original.get("portable_refinement_cache") else reader.exists("selection.json")
    manifest = dict(model=original["model"], records=records, development_sources=development,
        previous_selected=previous, selected_answers=len(records),
        original_scope=original.get("original_scope", dict(answers=original["selected_answers"],
            cohort=original.get("original_cohort"), official_population=original.get("official_population"))))
    resume = args.resume or args.stage not in ("run", "score")
    start_stage(args.output / "protocol.json", protocol(args), resume)
    start_stage(args.output / "manifest.json", manifest, resume)
    return manifest


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps(dict(review_archive=pack(args.output))))
        return
    from .packing import export_caches
    from .report import evaluate
    from .selection import select
    with closing(CaptureReader(args.input)) as reader:
        manifest = prepare(args, reader)
        if args.stage in ("run", "score"):
            coverage = score_all(args, reader, manifest)
            if args.stage == "score":
                print(json.dumps(coverage))
                return
        if args.stage == "select" or (args.stage == "run" and args.select_on_train):
            selection = select(args, reader, manifest)
            if args.stage == "select":
                print(json.dumps(selection))
                return
        if args.stage == "export":
            print(json.dumps(dict(measurement_archives=export_caches(args, reader, manifest))))
            return
        summary = evaluate(args, reader, manifest)
        archives = export_caches(args, reader, manifest)
        print(json.dumps(dict(output=str(args.output), status=summary["status"],
            primary_candidate=summary["primary_candidate"], test_by_dataset=summary["test_by_dataset"],
            review_archive=pack(args.output), measurement_archives=archives), ensure_ascii=False))


if __name__ == "__main__":
    main()
