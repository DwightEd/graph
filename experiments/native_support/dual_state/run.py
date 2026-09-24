"""Current plus persistent direction, with source-disjoint unlabelled head calibration."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm
from state_audit.storage import write_arrays, write_json

from ..choice_cache import CaptureReader
from .data import load_observations, validate_reference
from .report import attach_annotations, evaluate
from .scoring import fit_reference, head_scores, reference_records, scalar_scores


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features", choices=("heads", "scalar"), default="heads")
    parser.add_argument("--layers", choices=("middle", "all"), default="middle")
    parser.add_argument("--window", type=int, default=16, help="Total window width, including current token")
    parser.add_argument("--reference", type=Path, help="Optional disjoint, unlabelled full observable capture")
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("window must be positive")
    if args.reference and args.features != "heads":
        parser.error("--reference applies only to head calibration")
    return args


def save_reference(destination, source, fitted, selected):
    arrays, bins = {}, {}
    for position, cell in fitted.items():
        arrays[f"bin_{position}_values"] = cell["values"]
        arrays[f"bin_{position}_cdf"] = cell["cumulative"]
        bins[str(position)] = dict(tokens=cell["tokens"], sources=cell["sources"])
    write_arrays(destination / "reference.npz", **arrays)
    detail = dict(evaluation_source=source, reference_sources=sorted({r["source_id"] for r in selected}),
                  weighting="equal_source_per_position_bin", bins=bins)
    write_json(destination / "reference.json", detail)
    return detail


def score_all(destination, dataset, reference, window):
    records, predictions, folds = dataset["records"], [], {}
    fitted_by_source = {}
    pool = reference["records"] if reference else records
    for index, record in enumerate(tqdm(records, desc="dual scores")):
        directory = destination / "responses" / f"{index:04d}"
        scores = scalar_scores(record, window)
        source = record["source_id"]
        if dataset["schema"]:
            if source not in fitted_by_source:
                selected = reference_records(pool, source)
                fitted = fit_reference(selected)
                fitted_by_source[source] = fitted
                folds[source] = save_reference(destination / "references" / f"{len(folds):04d}", source, fitted, selected)
            head, channels = head_scores(record, fitted_by_source[source], window)
            scores.update(head)
            write_arrays(directory / "head_state.npz", route=record["head_route"],
                         response_energy=record["response_energy"], **channels)
        tokens = record["response"]["token_ids"][record["response"]["prompt_length"]:]
        write_arrays(directory / "scores.npz", target=record["target"], token_id=np.asarray(tokens),
                     **record["baselines"], **scores)
        predictions.append(scores)
    write_json(destination / "references.json", list(folds.values()))
    return predictions


def protocol_for(args, dataset):
    return dict(version="dual-state-v1", purpose="unsupervised_directional_current_and_persistent_readout",
        labels_used_for_training=False, labels_used_for_scoring=False, model_forward=False,
        feature_mode=args.features, layer_scope=args.layers, window=args.window,
        causal_past=args.window-1, offline_past=(args.window-1)//2, offline_future=args.window//2,
        primary_candidate="head_causal_dual" if dataset["schema"] else "observable_route_causal_dual",
        primary_is_frozen_not_selected=True, combine="maximum_current_and_persistent",
        position_bins="min(floor(log2(target+1)),7)", reference=str(args.reference) if args.reference else None,
        reference_mode="external_source_disjoint" if args.reference else "leave_one_source_out" if dataset["schema"] else "none",
        input=str(args.input), future_tokens_used=True,
        future_scope="offline_controls; other_complete_reference_answers_in_cohort_head_mode",
        target_future_used_by_causal_scores=False, threshold_calibrated=False,
        automatic_model_selection=False, hyperparameter_tuning=False,
        labels_previously_seen_for_design=True, independent_confirmatory_test=False,
        unavailable_semantics=["evidence_applicability", "factual_choice", "causal_fact_inheritance"],
        normal_control="unmatched_contiguous_annotation_negative_runs",
        responses=len(dataset["records"]), sources=len({r["source_id"] for r in dataset["records"]}))


def pack(destination):
    archive = destination.with_name(destination.name + "_review_light.zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name not in ("head_state.npz", "reference.npz"):
                output.write(path, path.relative_to(destination))
    return str(archive)


def main(argv=None):
    args = arguments(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = load_observations(args.input, args.features, args.layers)
    reference = None
    if args.reference:
        reference = load_observations(args.reference, args.features, args.layers)
        validate_reference(dataset, reference)
    protocol = protocol_for(args, dataset)
    write_json(args.output / "settings.json", dataset["settings"])
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "feature_schema.json", dataset["schema"])
    predictions = score_all(args.output, dataset, reference, args.window)
    # Every score is now frozen on disk. Only this boundary may open annotations.
    with closing(CaptureReader(args.input)) as reader:
        if not reader.exists("annotations.json"):
            write_json(args.output / "evaluation.json", dict(status="unavailable_missing_annotations"))
            print(json.dumps(dict(output=str(args.output), review_archive=pack(args.output))))
            return
        annotations = reader.json("annotations.json")
    write_json(args.output / "annotations.json", annotations)
    attach_annotations(dataset["records"], annotations)
    result = evaluate(args.output, dataset, predictions, protocol)
    print(json.dumps(dict(output=str(args.output), review_archive=pack(args.output),
                         all_error={name: phases["all_error"] for name, phases in result["methods"].items()})))
