"""Supervised source-held-out readout of immutable observable captures."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import sklearn
from threadpoolctl import threadpool_limits
from state_audit.storage import write_json

from ..choice_cache import CaptureReader
from .data import attach_choice_features, load_dataset
from .models import train_readouts
from .report import evaluate, pack


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Observable directory or light ZIP")
    parser.add_argument("--output", required=True, type=Path, help="New supervised result directory")
    parser.add_argument("--reference", type=Path, help="Separate labelled training capture; sources must be disjoint")
    parser.add_argument("--choice-input", type=Path, help="Optional choice_state_v2 directory/ZIP with matching tokens")
    parser.add_argument("--reference-choice", type=Path, help="Matching choice capture for the training reference")
    parser.add_argument("--features", choices=("summary", "heads"), default="summary")
    parser.add_argument("--layers", choices=("middle", "all"), default="middle")
    parser.add_argument("--models", nargs="+", choices=("logistic", "trees"), default=["logistic", "trees"])
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.window < 1 or args.cpu_threads < 1:
        parser.error("window and cpu-threads must be positive")
    if args.reference_choice and not (args.reference and args.choice_input):
        parser.error("--reference-choice requires --reference and --choice-input")
    if args.reference and args.choice_input and not args.reference_choice:
        parser.error("Both evaluation and reference require matching choice features")
    return args


def main(argv=None):
    args = arguments(argv)
    # A separate new directory makes old scores/captures immutable by construction.
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=args.cpu_threads):
        dataset = load_dataset(args.input, args.features, args.layers, args.window)
        reference = None if args.reference is None else load_dataset(args.reference, args.features, args.layers, args.window)
        if args.choice_input:
            attach_choice_features(dataset, args.choice_input, args.window)
        if args.reference_choice:
            attach_choice_features(reference, args.reference_choice, args.window)
        predictions, folds = train_readouts(dataset, args.output, args.models, reference)
    protocol = dict(purpose="supervised_representation_readout_diagnostic", labels_used_for_training=True,
                    labels_used_for_feature_construction=False, model_forward=False,
                    split="external_source_disjoint" if reference else "leave_one_source_out",
                    feature_mode=args.features, layer_scope=args.layers, window=args.window,
                    feature_dimensions={name: len(columns) for name, columns in dataset["schema"].items()},
                    models=args.models, input=str(args.input), reference=str(args.reference) if reference else None,
                    sklearn_version=sklearn.__version__, seed=37, automatic_model_selection=False,
                    hyperparameter_tuning=False, early_stopping=False,
                    training_source_weighting="equal_before_class_reweighting",
                    choice_input=str(args.choice_input) if args.choice_input else None,
                    reference_choice=str(args.reference_choice) if args.reference_choice else None,
                    class_weight="balanced", future_tokens_used=True, threshold_calibrated=False,
                    excluded_features=["transport_conditional", "transport_context", "answer/source IDs", "gold boundaries"],
                    unavailable_semantics=["evidence_applicability", "factual_choice", "causal_fact_inheritance"],
                    responses=len(dataset["records"]), sources=len({r["source_id"] for r in dataset["records"]}),
                    labelled_tokens=sum(int(r["valid"].sum()) for r in dataset["records"]),
                    positive_tokens=sum(int(r["labels"][r["valid"]].sum()) for r in dataset["records"]))
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "settings.json", dataset["settings"])
    with closing(CaptureReader(args.input)) as reader:
        write_json(args.output / "annotations.json", reader.json("annotations.json"))
    result = evaluate(args.output, dataset, predictions, folds, protocol)
    print(json.dumps(dict(output=str(args.output), protocol=protocol, review_archive=pack(args.output),
                         all_error={name: phases["all_error"] for name, phases in result["common_tokens"].items()})))
