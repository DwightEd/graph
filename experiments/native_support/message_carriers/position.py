"""Frozen exponential output-position control on existing scores; no model or label fitting."""

from contextlib import closing

from state_audit.storage import start_stage, write_arrays, write_json, read_json

from ..choice_cache import CaptureReader
from ..dual_state.report import attach_annotations
from ..evidence_contrast.data import copy_annotations
from ..evidence_contrast.run import pack
from ..evidence_contrast.unit_report import evaluate_units, load_records, unit_rows, unit_metrics
from .representation import TOKEN_METHODS, METHODS
from .token_representation import aggregate_units
from .token_report import method_names, ranking_scope


def analyze_positions(args):
    with closing(CaptureReader(args.input)) as reader:
        settings, original_protocol = reader.json("settings.json"), reader.json("protocol.json")
        if original_protocol["version"] == "message-carriers-token-v2":
            tokens, original_methods, _ = method_names(original_protocol)
        else:
            tokens, original_methods = list(TOKEN_METHODS), list(METHODS)
        methods = [*original_methods, *(f"{name}_unit_exp" for name in tokens)]
        protocol = dict(version="carrier-output-position-v1", input=str(args.input.resolve()),
            beta=args.position_beta, weighting="exp(beta * normalized_position_in_saved_text_unit)",
            key_distance_weighting=False, attention_modified=False, labels_used_for_scoring=False,
            parameter_fitting=False, automatic_model_selection=False, future_tokens_used=True,
            scope="post_hoc_position_control; not_sink_removal_or_semantic_disentanglement")
        start_stage(args.output / "protocol.json", protocol, args.resume)
        start_stage(args.output / "settings.json", settings, args.resume)
        start_stage(args.output / "input_protocol.json", original_protocol, args.resume)
        for index, _ in enumerate(settings["responses"]):
            directory = f"responses/{index:04d}"
            views, saved = reader.json(directory + "/views.json"), reader.arrays(directory + "/scores.npz")
            for name in tokens:
                saved[f"{name}_unit_exp"] = aggregate_units(saved[name], views["units"], args.position_beta)
            write_arrays(args.output / directory / "scores.npz", **saved)
            write_json(args.output / directory / "views.json", views)
    copy_annotations(args.output, args.input, args.annotations)
    return finish_positions(args, settings, protocol, tokens, methods)


def finish_positions(args, settings, protocol, tokens, methods):
    comparisons = [(name + "_unit_exp", name + "_unit_mean") for name in tokens]
    result = evaluate_units(args.output, settings, methods, tokens, comparisons)
    write_json(args.output / "ranking_scope.json", ranking_scope(result))
    if result["status"] == "evaluated":
        records, predictions = load_records(args.output, settings, methods)
        attach_annotations(records, read_json(args.output / "annotations.json"))
        rows = unit_rows(records, predictions, tokens, suffix="_unit_exp")
        write_json(args.output / "unit_exp_evaluation.json", unit_metrics(rows, tokens))
    write_json(args.output / "summary.json", dict(protocol=protocol, evaluation=result))
    return dict(output=str(args.output), status=result["status"], review_archive=pack(args.output),
        all_error={name: {key: phases["all_error"][key] for key in ("auroc", "ap")}
                   for name, phases in result.get("methods", {}).items() if name.endswith(("_unit_exp", "_unit_mean"))})
