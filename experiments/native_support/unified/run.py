"""CPU-only source/routing/finite-effect graph detector; freeze scores before labels."""

import argparse
import json
import math
from contextlib import closing
from pathlib import Path

from tqdm import tqdm
from state_audit.storage import read_json, start_stage, write_arrays, write_json

from ..choice_cache import CaptureReader
from ..evidence_contrast.data import copy_annotations
from ..evidence_contrast.run import pack
from ..evidence_contrast.unit_report import evaluate_units
from ..message_carriers.token_report import ranking_scope
from .calibration import fit_scales
from .data import baseline_methods, read_dataset, read_edges
from .scoring import NEW_METHODS, PRIMARY, score_record
from . import token_scoring


def measurement_protocol(args, input_protocol):
    protocol = dict(version="source-anchored-route-graph-v1", input=str(args.input.resolve()),
        input_version=input_protocol["version"], primary_candidate=PRIMARY,
        reference=str(args.reference.resolve()) if args.reference else None,
        calibration="external_source_disjoint" if args.reference else "unlabelled_transductive_cohort",
        scale="source_balanced_empirical_mid_CDF; token_route_and_unit_source_views",
        source_anchor="equal_local_and_selected_history_source_ranks",
        route="within_saved_unit_centered_raw_route_rank",
        graph="absolute_per_head_logp_source_interaction; one_nat_scale; degree_cap",
        graph_strength=args.graph_strength, gate_scope=input_protocol["gate_scope"],
        objective="min ||delta-centered_route||^2 + lambda*delta.T L delta; unit_mean(delta)=0",
        readout="source_anchor + graph_regularized_route_residual",
        sham="cached_matched_key_endpoints_with_identical_selected_weights; topology_control_only",
        labels_used_for_scoring=False, labels_used_for_training=False,
        fitted_quantities="unlabelled_empirical_scales_only", automatic_model_selection=False,
        probability_calibration=False, future_tokens_used=True, new_model_forwards=0,
        future_scope="complete_answer_graph_and_complete_unlabelled_calibration_reference",
        scope="offline_structured_score; smoothness_is_a_hypothesis_not_causal_truth_inheritance",
        default_legacy_risk="raw_route_preserved", token_covariance_model=False)
    if args.token_readout:
        protocol.update(version="token-source-soft-anchor-tv-v2", primary_candidate=token_scoring.PRIMARY,
            scale="source_balanced_empirical_mid_CDF; separate_token_and_unit_source_views",
            route="raw_route_rank_without_unit_centering", anchor_strength=args.anchor_strength,
            continuity_strength=args.continuity_strength,
            objective="0.5||z-y||^2 + alpha/2*sum_u n_u*(mean_u(z)-a_u)^2 "
                      "+ gamma/2*(z-s).T L (z-s) + tau*||D z||_1",
            readout="y=0.5*(token_source_rank_consensus+route_rank); soft_unit_mean; first_difference_TV",
            graph_scope="regularize_correction_to_token_source_observation; not_absolute_risk",
            continuity_scope="all_adjacent_answer_tokens; no_gold_or_saved_unit_boundary_reset",
            solver="sparse_ADMM_rho1_tol1e-9_max10000; exact_plateau_polish_KKT1e-7",
            legacy_hard_anchor_controls=True)
    return protocol


def active_methods(protocol):
    if protocol["version"] == "token-source-soft-anchor-tv-v2":
        return (*token_scoring.NEW_METHODS, *NEW_METHODS)
    return NEW_METHODS


def reference_records(path, records, input_protocol):
    if path is None:
        return records
    _, reference_protocol, reference = read_dataset(path)
    if reference_protocol["version"] != input_protocol["version"]:
        raise ValueError("Reference must use the same carrier measurement version")
    target_sources = {row["response"]["source_id"] for row in records}
    if target_sources & {row["response"]["source_id"] for row in reference}:
        raise ValueError("External calibration reference overlaps target sources")
    return reference


def freeze_scores(args):
    settings, input_protocol, records = read_dataset(args.input)
    reference = reference_records(args.reference, records, input_protocol)
    protocol = measurement_protocol(args, input_protocol)
    for name, value in (("protocol", protocol), ("settings", settings), ("input_protocol", input_protocol)):
        start_stage(args.output / f"{name}.json", value, args.resume)
    scales = fit_scales(reference, include_tokens=args.token_readout)
    for name, distribution in scales.items():
        write_arrays(args.output / "calibration" / f"{name}.npz", **distribution)
    write_json(args.output / "calibration.json", dict(scope=protocol["calibration"],
        sources=sorted({row["response"]["source_id"] for row in reference}),
        answers=len(reference), labels_used=False))
    diagnostics = []
    with closing(CaptureReader(args.input)) as reader:
        for record in tqdm(records, desc="unified source/routing graph"):
            edges = read_edges(reader, record, input_protocol)
            scores, components, row = score_record(record, edges, scales, args.graph_strength)
            if args.token_readout:
                row["token_readout"] = token_scoring.add_token_scores(record, edges, scores, components,
                    args.graph_strength, args.anchor_strength, args.continuity_strength)
            directory = args.output / record["directory"]
            write_json(directory / "views.json", record["views"])
            write_arrays(directory / "scores.npz", **scores)
            write_arrays(directory / "components.npz", **components)
            write_arrays(directory / "edges.npz", **edges)
            diagnostics.append(row)
    write_json(args.output / "diagnostics.json", diagnostics)
    write_json(args.output / "coverage.json", dict(status="complete", answers=len(records),
        represented_tokens=sum(len(row["scores"]["target"]) for row in records), dropped_tokens=0))
    return settings, protocol, input_protocol


def finish(args, settings, protocol, input_protocol):
    read_json(args.output / "coverage.json")
    copy_annotations(args.output, Path(protocol["input"]), args.annotations)
    base_tokens, base_methods = baseline_methods(input_protocol)
    candidates, primary = active_methods(protocol), protocol["primary_candidate"]
    tokens = [*candidates, *base_tokens]
    methods = [*candidates, *(name + "_unit_mean" for name in candidates), *base_methods]
    carrier = "source_selected" if input_protocol["version"] == "message-carriers-v1" else "token_source_selected"
    comparisons = [(primary, name) for name in (*candidates[1:], "raw_route", "raw_route_offline_mean",
                   "source_local_unit_mean", carrier + "_unit_mean")]
    result = evaluate_units(args.output, settings, methods, tokens, comparisons, offline_methods=candidates)
    write_json(args.output / "ranking_scope.json", ranking_scope(result))
    write_json(args.output / "summary.json", dict(protocol=protocol, evaluation=result,
        coverage=read_json(args.output / "coverage.json")))
    return dict(output=str(args.output), status=result["status"], primary_candidate=primary,
        review_archive=pack(args.output), input_measurement_version=input_protocol["version"],
        all_error={name: {key: row["all_error"][key] for key in ("auroc", "ap")}
                   for name, row in result.get("methods", {}).items()})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed carrier v1/v2 directory or review ZIP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional unlabelled source-disjoint carrier calibration cache")
    parser.add_argument("--stage", choices=("run", "score", "evaluate", "pack"), default="run")
    parser.add_argument("--graph-strength", type=float, default=1., help="Frozen regularization coefficient; no label tuning")
    parser.add_argument("--token-readout", action="store_true", help="Direct token sources, soft unit prior and TV continuity")
    parser.add_argument("--anchor-strength", type=float, default=1., help="Token readout: soft unit-mean penalty")
    parser.add_argument("--continuity-strength", type=float, default=.05, help="Token readout: first-difference TV in rank units")
    parser.add_argument("--annotations", type=Path, help="Evaluation only, after all scores are written")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("run", "score") and args.input is None:
        parser.error("--input is required to score cached observations")
    for name in ("graph_strength", "anchor_strength", "continuity_strength"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    if args.input and args.output.resolve() == args.input.resolve():
        parser.error("Use a separate output directory; input measurements are immutable")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps(dict(review_archive=pack(args.output))))
        return
    if args.stage in ("run", "score"):
        settings, protocol, input_protocol = freeze_scores(args)
    else:
        settings, protocol, input_protocol = [read_json(args.output / f"{name}.json")
                                             for name in ("settings", "protocol", "input_protocol")]
    print(json.dumps(finish(args, settings, protocol, input_protocol), ensure_ascii=False))


if __name__ == "__main__":
    main()
