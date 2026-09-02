"""One foreground entry for capture, label-free fitting, and evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .capture import RouteMessageReplay
from .controls import dense_endpoint_rewire, dense_weight_shuffle
from .data import TASK_TYPES, RouteSample, iter_route_samples
from .detector import StickyRouteHMM
from .evaluate import evaluate_scores, freeze_scores
from .graph import ResponseGraphBuilder, sparsify_route_chunk
from .lineage import LineageTracker
from .state import (
    EquationLockedRouteCollapseControl,
    build_route_state,
    prompt_log_volume,
    route_observation,
)

DEFAULT_MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
DEFAULT_SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
TOPOLOGY_CONTROLS = ("one_hop", "endpoint_rewire", "weight_shuffle")
SCORE_CAPTURE_FIELDS = (
    "response_start",
    "prediction_position",
    "route_log_volume",
    "raw_route_contraction",
    "takeover",
    "valid",
    "target_logprob",
    "functional_prompt_effective_sources",
    "functional_prompt_effective_rank",
    "functional_prompt_anchor_source",
    "attention_prompt_effective_sources",
    "attention_prompt_effective_rank",
    "attention_prompt_anchor_source",
    *(
        f"{name}_{field}"
        for name in TOPOLOGY_CONTROLS
        for field in (
            "route_log_volume",
            "takeover",
            "valid",
        )
    ),
)


def array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_capture(
    output: Path,
    sample: RouteSample,
    trace,
    lineage,
    state,
    graph,
    control_states,
) -> None:
    arrays = {
        "token_ids": array(trace.token_ids),
        "response_start": np.asarray(trace.response_start),
        "prompt_token_unit": array(sample.prompt_units.token_unit_id),
        "evidence_name": np.asarray(sample.prompt_units.evidence_name),
        "evidence_char_span": array(sample.prompt_units.evidence_char_span),
        "query_position": array(trace.events.query_position),
        "prediction_position": array(trace.events.prediction_position),
        "target_logprob": array(trace.target_logprob),
        "target_probability": array(trace.target_confidence),
        "target_margin": array(trace.target_margin),
        "reconstruction_max_abs": array(trace.reconstruction_max_abs),
        "reconstruction_relative_l2": array(trace.reconstruction_relative_l2),
        "mlp_write_norm": array(trace.mlp_write_norm),
        "mlp_relative_norm": array(trace.mlp_relative_norm),
        "mlp_state_cosine": array(trace.mlp_state_cosine),
        "prompt_evidence": array(lineage.prompt_evidence),
        "grounded_response_relay": array(lineage.grounded_response_relay),
        "unrooted_response_feedback": array(lineage.unrooted_response_feedback),
        "predictor_self": array(lineage.predictor_self),
        "unknown_route": array(lineage.unknown),
        "response_ancestry": array(lineage.ancestry[:, lineage.query_position]),
        "raw_route_contraction": state.raw_contraction,
        "takeover": state.takeover,
        "valid": state.valid,
        "route_effective_sources": state.volume.effective_sources,
        "route_effective_head_rank": state.volume.effective_head_rank,
        "route_effective_anchors": state.volume.effective_anchors,
        "route_log_volume": state.volume.log_volume,
        "route_normalized_volume": state.volume.normalized,
    }
    for name, control in control_states.items():
        arrays[f"{name}_route_log_volume"] = control.volume.log_volume
        arrays[f"{name}_raw_route_contraction"] = control.raw_contraction
        arrays[f"{name}_takeover"] = control.takeover
        arrays[f"{name}_valid"] = control.valid
    for family in ("attention_prompt", "functional_prompt"):
        carriers = getattr(trace, family)
        arrays[f"{family}_effective_sources"] = array(carriers.effective_sources)
        arrays[f"{family}_effective_rank"] = array(carriers.effective_rank)
        arrays[f"{family}_anchor_source"] = array(carriers.anchor_source)
    for field in fields(graph):
        arrays[f"graph_{field.name}"] = array(getattr(graph, field.name))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)


def capture_sample(
    replay: RouteMessageReplay,
    sample: RouteSample,
    output: Path,
    *,
    predictor_chunk: int,
    logit_chunk: int,
    route_coverage: float,
    graph_edges_per_head: int,
) -> None:
    def new_tracker(*, multi_hop: bool = True) -> LineageTracker:
        return LineageTracker(
            sample.token_root_unit_id,
            sample.response_start,
            sample.prompt_units.evidence_count,
            len(replay.layers),
            replay.heads,
            device=replay.device,
            multi_hop=multi_hop,
        )

    tracker = new_tracker()
    control_trackers = {
        "one_hop": new_tracker(multi_hop=False),
        "endpoint_rewire": new_tracker(),
        "weight_shuffle": new_tracker(),
    }
    graph_builder = ResponseGraphBuilder(sample.response_start)

    def consume(chunk) -> None:
        capacity = chunk.statistics.capacity
        support = chunk.statistics.support
        tracker.add_dense(chunk.layer, chunk.query_position, capacity, support)
        control_trackers["one_hop"].add_dense(
            chunk.layer,
            chunk.query_position,
            capacity,
            support,
        )
        for name, transform in (
            ("endpoint_rewire", dense_endpoint_rewire),
            ("weight_shuffle", dense_weight_shuffle),
        ):
            changed_capacity, changed_support = transform(
                capacity,
                support,
                chunk.query_position,
                sample.response_start,
            )
            control_trackers[name].add_dense(
                chunk.layer,
                chunk.query_position,
                changed_capacity,
                changed_support,
            )
        rows = sparsify_route_chunk(
            capacity,
            support,
            chunk.statistics.head_write,
            chunk.selected_messages,
            layer=chunk.layer,
            query_position=chunk.query_position,
            coverage=route_coverage,
            max_edges_per_head=graph_edges_per_head,
        )
        graph_builder.add_many(rows)

    trace = replay.capture(
        sample.token_ids,
        sample.response_start,
        predictor_chunk=predictor_chunk,
        logit_chunk=logit_chunk,
        consume_chunk=consume,
    )
    lineage = tracker.finish()
    state = build_route_state(lineage)
    control_states = {
        name: build_route_state(control.finish())
        for name, control in control_trackers.items()
    }
    save_capture(
        output,
        sample,
        trace,
        lineage,
        state,
        graph_builder.finish(),
        control_states,
    )


def sample_record(sample: RouteSample, split_root: Path, path: Path) -> dict:
    return {
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "task_type": sample.task_type,
        "split": sample.split,
        "split_root": str(split_root),
        "path": str(path),
        "data_source": sample.data_source,
        "generator_model": sample.generator_model,
        "response_token_ids": array(sample.token_ids[sample.response_start :]),
    }


def capture_all(args, replay, tokenizer) -> dict[str, dict[str, list[dict]]]:
    records = {task: {"train": [], "test": []} for task in TASK_TYPES}
    for split in ("train", "test"):
        split_root = args.cache / split
        counts = {task: 0 for task in TASK_TYPES}
        for sample in iter_route_samples(split_root, args.source_info, tokenizer):
            task = sample.task_type
            if args.limit is not None and counts[task] >= args.limit:
                continue
            safe_id = sample.sample_id.replace("/", "_")
            path = args.output / "captures" / split / task.casefold() / f"{safe_id}.npz"
            if not path.is_file():
                print(f"capture {split}/{task}: {sample.sample_id}")
                capture_sample(
                    replay,
                    sample,
                    path,
                    predictor_chunk=args.predictor_chunk,
                    logit_chunk=args.logit_chunk,
                    route_coverage=args.route_coverage,
                    graph_edges_per_head=args.graph_edges_per_head,
                )
            records[task][split].append(sample_record(sample, split_root, path))
            counts[task] += 1
            if args.limit is not None and all(
                count >= args.limit for count in counts.values()
            ):
                break

    flat = [
        {**record, "response_token_ids": record["response_token_ids"].tolist()}
        for task in TASK_TYPES
        for split in ("train", "test")
        for record in records[task][split]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in flat),
        encoding="utf-8",
    )
    return records


def load_capture(record: dict) -> dict[str, np.ndarray]:
    with np.load(record["path"]) as stored:
        return {name: stored[name] for name in SCORE_CAPTURE_FIELDS}


def control_record(record: dict, capture: dict, family: str) -> dict:
    return {
        "source_id": record["source_id"],
        "prompt_length": int(capture["response_start"]),
        "volume": prompt_log_volume(
            capture[f"{family}_prompt_effective_sources"],
            capture[f"{family}_prompt_effective_rank"],
            capture[f"{family}_prompt_anchor_source"],
        ),
    }


def lineage_control_record(
    record: dict,
    capture: dict,
    control: str | None = None,
) -> dict:
    """Expose lineage log volume to the equation-locked calibration."""

    key = "route_log_volume" if control is None else f"{control}_route_log_volume"
    return {
        "source_id": record["source_id"],
        "prompt_length": int(capture["response_start"]),
        "volume": np.asarray(capture[key]).T,
    }


def state_arrays(
    capture: dict,
    control: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = "" if control is None else f"{control}_"
    return capture[f"{prefix}takeover"], capture[f"{prefix}valid"]


def control_training_split(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split one physical training side into source-disjoint fit/calibration.

    The old five-fold control used three nuisance-fit folds, one calibration
    fold, and one test fold.  A fixed physical test half cannot reproduce those
    fold sizes, so sorted sources in the available training half use a fixed
    modulo-four split, approximately preserving the old 3:1 ratio.
    """

    sources = sorted({str(record["source_id"]) for record in records})
    if len(sources) == 1:
        # A one-sample run is only a pipeline smoke test. Full experiments use
        # the source-disjoint split below.
        return list(records), list(records)
    calibration_sources = set(sources[::4])
    nuisance = [
        record
        for record in records
        if str(record["source_id"]) not in calibration_sources
    ]
    calibration = [
        record for record in records if str(record["source_id"]) in calibration_sources
    ]
    return nuisance, calibration


def fit_route_detector(
    fit: list[tuple[dict, dict]],
    nuisance: list[tuple[dict, dict]],
    calibration: list[tuple[dict, dict]],
    control: str | None = None,
) -> tuple[EquationLockedRouteCollapseControl, StickyRouteHMM]:
    contraction = EquationLockedRouteCollapseControl.fit(
        [lineage_control_record(*item, control) for item in nuisance],
        [lineage_control_record(*item, control) for item in calibration],
    )
    observations = []
    masks = []
    for record, capture in fit:
        takeover, valid = state_arrays(capture, control)
        observations.append(
            route_observation(
                contraction.score(lineage_control_record(record, capture, control)),
                takeover,
                valid,
            )
        )
        masks.append(valid)
    return contraction, StickyRouteHMM().fit(observations, masks)


def score_fold(
    fit_records: list[dict],
    score_records: list[dict],
    model_path: Path,
) -> list[dict]:
    fit = [(record, load_capture(record)) for record in fit_records]
    scored = [(record, load_capture(record)) for record in score_records]
    nuisance_records, calibration_records = control_training_split(fit_records)
    nuisance_sources = {str(record["source_id"]) for record in nuisance_records}
    calibration_sources = {str(record["source_id"]) for record in calibration_records}
    nuisance = [item for item in fit if str(item[0]["source_id"]) in nuisance_sources]
    calibration = [
        item for item in fit if str(item[0]["source_id"]) in calibration_sources
    ]
    lineage_control, hmm = fit_route_detector(fit, nuisance, calibration)
    topology_models = {
        name: fit_route_detector(fit, nuisance, calibration, name)
        for name in TOPOLOGY_CONTROLS
    }
    functional = EquationLockedRouteCollapseControl.fit(
        [control_record(record, capture, "functional") for record, capture in nuisance],
        [
            control_record(record, capture, "functional")
            for record, capture in calibration
        ],
    )
    attention = EquationLockedRouteCollapseControl.fit(
        [control_record(record, capture, "attention") for record, capture in nuisance],
        [
            control_record(record, capture, "attention")
            for record, capture in calibration
        ],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    hmm.save(model_path)
    lineage_control.save(model_path.with_name(f"{stem}_contraction.npz"))
    functional.save(model_path.with_name(f"{stem}_functional_collapse.npz"))
    attention.save(model_path.with_name(f"{stem}_attention_collapse.npz"))
    for name, (calibration, detector) in topology_models.items():
        calibration.save(model_path.with_name(f"{stem}_{name}_contraction.npz"))
        detector.save(model_path.with_name(f"{stem}_{name}.npz"))
    detectors = {"primary": hmm}
    detectors.update({name: model[1] for name, model in topology_models.items()})
    diagnostics = {
        name: {
            "self_transition": np.diag(detector.transition_).tolist(),
            "expected_dwell_time": detector.expected_dwell_time().tolist(),
        }
        for name, detector in detectors.items()
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    output = []
    for record, capture in scored:
        contraction = lineage_control.score(lineage_control_record(record, capture))
        observation = route_observation(
            contraction,
            capture["takeover"],
            capture["valid"],
        )
        functional_record = control_record(record, capture, "functional")
        attention_record = control_record(record, capture, "attention")
        result = {
            **record,
            "prediction_position": capture["prediction_position"],
            "valid": capture["valid"],
            "contraction": contraction,
            "takeover": capture["takeover"],
            "captured_posterior": hmm.score(observation, capture["valid"]),
            "independent_token_posterior": hmm.independent_score(
                observation, capture["valid"]
            ),
            "functional_route_collapse": functional.score(functional_record),
            "attention_route_collapse": attention.score(attention_record),
            "route_contraction": contraction,
            "raw_route_contraction": capture["raw_route_contraction"],
            "unrooted_takeover": capture["takeover"],
            "confidence": -capture["target_logprob"],
        }
        for name, (calibration, detector) in topology_models.items():
            control_takeover, control_valid = state_arrays(capture, name)
            control_contraction = calibration.score(
                lineage_control_record(record, capture, name)
            )
            control_observation = route_observation(
                control_contraction,
                control_takeover,
                control_valid,
            )
            result[f"{name}_posterior"] = detector.score(
                control_observation,
                control_valid,
            )
        output.append(result)
    return output


def print_report(report: dict) -> None:
    prevalence = report["prevalence"]
    prevalence_text = "n/a" if prevalence is None else f"{prevalence:.4%}"
    print(f"\n=== ALL-{report['task_type'].upper()} EVIDENCE ROUTE STATE ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['positives']} "
        f"evaluated_tokens={report['evaluated_tokens']} "
        f"prevalence={prevalence_text}"
    )
    for name, result in report["detection"].items():
        role = "PRIMARY" if name == report["primary_score"] else "control"
        if result["auroc"] is None:
            print(f"{role:9s} {name:30s} AUROC=n/a AP=n/a")
            continue
        interval = result.get("auroc_ci95", [None, None])
        ci = "n/a" if interval[0] is None else f"[{interval[0]:.6f},{interval[1]:.6f}]"
        print(
            f"{role:9s} {name:30s} "
            f"AUROC={result['auroc']:.6f} "
            f"CI={ci} "
            f"AP={result['average_precision']:.6f} "
            f"lift={result['lift']:.3f}"
        )
    for name, result in report["paired_primary_minus_control"].items():
        if result["auroc_difference"] is None:
            continue
        interval = result.get("auroc_difference_ci95", [None, None])
        ci = "n/a" if interval[0] is None else f"[{interval[0]:.6f},{interval[1]:.6f}]"
        print(
            f"paired    primary - {name:20s} "
            f"dAUROC={result['auroc_difference']:.6f} CI={ci}"
        )


def run_all(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    replay = RouteMessageReplay.from_pretrained(
        args.model,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
    )
    records = capture_all(args, replay, tokenizer)
    provenance = {
        "observer_model": str(args.model),
        "observer_dtype": args.dtype,
        "device": args.device,
        "attention_cache": str(args.cache),
        "source_info": str(args.source_info),
        "predictor_chunk": args.predictor_chunk,
        "graph_edges_per_head": args.graph_edges_per_head,
        "graph_coverage_target": args.route_coverage,
        "generator_models": sorted(
            {
                str(record["generator_model"])
                for task in TASK_TYPES
                for split in ("train", "test")
                for record in records[task][split]
            }
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    for task in TASK_TYPES:
        train = records[task]["train"]
        test = records[task]["test"]
        train_sources = {record["source_id"] for record in train}
        test_sources = {record["source_id"] for record in test}
        if train_sources & test_sources:
            raise ValueError("physical train/test splits must be source-disjoint")
        scored = score_fold(
            train,
            test,
            args.output / "models" / f"{task.casefold()}_fit_train.npz",
        )
        scored += score_fold(
            test,
            train,
            args.output / "models" / f"{task.casefold()}_fit_test.npz",
        )
        report_root = args.output / "reports" / task.casefold()
        frozen = report_root / "frozen_scores.npz"
        freeze_scores(scored, frozen)
        report = evaluate_scores(
            scored,
            frozen,
            report_root / "report.json",
            task_type=task,
            bootstrap=args.bootstrap,
            seed=args.seed,
            provenance=provenance,
        )
        print_report(report)
        print(f"report: {report_root / 'report.json'}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Evidence-conditioned route states")
    root.add_argument("command", choices=("all",))
    root.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    root.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    root.add_argument("--source-info", type=Path, default=DEFAULT_SOURCE_INFO)
    root.add_argument("--output", type=Path)
    root.add_argument("--device", default="cuda:0")
    root.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    root.add_argument("--predictor-chunk", type=int, default=32)
    root.add_argument("--logit-chunk", type=int, default=32)
    root.add_argument("--route-coverage", type=float, default=0.9)
    root.add_argument("--graph-edges-per-head", type=int, default=2)
    root.add_argument("--limit", type=int)
    root.add_argument("--bootstrap", type=int, default=1000)
    root.add_argument("--seed", type=int, default=20260902)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "outputs" / args.model.name
    run_all(args)


if __name__ == "__main__":
    main()
