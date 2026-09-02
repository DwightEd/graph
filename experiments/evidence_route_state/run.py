"""Foreground orchestration for capture, label-free fitting, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .capture import RegisterGraphReplay
from .controls import RouteCollapseControl, prompt_log_volume
from .data import TASK_TYPES, RouteSample, iter_route_samples
from .detector import GraphRecord, TransitionDetector
from .evaluate import evaluate_scores, freeze_scores
from .graph import GraphSequence
from .registers import ORIGIN_NAMES

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

GRAPH_FLOAT_FIELDS = (
    "node_embedding",
    "residual_gram",
    "head_write_gram",
    "route_topology",
    "mlp_relation",
    "margin_contribution",
)
GRAPH_FLOAT32_FIELDS = ("residual_gram", "head_write_gram")
PROMPT_CONTROL_FIELDS = (
    "effective_sources",
    "effective_rank",
    "anchor_source",
)


def array(value: object) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value)


def save_capture(path: Path, sample: RouteSample, trace) -> None:
    """Persist one compact graph sequence; all dense endpoints already contributed."""

    graph = trace.graph
    arrays = {
        "token_ids": array(trace.token_ids),
        "response_start": np.asarray(trace.response_start, dtype=np.int32),
        "prompt_token_unit": array(sample.prompt_units.token_unit_id),
        "evidence_name": np.asarray(sample.prompt_units.evidence_name),
        "evidence_char_span": array(sample.prompt_units.evidence_char_span),
        "query_position": array(graph.query_position).astype(np.int32),
        "prediction_position": array(graph.prediction_position).astype(np.int32),
        "valid": array(graph.valid).astype(bool),
        "target_logprob": array(trace.target_logprob).astype(np.float32),
        "target_probability": array(trace.target_confidence).astype(np.float32),
        "target_margin": array(trace.target_margin).astype(np.float32),
        "attention_write_error": array(trace.attention_write_error).astype(np.float32),
        "register_closure_error": array(trace.register_closure_error).astype(
            np.float32
        ),
    }
    arrays.update(
        {
            name: array(getattr(graph, name)).astype(
                np.float32 if name in GRAPH_FLOAT32_FIELDS else np.float16
            )
            for name in GRAPH_FLOAT_FIELDS
        }
    )
    for family in ("attention", "functional"):
        geometry = getattr(trace.prompt_route, family)
        for name in PROMPT_CONTROL_FIELDS:
            value = array(getattr(geometry, name))
            dtype = np.int32 if name == "anchor_source" else np.float16
            arrays[f"{family}_prompt_{name}"] = value.astype(dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_sequence(path: str | Path) -> GraphSequence:
    """Load only the structured frame consumed by the detector."""

    with np.load(path) as stored:
        return GraphSequence(
            query_position=stored["query_position"],
            prediction_position=stored["prediction_position"],
            node_embedding=stored["node_embedding"],
            residual_gram=stored["residual_gram"],
            head_write_gram=stored["head_write_gram"],
            route_topology=stored["route_topology"],
            mlp_relation=stored["mlp_relation"],
            margin_contribution=stored["margin_contribution"],
            valid=stored["valid"],
        )


def sample_record(sample: RouteSample, split_root: Path, path: Path) -> dict:
    return {
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "task_type": sample.task_type,
        "split": sample.split,
        "split_root": str(split_root),
        "path": str(path),
        "prompt_length": sample.response_start,
        "data_source": sample.data_source,
        "generator_model": sample.generator_model,
        "response_token_ids": array(sample.token_ids[sample.response_start :]),
    }


def capture_all(args, replay, tokenizer) -> dict[str, dict[str, list[dict]]]:
    """Capture every task once on both physical source-disjoint sides."""

    records = {task: {"train": [], "test": []} for task in TASK_TYPES}
    for split in ("train", "test"):
        split_root = args.cache / split
        counts = {task: 0 for task in TASK_TYPES}
        for sample in iter_route_samples(split_root, args.source_info, tokenizer):
            task = sample.task_type
            if args.limit is not None and counts[task] >= args.limit:
                continue
            safe_id = sample.sample_id.replace("/", "_")
            path = (
                args.output
                / "graph_sequences"
                / split
                / task.casefold()
                / f"{safe_id}.npz"
            )
            if not path.is_file():
                print(f"capture {split}/{task}: {sample.sample_id}")
                trace = replay.capture(
                    sample.token_ids,
                    sample.response_start,
                    sample.prompt_units.evidence_mask,
                    predictor_chunk=args.predictor_chunk,
                )
                save_capture(path, sample, trace)
            records[task][split].append(sample_record(sample, split_root, path))
            counts[task] += 1
            if args.limit is not None and all(
                count >= args.limit for count in counts.values()
            ):
                break

    serializable = [
        {
            **record,
            "response_token_ids": record["response_token_ids"].tolist(),
        }
        for task in TASK_TYPES
        for split in ("train", "test")
        for record in records[task][split]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in serializable),
        encoding="utf-8",
    )
    return records


def split_reference(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Use three source groups for prototypes and one for score calibration."""

    sources = sorted({str(record["source_id"]) for record in records})
    if len(sources) < 2:
        raise ValueError(
            "each task/split needs at least two sources for disjoint "
            "reference and calibration; use --limit 2 or greater"
        )
    calibration_sources = set(sources[::4])
    reference = [
        record
        for record in records
        if str(record["source_id"]) not in calibration_sources
    ]
    calibration = [
        record for record in records if str(record["source_id"]) in calibration_sources
    ]
    return reference, calibration


def graph_records(records: list[dict]) -> list[GraphRecord]:
    return [
        GraphRecord(
            source_id=str(record["source_id"]),
            prompt_length=int(record["prompt_length"]),
            sequence=load_sequence(record["path"]),
        )
        for record in records
    ]


def route_control_record(record: dict, family: str) -> dict:
    with np.load(record["path"]) as stored:
        volume = prompt_log_volume(
            stored[f"{family}_prompt_effective_sources"],
            stored[f"{family}_prompt_effective_rank"],
            stored[f"{family}_prompt_anchor_source"],
        )
    return {
        "source_id": str(record["source_id"]),
        "prompt_length": int(record["prompt_length"]),
        "volume": volume,
    }


def fit_route_control(
    reference: list[dict], calibration: list[dict], family: str
) -> RouteCollapseControl:
    return RouteCollapseControl.fit(
        [route_control_record(record, family) for record in reference],
        [route_control_record(record, family) for record in calibration],
    )


def score_fold(
    fit_records: list[dict],
    score_records: list[dict],
    model_path: Path,
    *,
    prototype_count: int,
) -> list[dict]:
    """Fit on one physical side and score only the opposite physical side."""

    reference, calibration = split_reference(fit_records)
    reference_graphs = graph_records(reference)
    calibration_graphs = graph_records(calibration)
    detector = TransitionDetector(prototype_count).fit(reference_graphs)
    detector.calibrate(calibration_graphs)
    functional = fit_route_control(reference, calibration, "functional")
    attention = fit_route_control(reference, calibration, "attention")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)
    functional.save(model_path.with_name(f"{model_path.stem}_functional.npz"))
    attention.save(model_path.with_name(f"{model_path.stem}_attention.npz"))
    del reference_graphs, calibration_graphs

    scored = []
    for record in score_records:
        sequence = load_sequence(record["path"])
        graph_record = GraphRecord(
            source_id=str(record["source_id"]),
            prompt_length=int(record["prompt_length"]),
            sequence=sequence,
        )
        conditional = detector.score(graph_record)
        independent = detector.independent_score(graph_record)
        functional_score = functional.score(route_control_record(record, "functional"))
        attention_score = attention.score(route_control_record(record, "attention"))
        with np.load(record["path"]) as stored:
            confidence = -stored["target_logprob"].astype(np.float32)
        scored.append(
            {
                **record,
                "query_position": array(sequence.query_position),
                "prediction_position": array(sequence.prediction_position),
                "valid": array(sequence.valid) & np.isfinite(conditional),
                "conditional_graph_energy": conditional,
                "independent_graph_energy": independent,
                "functional_route_collapse": functional_score,
                "attention_route_collapse": attention_score,
                "confidence": confidence,
            }
        )
    return scored


def print_report(report: dict) -> None:
    prevalence = report["prevalence"]
    prevalence_text = "n/a" if prevalence is None else f"{prevalence:.4%}"
    print(f"\n=== ALL-{report['task_type'].upper()} REGISTERED ROUTE GRAPH ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['positives']} "
        f"evaluated_tokens={report['evaluated_tokens']} prevalence={prevalence_text}"
    )
    for name, result in report["detection"].items():
        role = "PRIMARY" if name == report["primary_score"] else "control"
        if result["auroc"] is None:
            print(f"{role:9s} {name:30s} AUROC=n/a AP=n/a")
            continue
        interval = result.get("auroc_ci95", [None, None])
        ci = "n/a" if interval[0] is None else f"[{interval[0]:.6f},{interval[1]:.6f}]"
        print(
            f"{role:9s} {name:30s} AUROC={result['auroc']:.6f} CI={ci} "
            f"AP={result['average_precision']:.6f} lift={result['lift']:.3f}"
        )
    for name, result in report["paired_primary_minus_control"].items():
        if result["auroc_difference"] is None:
            continue
        interval = result.get("auroc_ci95", [None, None])
        ci = "n/a" if interval[0] is None else f"[{interval[0]:.6f},{interval[1]:.6f}]"
        print(
            f"paired    primary - {name:20s} "
            f"dAUROC={result['auroc_difference']:.6f} CI={ci}"
        )


def run_all(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    replay = RegisterGraphReplay.from_pretrained(
        args.model,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
    )
    records = capture_all(args, replay, tokenizer)
    del replay
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    provenance = {
        "observer_model": str(args.model),
        "observer_dtype": args.dtype,
        "device": args.device,
        "attention_cache": str(args.cache),
        "source_info": str(args.source_info),
        "predictor_chunk": args.predictor_chunk,
        "register_origins": ORIGIN_NAMES,
        "transition_history": 2,
        "prototype_count": args.prototype_count,
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
            prototype_count=args.prototype_count,
        )
        scored += score_fold(
            test,
            train,
            args.output / "models" / f"{task.casefold()}_fit_test.npz",
            prototype_count=args.prototype_count,
        )
        report_root = args.output / "reports" / task.casefold()
        frozen_path = report_root / "frozen_scores.npz"
        freeze_scores(scored, frozen_path)
        report = evaluate_scores(
            scored,
            frozen_path,
            report_root / "report.json",
            task_type=task,
            bootstrap=args.bootstrap,
            seed=args.seed,
            provenance=provenance,
        )
        print_report(report)
        print(f"report: {report_root / 'report.json'}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Registered information-route graph")
    root.add_argument("command", choices=("all",))
    root.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    root.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    root.add_argument("--source-info", type=Path, default=DEFAULT_SOURCE_INFO)
    root.add_argument("--output", type=Path)
    root.add_argument("--device", default="cuda:0")
    root.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    root.add_argument("--predictor-chunk", type=int, default=16)
    root.add_argument("--prototype-count", type=int, default=8)
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
