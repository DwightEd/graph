"""Evaluate frozen hallucination scores under controlled dataset conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import ArtifactSpec, load_score_artifact
from .dataset import align_artifacts, attach_dataset_evaluation
from .runner import BenchmarkConfig, run_benchmark


def _artifact_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be NAME=/path/to/scores.npz")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("artifact name and path must be non-empty")
    return {"name": name, "path": path, "adapter": "auto"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="optional JSON configuration")
    parser.add_argument("--split-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--artifact",
        action="append",
        type=_artifact_argument,
        help="repeatable NAME=/path/to/frozen_scores.npz",
    )
    parser.add_argument("--task-type", action="append")
    parser.add_argument("--data-source", action="append")
    parser.add_argument("--generator-model", action="append")
    parser.add_argument("--positive-rate", action="append")
    parser.add_argument("--metric", action="append")
    parser.add_argument("--evaluation-unit", choices=("token", "response"))
    parser.add_argument(
        "--response-aggregation", choices=("max", "mean", "topk_mean")
    )
    parser.add_argument("--response-top-fraction", type=float)
    parser.add_argument("--ratio-mode", choices=("reweight", "subsample"))
    parser.add_argument("--ratio-repeats", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--relative-position-min", type=float)
    parser.add_argument("--relative-position-max", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _load_configuration(args):
    value = {}
    if args.config:
        value = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("configuration root must be a JSON object")
    split_root = args.split_root or value.get("split_root")
    output_dir = args.output_dir or value.get("output_dir")
    artifacts = args.artifact or value.get("artifacts")
    if not split_root or not output_dir or not artifacts:
        raise ValueError("split_root, output_dir, and at least one artifact are required")

    benchmark = dict(value.get("benchmark", {}))
    overrides = {
        "task_types": args.task_type,
        "data_sources": args.data_source,
        "generator_models": args.generator_model,
        "positive_rates": args.positive_rate,
        "metrics": args.metric,
        "evaluation_unit": args.evaluation_unit,
        "response_aggregation": args.response_aggregation,
        "response_top_fraction": args.response_top_fraction,
        "ratio_mode": args.ratio_mode,
        "ratio_repeats": args.ratio_repeats,
        "bootstrap_replicates": args.bootstrap,
        "relative_position_min": args.relative_position_min,
        "relative_position_max": args.relative_position_max,
        "seed": args.seed,
    }
    benchmark.update({name: item for name, item in overrides.items() if item is not None})
    specs = [ArtifactSpec.from_mapping(item) for item in artifacts]
    return str(split_root), str(output_dir), specs, BenchmarkConfig.from_mapping(benchmark)


def main(argv=None):
    args = parse_args(argv)
    split_root, output_dir, specs, config = _load_configuration(args)
    artifacts = [load_score_artifact(spec) for spec in specs]
    sample_id, token_index, methods, metadata = align_artifacts(artifacts)
    frame = attach_dataset_evaluation(
        sample_id,
        token_index,
        methods,
        metadata,
        split_root,
        device=args.device,
    )
    report = run_benchmark(
        frame,
        output_dir,
        config=config,
        artifacts=[
            {
                "name": artifact.name,
                "path": artifact.path,
                "schema": artifact.schema,
                "original_rows": int(len(artifact.sample_id)),
            }
            for artifact in artifacts
        ],
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "aligned_rows": report["aligned_rows"],
                "aligned_samples": report["aligned_samples"],
                "methods": len(report["methods"]),
                "conditions": sum(
                    row["state"] == "complete" for row in report["conditions"]
                ),
                "results": str(Path(output_dir) / "results.json"),
                "metrics": str(Path(output_dir) / "metrics_long.csv"),
                "wide_metrics": str(Path(output_dir) / "metrics_wide.csv"),
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    main()
