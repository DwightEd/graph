"""Orchestrate method-by-condition evaluation and write tidy reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiment_protocol import FrozenEvaluation, FrozenFile
from research_dataset import open_research_dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments

    def tqdm(iterable, **_):
        return iterable


from .artifacts import ArtifactSpec, load_score_artifact
from .conditions import (
    aggregate_responses,
    condition_grid,
    condition_mask,
    prevalence_weights,
    stratified_subsample,
)
from .dataset import build_benchmark_frame
from .metrics import (
    DEFAULT_METRICS,
    METRICS,
    cluster_bootstrap_indices,
    evaluate_metrics,
)
from .types import BenchmarkFrame, EvaluatedArtifact

REPORT_SCHEMA = "conditioned-detector-benchmark-v2"


@dataclass(frozen=True)
class BenchmarkConfig:
    task_types: tuple[str, ...] = ("all", "each")
    data_sources: tuple[str, ...] = ("all",)
    generator_models: tuple[str, ...] = ("all",)
    positive_rates: tuple[str | float, ...] = (
        "native",
        0.01,
        0.03,
        0.05,
        0.10,
        0.25,
        0.50,
    )
    metrics: tuple[str, ...] = DEFAULT_METRICS
    evaluation_unit: str = "token"
    response_aggregation: str = "max"
    response_top_fraction: float = 0.10
    ratio_mode: str = "reweight"
    ratio_repeats: int = 20
    bootstrap_replicates: int = 200
    relative_position_min: float = 0.0
    relative_position_max: float = 1.0
    seed: int = 20260817

    @classmethod
    def from_mapping(cls, value):
        allowed = {
            "task_types",
            "data_sources",
            "generator_models",
            "positive_rates",
            "metrics",
            "evaluation_unit",
            "response_aggregation",
            "response_top_fraction",
            "ratio_mode",
            "ratio_repeats",
            "bootstrap_replicates",
            "relative_position_min",
            "relative_position_max",
            "seed",
        }
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported benchmark settings: {unknown}")
        fields = {
            "task_types": tuple(value.get("task_types", cls.task_types)),
            "data_sources": tuple(value.get("data_sources", cls.data_sources)),
            "generator_models": tuple(
                value.get("generator_models", cls.generator_models)
            ),
            "positive_rates": tuple(value.get("positive_rates", cls.positive_rates)),
            "metrics": tuple(value.get("metrics", cls.metrics)),
            "evaluation_unit": str(value.get("evaluation_unit", cls.evaluation_unit)),
            "response_aggregation": str(
                value.get("response_aggregation", cls.response_aggregation)
            ),
            "response_top_fraction": float(
                value.get("response_top_fraction", cls.response_top_fraction)
            ),
            "ratio_mode": str(value.get("ratio_mode", cls.ratio_mode)),
            "ratio_repeats": int(value.get("ratio_repeats", cls.ratio_repeats)),
            "bootstrap_replicates": int(
                value.get("bootstrap_replicates", cls.bootstrap_replicates)
            ),
            "relative_position_min": float(
                value.get("relative_position_min", cls.relative_position_min)
            ),
            "relative_position_max": float(
                value.get("relative_position_max", cls.relative_position_max)
            ),
            "seed": int(value.get("seed", cls.seed)),
        }
        return cls(**fields).validate()

    def validate(self):
        if self.evaluation_unit not in {"token", "response"}:
            raise ValueError("evaluation_unit must be token or response")
        if self.ratio_mode not in {"reweight", "subsample"}:
            raise ValueError("ratio_mode must be reweight or subsample")
        if not 0.0 <= self.relative_position_min <= self.relative_position_max <= 1.0:
            raise ValueError(
                "relative position bounds must satisfy 0 <= min <= max <= 1"
            )
        if self.ratio_repeats < 1 or self.bootstrap_replicates < 0:
            raise ValueError("repeat counts are invalid")
        unknown = set(self.metrics).difference(METRICS)
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)}")
        return self


def _interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition_rows_reweighted(frame, selected, condition, config, condition_seed):
    labels = frame.labels[selected]
    weights = prevalence_weights(labels, condition.target_positive_rate)
    bootstraps = cluster_bootstrap_indices(
        frame.source_id[selected],
        replicates=config.bootstrap_replicates,
        seed=condition_seed,
    )
    return [(selected, weights)], bootstraps


def _condition_rows_subsampled(frame, selected, condition, config, condition_seed):
    if condition.target_positive_rate is None:
        return [(selected, np.ones(len(selected), dtype=np.float64))], []
    repetitions = []
    for repeat in range(config.ratio_repeats):
        local = stratified_subsample(
            frame.labels[selected],
            condition.target_positive_rate,
            seed=condition_seed + repeat,
        )
        rows = selected[local]
        repetitions.append((rows, np.ones(len(rows), dtype=np.float64)))
    return repetitions, []


def _evaluate_and_write(
    frame: BenchmarkFrame,
    output_dir,
    *,
    config: BenchmarkConfig,
    artifacts: list[dict],
):
    aligned_token_rows = len(frame.labels)
    aligned_samples = len(np.unique(frame.sample_id))
    position = (frame.relative_position >= config.relative_position_min) & (
        frame.relative_position <= config.relative_position_max
    )
    frame = frame.subset(position)
    if config.evaluation_unit == "response":
        frame = aggregate_responses(
            frame,
            aggregation=config.response_aggregation,
            top_fraction=config.response_top_fraction,
        )
    evaluated_rows = len(frame.labels)
    evaluated_samples = len(np.unique(frame.sample_id))
    conditions = condition_grid(
        frame,
        tasks=config.task_types,
        data_sources=config.data_sources,
        generator_models=config.generator_models,
        positive_rates=config.positive_rates,
    )
    metric_rows = []
    wide_rows = []
    condition_reports = []
    for condition_index, condition in enumerate(
        tqdm(conditions, desc="benchmark conditions", unit="condition")
    ):
        selected = np.flatnonzero(condition_mask(frame, condition))
        if not len(selected) or len(np.unique(frame.labels[selected])) < 2:
            condition_reports.append(
                {
                    "condition": asdict(condition),
                    "state": "skipped",
                    "reason": "condition has fewer than two label classes",
                    "rows": len(selected),
                }
            )
            continue
        condition_seed = config.seed + 1009 * condition_index
        if config.ratio_mode == "reweight":
            repetitions, bootstraps = _condition_rows_reweighted(
                frame, selected, condition, config, condition_seed
            )
        else:
            repetitions, bootstraps = _condition_rows_subsampled(
                frame, selected, condition, config, condition_seed
            )
        method_reports = {}
        for method_name, method in frame.methods.items():
            repeat_metrics = []
            for rows, weights in repetitions:
                repeat_metrics.append(
                    evaluate_metrics(
                        frame.labels[rows],
                        method.values[rows],
                        weights,
                        config.metrics,
                    )
                )
            point = {
                name: float(np.mean([value[name] for value in repeat_metrics]))
                for name in config.metrics
            }
            bootstrap_distributions = {name: [] for name in config.metrics}
            if bootstraps:
                for local_rows in bootstraps:
                    rows = selected[local_rows]
                    if len(np.unique(frame.labels[rows])) < 2:
                        continue
                    weights = prevalence_weights(
                        frame.labels[rows], condition.target_positive_rate
                    )
                    values = evaluate_metrics(
                        frame.labels[rows],
                        method.values[rows],
                        weights,
                        config.metrics,
                    )
                    for name in config.metrics:
                        bootstrap_distributions[name].append(values[name])
            metrics = {}
            for name in config.metrics:
                if config.ratio_mode == "reweight":
                    valid_resamples = np.asarray(
                        bootstrap_distributions[name], dtype=np.float64
                    )
                    valid_resamples = valid_resamples[np.isfinite(valid_resamples)]
                    if len(valid_resamples) < 2:
                        uncertainty = {
                            "uncertainty_scope": (
                                "not_estimated_insufficient_resamples"
                            )
                        }
                    else:
                        ci_low, ci_high = _interval(valid_resamples)
                        uncertainty = {
                            "uncertainty_scope": (
                                "source_cluster_bootstrap_percentile_95"
                            ),
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
                else:
                    valid_repeats = np.asarray(
                        [value[name] for value in repeat_metrics], dtype=np.float64
                    )
                    valid_repeats = valid_repeats[np.isfinite(valid_repeats)]
                    if condition.target_positive_rate is None:
                        uncertainty = {
                            "uncertainty_scope": "not_estimated_native_prevalence"
                        }
                    elif len(valid_repeats) < 2:
                        uncertainty = {
                            "uncertainty_scope": (
                                "not_estimated_insufficient_resamples"
                            )
                        }
                    else:
                        repeat_q025, repeat_q975 = _interval(valid_repeats)
                        uncertainty = {
                            "uncertainty_scope": (
                                "repeated_row_subsample_variability"
                            ),
                            "repeat_q025": repeat_q025,
                            "repeat_q975": repeat_q975,
                        }
                metrics[name] = {"value": point[name], **uncertainty}
                metric_rows.append(
                    {
                        "condition_id": condition.identifier,
                        "task_type": condition.task_type or "ALL",
                        "data_source": condition.data_source or "ALL",
                        "generator_model": condition.generator_model or "ALL",
                        "target_positive_rate": (
                            "native"
                            if condition.target_positive_rate is None
                            else condition.target_positive_rate
                        ),
                        "ratio_mode": config.ratio_mode,
                        "evaluation_unit": config.evaluation_unit,
                        "method": method_name,
                        "protocol": method.protocol,
                        "metric": name,
                        "prevalence_sensitive": METRICS[name].prevalence_sensitive,
                        "value": point[name],
                        **uncertainty,
                        "rows": len(repetitions[0][0]),
                        "positives": int(frame.labels[repetitions[0][0]].sum()),
                        "native_prevalence": float(frame.labels[selected].mean()),
                    }
                )
            wide_row = {
                "condition_id": condition.identifier,
                "task_type": condition.task_type or "ALL",
                "data_source": condition.data_source or "ALL",
                "generator_model": condition.generator_model or "ALL",
                "target_positive_rate": (
                    "native"
                    if condition.target_positive_rate is None
                    else condition.target_positive_rate
                ),
                "ratio_mode": config.ratio_mode,
                "evaluation_unit": config.evaluation_unit,
                "method": method_name,
                "protocol": method.protocol,
                "source_direction": method.source_direction or "higher",
                "rows": len(repetitions[0][0]),
                "positives": int(frame.labels[repetitions[0][0]].sum()),
                "native_prevalence": float(frame.labels[selected].mean()),
            }
            for name, values in metrics.items():
                wide_row[name] = values["value"]
                wide_row["uncertainty_scope"] = values["uncertainty_scope"]
                if config.ratio_mode == "reweight":
                    wide_row[f"{name}_ci_low"] = values.get("ci_low")
                    wide_row[f"{name}_ci_high"] = values.get("ci_high")
                else:
                    wide_row[f"{name}_repeat_q025"] = values.get("repeat_q025")
                    wide_row[f"{name}_repeat_q975"] = values.get("repeat_q975")
            wide_rows.append(wide_row)
            method_reports[method_name] = {
                "protocol": method.protocol,
                "source_field": method.source_field,
                "source_direction": method.source_direction or "higher",
                "metrics": metrics,
            }
        condition_reports.append(
            {
                "condition": asdict(condition),
                "state": "complete",
                "native_rows": len(selected),
                "native_positives": int(frame.labels[selected].sum()),
                "native_prevalence": float(frame.labels[selected].mean()),
                "evaluated_rows": len(repetitions[0][0]),
                "methods": method_reports,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": REPORT_SCHEMA,
        "state": "complete",
        "labels_used_during": "posthoc_conditioned_evaluation_only",
        "score_fitting_repeated_per_condition": False,
        "alignment": "intersection_of_all_artifact_token_rows",
        "config": asdict(config),
        "artifacts": artifacts,
        "methods": {
            name: {
                "protocol": method.protocol,
                "source_field": method.source_field,
                "source_direction": method.source_direction or "higher",
            }
            for name, method in frame.methods.items()
        },
        "metric_registry": {
            name: {
                "description": METRICS[name].description,
                "prevalence_sensitive": METRICS[name].prevalence_sensitive,
            }
            for name in config.metrics
        },
        "aligned_token_rows": aligned_token_rows,
        "aligned_samples": aligned_samples,
        "evaluated_rows": evaluated_rows,
        "evaluated_samples": evaluated_samples,
        "evaluation_unit": config.evaluation_unit,
        "conditions": condition_reports,
    }
    report_path = output_dir / "results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "condition_id",
        "task_type",
        "data_source",
        "generator_model",
        "target_positive_rate",
        "ratio_mode",
        "evaluation_unit",
        "method",
        "protocol",
        "metric",
        "prevalence_sensitive",
        "value",
        "uncertainty_scope",
        "rows",
        "positives",
        "native_prevalence",
    ]
    if config.ratio_mode == "reweight":
        fields[fields.index("rows") : fields.index("rows")] = ["ci_low", "ci_high"]
    else:
        fields[fields.index("rows") : fields.index("rows")] = [
            "repeat_q025",
            "repeat_q975",
        ]
    _write_csv(output_dir / "metrics_long.csv", metric_rows, fields)
    wide_fields = [
        "condition_id",
        "task_type",
        "data_source",
        "generator_model",
        "target_positive_rate",
        "ratio_mode",
        "evaluation_unit",
        "method",
        "protocol",
        "source_direction",
        "uncertainty_scope",
        "rows",
        "positives",
        "native_prevalence",
    ]
    for name in config.metrics:
        if config.ratio_mode == "reweight":
            wide_fields.extend((name, f"{name}_ci_low", f"{name}_ci_high"))
        else:
            wide_fields.extend((name, f"{name}_repeat_q025", f"{name}_repeat_q975"))
    _write_csv(output_dir / "metrics_wide.csv", wide_rows, wide_fields)
    summary = [
        f"schema={REPORT_SCHEMA}",
        (
            f"aligned_token_rows={aligned_token_rows} "
            f"aligned_samples={aligned_samples}"
        ),
        (
            f"evaluated_rows={evaluated_rows} "
            f"evaluated_samples={evaluated_samples}"
        ),
        (
            f"methods={len(frame.methods)} conditions_complete="
            f"{sum(row['state'] == 'complete' for row in condition_reports)}"
        ),
        f"ratio_mode={config.ratio_mode} evaluation_unit={config.evaluation_unit}",
        (
            "AUPRC and AUPRC lift change with positive prevalence; AUROC "
            "is the cross-ratio ranking control."
        ),
        f"results={report_path}",
        f"tidy_metrics={output_dir / 'metrics_long.csv'}",
        f"wide_metrics={output_dir / 'metrics_wide.csv'}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return report


class ConditionedBenchmark:
    """Run the complete sealed conditioned-evaluation workflow."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = (BenchmarkConfig() if config is None else config).validate()

    def run(
        self,
        split_root,
        output_dir,
        artifact_specs: list[ArtifactSpec],
        device="cpu",
    ):
        specs = list(artifact_specs)
        if not specs:
            raise ValueError("at least one artifact is required")

        frozen_files = [FrozenFile.capture(spec.path) for spec in specs]
        scores = [
            load_score_artifact(spec, frozen)
            for spec, frozen in zip(specs, frozen_files, strict=True)
        ]
        for frozen in frozen_files:
            frozen.verify(frozen.path)

        dataset = open_research_dataset(
            split_root,
            device=device,
            retain_embedded_labels=True,
        )
        evaluations = [
            FrozenEvaluation(frozen, expected_split="test") for frozen in frozen_files
        ]
        evaluation_rows = [score.evaluation_rows() for score in scores]
        aligned_labels = FrozenEvaluation.align_all(
            dataset,
            zip(evaluations, evaluation_rows, strict=True),
        )
        evaluated = [
            EvaluatedArtifact(
                score=score,
                labels=labels,
            )
            for score, labels in zip(scores, aligned_labels, strict=True)
        ]
        frame = build_benchmark_frame(evaluated, dataset)
        manifests = [
            {
                "name": score.name,
                "path": str(frozen.path),
                "sha256": frozen.sha256,
                "schema": score.schema,
                "dataset_manifest_sha256": score.dataset_manifest_sha256,
                "evaluation_rows": len(score.sample_id),
            }
            for score, frozen in zip(scores, frozen_files, strict=True)
        ]
        return _evaluate_and_write(
            frame,
            output_dir,
            config=self.config,
            artifacts=manifests,
        )
