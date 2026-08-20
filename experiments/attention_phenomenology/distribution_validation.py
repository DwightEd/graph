"""Label-free suitability audit for Dirichlet attention-composition models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset

from .artifacts import write_json
from .compositions import composition_views
from .config import PhenomenologyConfig
from .distributions import distribution_diagnostics
from .features import analyze_routing
from .reference import token_buckets
from .routing import collect_routing_edges


REFERENCE_SCHEMA = "attention-composition-distribution-reference-v1"
VALIDATION_SCHEMA = "attention-composition-distribution-validation-v1"


@dataclass(frozen=True)
class DistributionValidationConfig:
    """Sampling and model choices for the distribution-suitability audit."""

    representations: tuple[str, ...] = ("role", "provenance")
    fit_reservoir_rows: int = 1024
    validation_reservoir_rows: int = 1024
    minimum_group_rows: int = 128
    pseudocounts: tuple[float, ...] = (1e-6, 1e-4, 1e-3)
    simulation_rows: int = 4096
    random_seed: int = 20260820


class PriorityReservoir:
    """Uniform fixed-size sample implemented with random priorities."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.values: np.ndarray | None = None
        self.priority: np.ndarray | None = None
        self.seen = 0

    def add_batch(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float32)
        if batch.ndim != 2 or not len(batch):
            return
        priority = self.rng.random(len(batch))
        self.seen += len(batch)
        if self.values is None:
            combined_values = batch
            combined_priority = priority
        else:
            combined_values = np.concatenate((self.values, batch), axis=0)
            combined_priority = np.concatenate((self.priority, priority), axis=0)
        if len(combined_values) > self.capacity:
            selected = np.argpartition(
                combined_priority, -self.capacity
            )[-self.capacity :]
            combined_values = combined_values[selected]
            combined_priority = combined_priority[selected]
        self.values = combined_values
        self.priority = combined_priority

    def matrix(self) -> np.ndarray:
        if self.values is None:
            raise ValueError("reservoir is empty")
        return self.values


GroupKey = tuple[str, str, int, int]


def _samples(dataset, limit: int | None):
    sample_ids = dataset.sample_ids if limit is None else dataset.sample_ids[:limit]
    return (dataset[sample_id] for sample_id in sample_ids)


def _collect_reservoirs(
    *,
    split_root,
    device: str,
    phenomenology_config: PhenomenologyConfig,
    representations: tuple[str, ...],
    reservoir_rows: int,
    seed: int,
    limit: int | None,
) -> tuple[dict[GroupKey, PriorityReservoir], dict[str, tuple[str, ...]]]:
    dataset = open_research_dataset(split_root, device=device)
    rng = np.random.default_rng(seed)
    reservoirs: dict[GroupKey, PriorityReservoir] = {}
    component_names: dict[str, tuple[str, ...]] = {}

    for sample in _samples(dataset, limit):
        analysis = analyze_routing(
            collect_routing_edges(sample, config=phenomenology_config),
            config=phenomenology_config,
        )
        views = composition_views(analysis, epsilon=phenomenology_config.epsilon)
        buckets = token_buckets(
            analysis.layer_features.shape[0],
            phenomenology_config.causal_position_bins,
        )
        task = str(sample.task_type or "unknown")

        for representation in representations:
            view = views[representation]
            component_names[representation] = view.component_names
            values = view.values.detach().cpu().numpy().astype(np.float32)
            for bucket in np.unique(buckets):
                token_selected = buckets == bucket
                for layer in range(values.shape[1]):
                    rows = values[token_selected, layer].reshape(
                        -1, values.shape[-1]
                    )
                    key = (representation, task, int(bucket), layer)
                    reservoir = reservoirs.get(key)
                    if reservoir is None:
                        reservoir = PriorityReservoir(reservoir_rows, rng)
                        reservoirs[key] = reservoir
                    reservoir.add_batch(rows)
        sample.release_attention()
    return reservoirs, component_names


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(rows: list[dict], name: str) -> float:
    weights = np.asarray([row["validation_rows"] for row in rows], dtype=np.float64)
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _summary_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    representations = sorted({str(row["representation"]) for row in rows})
    for representation in representations:
        result[representation] = {"pseudocounts": {}}
        values = sorted(
            {
                float(row["pseudocount"])
                for row in rows
                if row["representation"] == representation
            }
        )
        for pseudocount in values:
            selected = [
                row
                for row in rows
                if row["representation"] == representation
                and float(row["pseudocount"]) == pseudocount
            ]
            delta = np.asarray(
                [row["dirichlet_minus_logistic_normal_nats"] for row in selected]
            )
            result[representation]["pseudocounts"][str(pseudocount)] = {
                "groups": len(selected),
                "validation_rows": int(
                    sum(int(row["validation_rows"]) for row in selected)
                ),
                "dirichlet_converged_fraction": float(
                    np.mean([bool(row["dirichlet_converged"]) for row in selected])
                ),
                "weighted_dirichlet_minus_logistic_normal_nats": _weighted_mean(
                    selected, "dirichlet_minus_logistic_normal_nats"
                ),
                "groups_where_dirichlet_has_higher_log_likelihood_fraction": float(
                    np.mean(delta > 0)
                ),
                "groups_where_dirichlet_is_within_005_nats_fraction": float(
                    np.mean(delta > -0.05)
                ),
                "weighted_covariance_relative_frobenius_error": _weighted_mean(
                    selected, "covariance_relative_frobenius_error"
                ),
                "weighted_positive_offdiagonal_covariance_fraction": _weighted_mean(
                    selected, "positive_offdiagonal_covariance_fraction"
                ),
                "weighted_nll_pit_ks_statistic": _weighted_mean(
                    selected, "nll_pit_ks_statistic"
                ),
                "weighted_tail_probability_below_005_fraction": _weighted_mean(
                    selected, "nll_tail_probability_below_005_fraction"
                ),
            }
    return result


def validate_composition_distributions(
    *,
    fit_split,
    validation_split,
    output_dir,
    device: str = "cpu",
    phenomenology_config: PhenomenologyConfig | None = None,
    validation_config: DistributionValidationConfig | None = None,
    fit_limit: int | None = None,
    validation_limit: int | None = None,
) -> dict[str, object]:
    """Fit on one unlabeled split and evaluate model adequacy on another."""

    phenomenology_config = (
        PhenomenologyConfig()
        if phenomenology_config is None
        else phenomenology_config
    )
    validation_config = (
        DistributionValidationConfig()
        if validation_config is None
        else validation_config
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir must be empty")
    output.mkdir(parents=True, exist_ok=True)

    fit_reservoirs, component_names = _collect_reservoirs(
        split_root=fit_split,
        device=device,
        phenomenology_config=phenomenology_config,
        representations=validation_config.representations,
        reservoir_rows=validation_config.fit_reservoir_rows,
        seed=validation_config.random_seed,
        limit=fit_limit,
    )
    validation_reservoirs, validation_names = _collect_reservoirs(
        split_root=validation_split,
        device=device,
        phenomenology_config=phenomenology_config,
        representations=validation_config.representations,
        reservoir_rows=validation_config.validation_reservoir_rows,
        seed=validation_config.random_seed + 1,
        limit=validation_limit,
    )
    if component_names != validation_names:
        raise ValueError("fit and validation composition schemas differ")

    metrics_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    keys = sorted(set(fit_reservoirs) & set(validation_reservoirs))
    for index, key in enumerate(keys):
        representation, task, bucket, layer = key
        fit_reservoir = fit_reservoirs[key]
        validation_reservoir = validation_reservoirs[key]
        fit_values = fit_reservoir.matrix()
        validation_values = validation_reservoir.matrix()
        if (
            len(fit_values) < validation_config.minimum_group_rows
            or len(validation_values) < validation_config.minimum_group_rows
        ):
            continue

        for pseudocount_index, pseudocount in enumerate(
            validation_config.pseudocounts
        ):
            dirichlet, logistic_normal, metrics = distribution_diagnostics(
                fit_values,
                validation_values,
                pseudocount=pseudocount,
                simulation_rows=validation_config.simulation_rows,
                seed=(
                    validation_config.random_seed
                    + index * len(validation_config.pseudocounts)
                    + pseudocount_index
                ),
            )
            metrics_rows.append(
                {
                    "representation": representation,
                    "task": task,
                    "causal_position_bucket": bucket,
                    "layer": layer,
                    "pseudocount": pseudocount,
                    "fit_rows_seen": fit_reservoir.seen,
                    "validation_rows_seen": validation_reservoir.seen,
                    **metrics,
                }
            )
            reference_rows.append(
                {
                    "representation": representation,
                    "task": task,
                    "causal_position_bucket": bucket,
                    "layer": layer,
                    "pseudocount": pseudocount,
                    "component_names": list(component_names[representation]),
                    "alpha": dirichlet.alpha.tolist(),
                    "logistic_normal_mean": logistic_normal.mean.tolist(),
                    "logistic_normal_covariance": logistic_normal.covariance.tolist(),
                    "fit_rows_seen": fit_reservoir.seen,
                    "fit_rows_sampled": len(fit_values),
                }
            )

    _write_csv(output / "group_metrics.csv", metrics_rows)
    reference = {
        "schema": REFERENCE_SCHEMA,
        "labels_read": False,
        "fit_split": str(Path(fit_split).resolve()),
        "phenomenology_config": phenomenology_config.to_dict(),
        "validation_config": asdict(validation_config),
        "groups": reference_rows,
    }
    write_json(output / "reference.json", reference)

    summary = {
        "schema": VALIDATION_SCHEMA,
        "labels_read": False,
        "fit_split": str(Path(fit_split).resolve()),
        "validation_split": str(Path(validation_split).resolve()),
        "phenomenology_config": phenomenology_config.to_dict(),
        "validation_config": asdict(validation_config),
        "evaluated_groups": len(metrics_rows),
        "representations": _summary_rows(metrics_rows),
        "outputs": {
            "reference": "reference.json",
            "group_metrics": "group_metrics.csv",
        },
    }
    write_json(output / "summary.json", summary)
    return summary
