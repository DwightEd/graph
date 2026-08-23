"""Association and structure-null statistics for frozen relation scores."""

from __future__ import annotations

import numpy as np

from .bounded_ensemble import EnsembleAUPRC
from .config import EvaluationConfig
from .evaluation_data import (
    EvaluationBundle,
    aligned_relation,
    grouped_relation,
)
from .features import RELATION_NAMES
from .statistics import (
    benjamini_hochberg,
    circular_shift_p_value,
    grouped_bootstrap_delta,
    grouped_metric_interval,
)


def _ensemble_p_value(ensemble: EnsembleAUPRC, relation: int) -> float:
    observed = ensemble.real[relation]
    if not np.isfinite(observed):
        return float("nan")
    null = ensemble.null[:, relation]
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _adjust(rows: list[dict[str, object]], p_name: str, q_name: str) -> None:
    selected = [
        index
        for index, row in enumerate(rows)
        if row.get(p_name) is not None and np.isfinite(row[p_name])
    ]
    if not selected:
        return
    adjusted = benjamini_hochberg(
        np.asarray([rows[index][p_name] for index in selected], dtype=np.float64)
    )
    for index, q_value in zip(selected, adjusted):
        rows[index][q_name] = float(q_value)


def relation_rows(
    bundle: EvaluationBundle,
    endpoint_relations: set[str],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    samples = bundle.samples
    rows = []
    for relation_index, name in enumerate(RELATION_NAMES):
        labels_by_source, real_by_source = grouped_relation(
            samples, lambda sample: sample.relation, relation_index
        )
        labels_by_sample, scores_by_sample = [], []
        for sample in samples:
            labels, values = aligned_relation(
                sample.relation,
                sample.labels,
                sample.eligible,
                relation_index,
            )
            labels_by_sample.append(labels)
            scores_by_sample.append(values)

        labels = np.concatenate(labels_by_source)
        row: dict[str, object] = {
            "relation": name,
            "tokens": len(labels),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            **grouped_metric_interval(
                labels_by_source,
                real_by_source,
                replicates=config.bootstrap_replicates,
                seed=config.random_seed + relation_index,
            ),
            "association_shift_p": circular_shift_p_value(
                labels_by_sample,
                scores_by_sample,
                replicates=config.permutation_replicates,
                seed=config.random_seed + relation_index,
            ),
            "endpoint_null_valid": None,
            "endpoint_null_p": None,
            "layer_shuffle_p": None,
        }
        if name in endpoint_relations:
            _, endpoint_by_source = grouped_relation(
                samples,
                lambda sample: sample.endpoint_null,
                relation_index,
            )
            endpoint_delta = grouped_bootstrap_delta(
                labels_by_source,
                real_by_source,
                endpoint_by_source,
                replicates=config.bootstrap_replicates,
                seed=config.random_seed + 100 + relation_index,
            )
            changed_mean = float(
                np.mean([sample.endpoint_changed_fraction_mean for sample in samples])
            )
            changed_min = float(
                np.min([sample.endpoint_changed_fraction_min for sample in samples])
            )
            positive_samples = [
                sample
                for sample in samples
                if bool(sample.labels[1:][sample.eligible[1:]].any())
            ]
            correct_samples = [
                sample
                for sample in samples
                if not bool(sample.labels[1:][sample.eligible[1:]].any())
            ]
            positive_changed = (
                float(
                    np.mean(
                        [
                            sample.endpoint_changed_fraction_mean
                            for sample in positive_samples
                        ]
                    )
                )
                if positive_samples
                else float("nan")
            )
            correct_changed = (
                float(
                    np.mean(
                        [
                            sample.endpoint_changed_fraction_mean
                            for sample in correct_samples
                        ]
                    )
                )
                if correct_samples
                else float("nan")
            )
            row.update(
                endpoint_changed_fraction_mean=changed_mean,
                endpoint_changed_fraction_min=changed_min,
                endpoint_changed_fraction_positive_mean=positive_changed,
                endpoint_changed_fraction_correct_mean=correct_changed,
                endpoint_changed_fraction_label_gap=positive_changed - correct_changed,
                endpoint_null_valid=changed_min
                >= config.endpoint_minimum_changed_fraction,
                endpoint_null_p=_ensemble_p_value(
                    bundle.endpoint_auprc, relation_index
                ),
                **{f"endpoint_{key}": value for key, value in endpoint_delta.items()},
            )
            final_labels, final_by_source = grouped_relation(
                samples, lambda sample: sample.final_relation, relation_index
            )
            _, shuffled_by_source = grouped_relation(
                samples,
                lambda sample: sample.layer_shuffle,
                relation_index,
            )
            layer_delta = grouped_bootstrap_delta(
                final_labels,
                final_by_source,
                shuffled_by_source,
                replicates=config.bootstrap_replicates,
                seed=config.random_seed + 200 + relation_index,
            )
            row.update(
                layer_shuffle_p=_ensemble_p_value(bundle.layer_auprc, relation_index),
                **{f"layer_{key}": value for key, value in layer_delta.items()},
            )
        rows.append(row)

    _adjust(rows, "association_shift_p", "association_shift_q")
    _adjust(rows, "endpoint_null_p", "endpoint_null_q")
    _adjust(rows, "layer_shuffle_p", "layer_shuffle_q")
    return rows
