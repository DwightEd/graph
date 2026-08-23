"""Label-aware tests run only after structure scores are frozen."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attention_lifecycle import loaded_attention
from research_dataset import open_research_dataset

from .artifacts import load_npz, read_json, write_csv, write_json
from .config import EvaluationConfig
from .features import RELATION_NAMES
from .joint_form import compare_joint_forms
from .statistics import (
    benjamini_hochberg,
    binary_metrics,
    circular_shift_p_value,
    grouped_bootstrap_delta,
    grouped_metric_interval,
)
from .token_classes import content_token_mask


@dataclass(frozen=True)
class AlignedRows:
    score: np.ndarray
    labels: np.ndarray
    token_index: np.ndarray


def align_query_to_next_token(
    score: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
) -> AlignedRows:
    """Align a post-token query at t with the token predicted at t + 1."""

    score = np.asarray(score)
    labels = np.asarray(labels)
    eligible = np.asarray(eligible, dtype=bool)
    if score.shape[0] != labels.shape[0] or labels.shape != eligible.shape:
        raise ValueError("score, labels, and eligible mask must share token rows")
    selected = eligible[1:]
    return AlignedRows(
        score=score[:-1][selected],
        labels=labels[1:][selected],
        token_index=np.arange(1, len(labels), dtype=np.int32)[selected],
    )


def pre_onset_slope(
    score: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    window: int,
) -> float | None:
    """Slope of risk scores available before the first positive token is emitted."""

    positive = np.flatnonzero(np.asarray(labels) == 1)
    if not len(positive):
        return None
    onset = int(positive[0])
    aligned = align_query_to_next_token(score, labels, eligible)
    values = aligned.score[aligned.token_index <= onset][-window:]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


@dataclass(frozen=True)
class FrozenSample:
    sample_id: str
    source_id: str
    labels: np.ndarray
    eligible: np.ndarray
    relation: np.ndarray
    final_relation: np.ndarray
    endpoint_null: np.ndarray
    layer_shuffle: np.ndarray
    endpoint_changed_fraction: float


def _aligned_matrix(
    values: np.ndarray, labels: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(eligible, dtype=bool)[1:]
    return np.asarray(labels, dtype=np.int8)[1:][selected], values[:-1][selected]


def _grouped_relation(
    samples: list[FrozenSample], getter, relation: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    labels_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    scores_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in samples:
        labels, values = _aligned_matrix(
            getter(sample), sample.labels, sample.eligible
        )
        labels_by_source[sample.source_id].append(labels)
        scores_by_source[sample.source_id].append(values[:, relation])
    source_ids = sorted(labels_by_source)
    return (
        [np.concatenate(labels_by_source[source]) for source in source_ids],
        [np.concatenate(scores_by_source[source]) for source in source_ids],
    )


def _ensemble_p_value(
    samples: list[FrozenSample], real_getter, null_getter, relation: int
) -> float:
    labels = []
    real = []
    for sample in samples:
        current_labels, current_real = _aligned_matrix(
            real_getter(sample), sample.labels, sample.eligible
        )
        labels.append(current_labels)
        real.append(current_real[:, relation])
    labels = np.concatenate(labels)
    observed = binary_metrics(labels, np.concatenate(real))["auprc"]
    replicates = min(len(null_getter(sample)) for sample in samples)
    null_metrics = []
    for replicate in range(replicates):
        null_scores = []
        for sample in samples:
            _, current = _aligned_matrix(
                null_getter(sample)[replicate], sample.labels, sample.eligible
            )
            null_scores.append(current[:, relation])
        null_metrics.append(
            binary_metrics(labels, np.concatenate(null_scores))["auprc"]
        )
    return float(
        (1 + np.sum(np.asarray(null_metrics) >= observed)) / (replicates + 1)
    )


def _adjust(rows: list[dict[str, object]], p_name: str, q_name: str) -> None:
    selected = [index for index, row in enumerate(rows) if row.get(p_name) is not None]
    adjusted = benjamini_hochberg(
        np.asarray([rows[index][p_name] for index in selected], dtype=np.float64)
    )
    for index, q_value in zip(selected, adjusted):
        rows[index][q_name] = float(q_value)


def _relation_rows(
    samples: list[FrozenSample],
    endpoint_relations: set[str],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    rows = []
    for relation_index, name in enumerate(RELATION_NAMES):
        labels_by_source, real_by_source = _grouped_relation(
            samples, lambda sample: sample.relation, relation_index
        )
        labels_by_sample = []
        scores_by_sample = []
        for sample in samples:
            labels, values = _aligned_matrix(
                sample.relation, sample.labels, sample.eligible
            )
            labels_by_sample.append(labels)
            scores_by_sample.append(values[:, relation_index])

        labels = np.concatenate(labels_by_source)
        row: dict[str, object] = {
            "relation": name,
            "tokens": int(len(labels)),
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
            _, endpoint_by_source = _grouped_relation(
                samples,
                lambda sample: sample.endpoint_null.mean(axis=0),
                relation_index,
            )
            endpoint_delta = grouped_bootstrap_delta(
                labels_by_source,
                real_by_source,
                endpoint_by_source,
                replicates=config.bootstrap_replicates,
                seed=config.random_seed + 100 + relation_index,
            )
            changed = float(
                np.mean([sample.endpoint_changed_fraction for sample in samples])
            )
            row.update(
                endpoint_changed_fraction=changed,
                endpoint_null_valid=changed
                >= config.endpoint_minimum_changed_fraction,
                endpoint_null_p=_ensemble_p_value(
                    samples,
                    lambda sample: sample.relation,
                    lambda sample: sample.endpoint_null,
                    relation_index,
                ),
                **{f"endpoint_{key}": value for key, value in endpoint_delta.items()},
            )

            final_labels, final_by_source = _grouped_relation(
                samples, lambda sample: sample.final_relation, relation_index
            )
            _, shuffled_by_source = _grouped_relation(
                samples,
                lambda sample: sample.layer_shuffle.mean(axis=0),
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
                layer_shuffle_p=_ensemble_p_value(
                    samples,
                    lambda sample: sample.final_relation,
                    lambda sample: sample.layer_shuffle,
                    relation_index,
                ),
                **{f"layer_{key}": value for key, value in layer_delta.items()},
            )
        rows.append(row)

    _adjust(rows, "association_shift_p", "association_shift_q")
    _adjust(rows, "endpoint_null_p", "endpoint_null_q")
    _adjust(rows, "layer_shuffle_p", "layer_shuffle_q")
    return rows


def _slope_before_cutoff(
    score: np.ndarray, eligible: np.ndarray, cutoff: int, window: int
) -> float | None:
    target = np.arange(1, len(score))[np.asarray(eligible, dtype=bool)[1:]]
    values = score[:-1][np.asarray(eligible, dtype=bool)[1:]][target <= cutoff][-window:]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _post_onset_change(
    score: np.ndarray,
    eligible: np.ndarray,
    onset: int,
    window: int,
) -> float | None:
    target = np.arange(1, len(score))[np.asarray(eligible, dtype=bool)[1:]]
    values = score[:-1][np.asarray(eligible, dtype=bool)[1:]][target > onset]
    if len(values) < 2 * window:
        return None
    return float(values[-window:].mean() - values[:window].mean())


def _effect(values: list[float]) -> tuple[int, float | None, float | None]:
    if not values:
        return 0, None, None
    array = np.asarray(values, dtype=np.float64)
    deviation = array.std(ddof=1) if len(array) > 1 else 0.0
    return (
        len(array),
        float(array.mean()),
        float(array.mean() / deviation) if deviation > 0 else None,
    )


def _temporal_rows(
    samples: list[FrozenSample], config: EvaluationConfig
) -> list[dict[str, object]]:
    hallucinating = [sample for sample in samples if sample.labels.any()]
    correct = [sample for sample in samples if not sample.labels.any()]
    fractions = [
        float(np.flatnonzero(sample.labels)[0] / max(len(sample.labels) - 1, 1))
        for sample in hallucinating
    ]
    rows = []
    for relation, name in enumerate(RELATION_NAMES):
        pre = [
            value
            for sample in hallucinating
            if (
                value := pre_onset_slope(
                    sample.relation[:, relation],
                    sample.labels,
                    sample.eligible,
                    window=config.onset_window,
                )
            )
            is not None
        ]
        pseudo = []
        for index, sample in enumerate(correct):
            if not fractions:
                break
            cutoff = int(round(fractions[index % len(fractions)] * (len(sample.labels) - 1)))
            value = _slope_before_cutoff(
                sample.relation[:, relation],
                sample.eligible,
                cutoff,
                config.onset_window,
            )
            if value is not None:
                pseudo.append(value)
        lockin = []
        for sample in hallucinating:
            onset = int(np.flatnonzero(sample.labels)[0])
            value = _post_onset_change(
                sample.relation[:, relation],
                sample.eligible,
                onset,
                config.onset_window,
            )
            if value is not None:
                lockin.append(value)
        pre_count, pre_mean, pre_dz = _effect(pre)
        pseudo_count, pseudo_mean, _ = _effect(pseudo)
        lockin_count, lockin_mean, lockin_dz = _effect(lockin)
        rows.append(
            {
                "relation": name,
                "pre_onset_responses": pre_count,
                "pre_onset_slope": pre_mean,
                "pre_onset_paired_dz": pre_dz,
                "pseudo_onset_responses": pseudo_count,
                "pseudo_onset_slope": pseudo_mean,
                "pre_minus_pseudo_slope": None
                if pre_mean is None or pseudo_mean is None
                else pre_mean - pseudo_mean,
                "lockin_responses": lockin_count,
                "late_minus_early": lockin_mean,
                "lockin_paired_dz": lockin_dz,
            }
        )
    return rows


def _joint_rows(
    samples: list[FrozenSample], config: EvaluationConfig
) -> list[dict[str, object]]:
    labels, relations, groups = [], [], []
    for sample in samples:
        current_labels, current_relations = _aligned_matrix(
            sample.relation, sample.labels, sample.eligible
        )
        labels.append(current_labels)
        relations.append(current_relations)
        groups.extend([sample.source_id] * len(current_labels))
    labels = np.concatenate(labels)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    positive_groups = np.unique(groups[labels == 1])
    negative_groups = np.unique(groups[labels == 0])
    folds = min(
        config.grouped_cv_folds,
        len(unique_groups),
        len(positive_groups),
        len(negative_groups),
    )
    if folds < 2:
        return []
    direct = tuple(
        RELATION_NAMES.index(name)
        for name in (
            "direct_role",
            "endpoint_concentration",
            "head_fracture",
            "censoring_control",
        )
    )
    return compare_joint_forms(
        labels,
        np.concatenate(relations),
        groups,
        direct_columns=direct,
        folds=folds,
        seed=config.random_seed,
    )


def _gate(
    audit: str,
    status: str,
    question: str,
    authorized: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "audit": audit,
        "status": status,
        "question": question,
        "authorized_model_component": authorized,
        "evidence": evidence,
    }


def _decisions(
    relation_rows: list[dict[str, object]],
    joint_rows: list[dict[str, object]],
    *,
    scope: str,
    positive_responses: int,
    config: EvaluationConfig,
) -> list[dict[str, str]]:
    questions = (
        ("A0", "对齐与标签泄漏", "subsequent_audits"),
        ("A1", "直接 prompt/response role 是否有信号", "linear_lookback"),
        ("A2", "response endpoint identity 是否超越粗粒度 lag", "exact_token_graph"),
        ("A3", "head fracture 是否有独立增益", "head_resolved_module"),
        ("A4", "layer 顺序是否重要", "ordered_recurrent_update"),
        ("A5", "多跳 lineage 是否优于 direct/one-hop", "multi_hop_message_passing"),
        ("A6", "哪种 aggregation 形式必要", "typed_aggregator"),
        ("A7", "错误后是否存在 response-history lock-in", "gated_temporal_state"),
        ("A8", "结构变化是否领先首个错误", "online_change_point"),
        ("A9", "非线性交互是否优于加性形式", "neural_joint_model"),
        ("A10", "结构 proxy 是否具有原模型因果作用", "causal_claim"),
    )
    decisions = [
        _gate(
            "A0",
            "PASS_PROTOCOL",
            questions[0][1],
            questions[0][2],
            "scores were frozen label-free; query t is evaluated against token t+1",
        )
    ]
    if scope == "smoke":
        decisions.extend(
            _gate(audit, "NOT_EVALUATED_SMOKE", question, "none", "run is below confirmation size")
            for audit, question, _ in questions[1:]
        )
        return decisions
    if positive_responses < config.minimum_positive_responses:
        decisions.extend(
            _gate(
                audit,
                "INCONCLUSIVE_LOW_POWER",
                question,
                "none",
                "too few hallucination-containing responses",
            )
            for audit, question, _ in questions[1:]
        )
        return decisions

    by_relation = {row["relation"]: row for row in relation_rows}
    direct = by_relation["direct_role"]
    a1_pass = (
        direct["association_shift_q"] < 0.05
        and direct["auprc"] - direct["prevalence"] >= 0.02
        and direct["auprc_ci_low"] > direct["prevalence"]
    )
    decisions.append(
        _gate(
            "A1",
            "PASS" if a1_pass else "FAIL",
            questions[1][1],
            questions[1][2] if a1_pass else "none",
            "direct_role AUPRC versus prevalence and circular-shift q",
        )
    )
    lineage = by_relation["lineage_margin"]
    a2_pass = (
        lineage["endpoint_null_valid"]
        and lineage["endpoint_auprc_delta"] >= 0.01
        and lineage["endpoint_auprc_delta_ci_low"] > 0
        and lineage["endpoint_null_q"] < 0.05
    )
    decisions.append(
        _gate(
            "A2",
            "PASS" if a2_pass else "FAIL",
            questions[2][1],
            questions[2][2] if a2_pass else "none",
            "lineage_margin real versus constrained response-endpoint null",
        )
    )
    decisions.append(
        _gate("A3", "INCONCLUSIVE_CONTROL_MISSING", questions[3][1], "none", "head mean conditional control is not yet implemented")
    )
    a4_pass = (
        lineage["layer_auprc_delta"] >= 0.01
        and lineage["layer_auprc_delta_ci_low"] > 0
        and lineage["layer_shuffle_q"] < 0.05
    )
    decisions.append(
        _gate(
            "A4",
            "PASS" if a4_pass else "FAIL",
            questions[4][1],
            questions[4][2] if a4_pass else "none",
            "final lineage state versus layer-order permutations",
        )
    )
    decisions.extend(
        (
            _gate("A5", "INCONCLUSIVE_DEPTH_CONTROL_MISSING", questions[5][1], "none", "explicit direct/one-hop/two-hop/full ablation is required"),
            _gate("A6", "NOT_IMPLEMENTED_WEIGHT_NULL", questions[6][1], "none", "fixed-endpoint weight shuffle is required"),
            _gate("A7", "EXPLORATORY_ONLY", questions[7][1], "none", "temporal effect is reported but not a persistence gate"),
            _gate("A8", "NOT_IMPLEMENTED_CHANGE_POINT", questions[8][1], "none", "discovery-frozen threshold and confirmation FPR are required"),
        )
    )
    if joint_rows:
        additive = next(row for row in joint_rows if row["model"] == "additive")
        interaction = next(row for row in joint_rows if row["model"] == "interaction")
        fold_delta = np.asarray(interaction["fold_auprc"]) - np.asarray(
            additive["fold_auprc"]
        )
        a9_pass = (
            interaction["delta_auprc_from_previous"] >= 0.01
            and int((fold_delta > 0).sum()) >= int(np.ceil(0.8 * len(fold_delta)))
        )
        status = "CANDIDATE_PASS_CV" if a9_pass else "FAIL"
    else:
        status = "INCONCLUSIVE_GROUPS"
        a9_pass = False
    decisions.append(
        _gate(
            "A9",
            status,
            questions[9][1],
            questions[9][2] if a9_pass else "none",
            "grouped CV interaction versus additive logistic readout",
        )
    )
    decisions.append(
        _gate("A10", "NOT_AVAILABLE_FROM_ATTENTION_CACHE", questions[10][1], "none", "requires matched interventions in the base LLM")
    )
    return decisions


class StructureEvaluator:
    """Open labels after score freezing and produce model-authorization gates."""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = EvaluationConfig() if config is None else config

    def run(
        self,
        *,
        split_root,
        score_dir,
        output_dir,
        tokenizer_path=None,
    ) -> None:
        score_dir = Path(score_dir)
        manifest = read_json(score_dir / "manifest.json")
        if manifest["labels_read"] is not False:
            raise ValueError("evaluation requires label-free frozen scores")
        if manifest["evaluation_alignment"] != "query_t_to_response_token_t_plus_1":
            raise ValueError("unexpected score/label alignment")

        tokenizer = None
        if tokenizer_path is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True
            )
        dataset = open_research_dataset(split_root, device="cpu")
        label_store = dataset.prepare_evaluation_labels()
        samples = []
        for row in manifest["samples"]:
            sample = dataset[row["sample_id"]]
            with loaded_attention(sample):
                labels = (
                    label_store.response_labels(sample)
                    .cpu()
                    .numpy()
                    .astype(np.int8)
                )
            arrays = load_npz(score_dir / row["score_path"])
            token_ids = arrays["response_token_ids"]
            eligible = (
                np.ones(len(token_ids), dtype=bool)
                if tokenizer is None
                else content_token_mask(token_ids, tokenizer)
            )
            if len(labels) != len(token_ids):
                raise ValueError("label and frozen response token counts differ")
            samples.append(
                FrozenSample(
                    sample_id=str(row["sample_id"]),
                    source_id=str(row["source_id"]),
                    labels=labels,
                    eligible=eligible,
                    relation=arrays["relation_scores"].astype(np.float32),
                    final_relation=arrays["final_relation_scores"].astype(np.float32),
                    endpoint_null=arrays[
                        "response_endpoint_null_relation_scores"
                    ].astype(np.float32),
                    layer_shuffle=arrays["layer_shuffle_relation_scores"].astype(
                        np.float32
                    ),
                    endpoint_changed_fraction=float(
                        arrays["response_endpoint_null_changed_fraction"].mean()
                    ),
                )
            )

        relation_rows = _relation_rows(
            samples,
            set(manifest["response_endpoint_null_relations"]),
            self.config,
        )
        temporal_rows = _temporal_rows(samples, self.config)
        joint_rows = _joint_rows(samples, self.config)
        positive_responses = sum(bool(sample.labels.any()) for sample in samples)
        scope = (
            "confirmation"
            if len(samples) >= self.config.minimum_confirmation_samples
            else "smoke"
        )
        decisions = _decisions(
            relation_rows,
            joint_rows,
            scope=scope,
            positive_responses=positive_responses,
            config=self.config,
        )

        output_dir = Path(output_dir)
        write_csv(output_dir / "relation_metrics.csv", relation_rows)
        write_csv(output_dir / "temporal_audit.csv", temporal_rows)
        write_csv(output_dir / "joint_form_cv.csv", joint_rows)
        write_csv(output_dir / "decision_table.csv", decisions)
        labels = np.concatenate([sample.labels[1:][sample.eligible[1:]] for sample in samples])
        write_json(
            output_dir / "evaluation.json",
            {
                "schema": "non-neural-structure-evaluation-v1",
                "labels_read": True,
                "scope": scope,
                "token_scope": "content_alphanumeric"
                if tokenizer is not None
                else "all_tokens",
                "samples": len(samples),
                "source_groups": len({sample.source_id for sample in samples}),
                "positive_responses": positive_responses,
                "evaluated_tokens": int(len(labels)),
                "positive_tokens": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "relation_metrics": relation_rows,
                "temporal_audit": temporal_rows,
                "joint_form_cv": joint_rows,
                "decisions": decisions,
                "claim_scope": manifest["claim_scope"],
                "limitations": [
                    "all prompt tokens share one type; evidence/question/system are not separated",
                    "raw attention omits values, output projection, residual stream, and FFN",
                    "teacher-forced cached rows are routing proxies, not autoregressive interventions",
                    "A10 requires new base-model runs and is not inferred from cached attention",
                ],
            },
        )
