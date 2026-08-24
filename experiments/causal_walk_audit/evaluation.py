"""Label-late evaluation for the causal-walk hypotheses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import load_npz, read_json, write_csv, write_json
from .config import WalkAuditConfig

EVALUATION_SCHEMA = "causal-walk-audit-evaluation-v1"

SCORE_DIRECTION = {
    "order1_error": 1.0,
    "order2_error": 1.0,
    "order3_error": 1.0,
    "order2_gain": -1.0,
    "order3_gain": -1.0,
    "order2_path_gain": -1.0,
    "order3_path_gain": -1.0,
    "direct_role": 1.0,
    "anchor_js_mean": 1.0,
    "anchor_js_peak": 1.0,
    "anchor_js_excess": 1.0,
    "recoupling_depth": 1.0,
    "recoupling_failure": 1.0,
    "response_persistence": 1.0,
    "evidence_escape": -1.0,
    "lock_in": 1.0,
    "known_anchor_mass": -1.0,
    "response_base_mass": 1.0,
}


def _open_dataset(root):
    from research_dataset import open_research_dataset

    return open_research_dataset(root, device="cpu", retain_embedded_labels=True)


def _labels(store, sample, count: int) -> np.ndarray:
    if hasattr(store, "response_labels"):
        return store.response_labels(sample).cpu().numpy().astype(np.int8)
    result = np.zeros(count, dtype=np.int8)
    for start, stop in store.positive_runs(sample.sample_id, response_count=count):
        result[start:stop] = 1
    return result


def _validate_score_artifacts(dataset, score_dir: Path, manifest) -> None:
    """Validate frozen score identity before opening any hallucination label."""

    from attention_lifecycle import loaded_attention

    if manifest.get("labels_read") is not False:
        raise ValueError("causal-walk evaluation requires label-free frozen scores")
    expected = np.asarray(tuple(SCORE_DIRECTION), dtype=str)
    for row in manifest["samples"]:
        arrays = load_npz(score_dir / row["score_path"])
        sample = dataset[str(row["sample_id"])]
        with loaded_attention(sample) as attention:
            canonical = (
                attention.token_ids[attention.response_idx :]
                .cpu()
                .numpy()
                .astype(np.int32)
                .copy()
            )
        token_ids = arrays["response_token_ids"].astype(np.int32)
        valid = (
            str(arrays["schema"].item()) == "causal-walk-audit-score-v1"
            and bool(arrays["labels_included"].item()) is False
            and str(arrays["sample_id"].item()) == str(row["sample_id"])
            and str(arrays["source_id"].item()) == str(row["source_id"])
            and np.array_equal(token_ids, canonical)
            and np.array_equal(arrays["score_names"].astype(str), expected)
            and arrays["scores"].shape == (len(token_ids), len(expected))
            and arrays["token_index"].shape == (len(token_ids),)
            and arrays["valid_rows"].shape == (len(token_ids),)
        )
        if not valid:
            raise ValueError(
                f"frozen causal-walk score mismatch for sample {row['sample_id']}"
            )


def _load_samples(split_root, score_dir, manifest):
    dataset = _open_dataset(split_root)
    score_dir = Path(score_dir)
    _validate_score_artifacts(dataset, score_dir, manifest)
    label_store = dataset.prepare_evaluation_labels()
    samples = []
    for row in manifest["samples"]:
        arrays = load_npz(score_dir / row["score_path"])
        matrix = arrays["scores"].astype(np.float32)
        names = arrays["score_names"].astype(str)
        labels = _labels(
            label_store,
            dataset[str(row["sample_id"])],
            len(matrix),
        )
        selected = (arrays["valid_rows"] > 0)[:-1]
        item = {
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "task_type": str(row["task_type"]),
            "token_index": arrays["token_index"][:-1][selected] + 1,
            "labels": labels[1:][selected],
        }
        for index, name in enumerate(names):
            item[name] = matrix[:-1, index][selected]
        samples.append(item)
    return samples


def _metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _group_bootstrap(labels, score, groups, *, replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    locations = {name: np.flatnonzero(groups == name) for name in names}
    difference = []
    for _ in range(replicates):
        chosen = rng.choice(names, len(names), replace=True)
        index = np.concatenate([locations[name] for name in chosen])
        if len(np.unique(labels[index])) < 2:
            continue
        difference.append(
            score[index][labels[index] == 1].mean()
            - score[index][labels[index] == 0].mean()
        )
    if not difference:
        return {
            "difference_ci_low": float("nan"),
            "difference_ci_high": float("nan"),
        }
    return {
        "difference_ci_low": float(np.quantile(difference, 0.025)),
        "difference_ci_high": float(np.quantile(difference, 0.975)),
    }


def _circular_shift_p(
    samples,
    name: str,
    direction: float,
    *,
    replicates: int,
    seed: int,
):
    labels = np.concatenate([item["labels"] for item in samples])
    raw = np.concatenate([item[name] for item in samples])
    finite = np.isfinite(raw)
    labels = labels[finite]
    score = raw[finite] * direction
    if len(np.unique(labels)) < 2:
        return float("nan")
    observed = average_precision_score(labels, score)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        shifted, shifted_score = [], []
        for item in samples:
            current = item["labels"]
            current_score = item[name] * direction
            valid = np.isfinite(current_score)
            offset = int(rng.integers(len(current))) if len(current) else 0
            shifted.append(np.roll(current, offset)[valid])
            shifted_score.append(current_score[valid])
        current_labels = np.concatenate(shifted)
        current_score = np.concatenate(shifted_score)
        if len(np.unique(current_labels)) == 2:
            null.append(average_precision_score(current_labels, current_score))
    return float((1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1))


def _metric_rows(samples, config: WalkAuditConfig):
    tasks = sorted({item["task_type"] for item in samples})
    rows = []
    for index, (name, direction) in enumerate(SCORE_DIRECTION.items()):
        for task in ("__all__", *tasks):
            selected = (
                samples
                if task == "__all__"
                else [item for item in samples if item["task_type"] == task]
            )
            labels = np.concatenate([item["labels"] for item in selected])
            raw = np.concatenate([item[name] for item in selected])
            groups = np.concatenate(
                [
                    np.repeat(item["source_id"], len(item["labels"]))
                    for item in selected
                ]
            )
            finite = np.isfinite(raw)
            labels, raw, groups = labels[finite], raw[finite], groups[finite]
            score = raw * direction
            correct = raw[labels == 0]
            hallucination = raw[labels == 1]
            difference = (
                float(hallucination.mean() - correct.mean())
                if len(correct) and len(hallucination)
                else float("nan")
            )
            pooled = raw.std(ddof=1) if len(raw) > 1 else 0.0
            row = {
                "score": name,
                "task": task,
                "direction": direction,
                "tokens": len(labels),
                "positives": int(labels.sum()),
                "prevalence": float(labels.mean()) if len(labels) else float("nan"),
                "correct_mean": (
                    float(correct.mean()) if len(correct) else float("nan")
                ),
                "hallucination_mean": (
                    float(hallucination.mean())
                    if len(hallucination)
                    else float("nan")
                ),
                "hallucination_minus_correct": difference,
                "standardized_difference": difference / pooled if pooled > 0 else 0.0,
                "mann_whitney_p": (
                    float(
                        mannwhitneyu(
                            hallucination,
                            correct,
                            alternative="two-sided",
                        ).pvalue
                    )
                    if len(correct) and len(hallucination)
                    else float("nan")
                ),
                **_metrics(labels, score),
                **_group_bootstrap(
                    labels,
                    raw,
                    groups,
                    replicates=config.bootstrap_replicates,
                    seed=config.random_seed + index,
                ),
                "circular_shift_p": None,
            }
            if task == "__all__":
                row["circular_shift_p"] = _circular_shift_p(
                    selected,
                    name,
                    direction,
                    replicates=config.permutation_replicates,
                    seed=config.random_seed + index,
                )
            rows.append(row)
    return rows


def _matched_rows(samples):
    pairs = []
    for item in samples:
        positive = np.flatnonzero(item["labels"] == 1)
        negative = np.flatnonzero(item["labels"] == 0)
        for current in positive:
            if not len(negative):
                continue
            cost = (
                np.abs(
                    item["token_index"][negative]
                    - item["token_index"][current]
                )
                + 0.5
                * np.abs(
                    item["known_anchor_mass"][negative]
                    - item["known_anchor_mass"][current]
                )
                + 0.5
                * np.abs(
                    item["response_base_mass"][negative]
                    - item["response_base_mass"][current]
                )
            )
            pairs.append((item, current, int(negative[np.argmin(cost)])))

    rows = []
    for name in SCORE_DIRECTION:
        difference = np.asarray(
            [
                item[name][positive] - item[name][negative]
                for item, positive, negative in pairs
                if np.isfinite(item[name][positive])
                and np.isfinite(item[name][negative])
            ],
            dtype=np.float64,
        )
        std = difference.std(ddof=1) if len(difference) > 1 else 0.0
        rows.append(
            {
                "score": name,
                "pairs": len(difference),
                "hallucination_minus_matched_correct": (
                    float(difference.mean()) if len(difference) else float("nan")
                ),
                "paired_dz": float(difference.mean() / std) if std > 0 else 0.0,
            }
        )
    return rows


def _onset_rows(samples, window: int):
    names = (
        "anchor_js_peak",
        "evidence_escape",
        "response_persistence",
        "lock_in",
        "order2_gain",
        "order3_gain",
    )
    positive_fraction = [
        item["token_index"][np.flatnonzero(item["labels"] == 1)[0]]
        / max(item["token_index"][-1], 1)
        for item in samples
        if bool((item["labels"] == 1).any())
    ]
    rows = []
    for name in names:
        onset, pseudo = [], []
        for item in samples:
            positive = np.flatnonzero(item["labels"] == 1)
            if len(positive):
                current = int(positive[0])
                destination = onset
            elif positive_fraction:
                fraction = positive_fraction[len(pseudo) % len(positive_fraction)]
                current = int(round(fraction * max(len(item[name]) - 1, 0)))
                destination = pseudo
            else:
                continue
            left = item[name][max(0, current - window) : current]
            right = item[name][current : current + window]
            if len(left) and len(right):
                destination.append(float(right.mean() - left.mean()))
        rows.append(
            {
                "score": name,
                "onset_responses": len(onset),
                "onset_change": float(np.mean(onset)) if onset else float("nan"),
                "pseudo_responses": len(pseudo),
                "pseudo_change": float(np.mean(pseudo)) if pseudo else float("nan"),
                "onset_minus_pseudo": (
                    float(np.mean(onset) - np.mean(pseudo))
                    if onset and pseudo
                    else float("nan")
                ),
            }
        )
    return rows


def _decisions(metrics, matched, onset, manifest):
    metric = {
        row["score"]: row for row in metrics if row["task"] == "__all__"
    }
    paired = {row["score"]: row for row in matched}
    temporal = {row["score"]: row for row in onset}
    validation = manifest["validation"]
    evidence_anchors = {
        row["anchor_mode"] for row in manifest["samples"]
    } == {"manifest"}

    order2 = (
        validation.get("order2_gain", 0.0) > 0
        and validation.get("order2_path_gain", 0.0) > 0
    )
    order3 = (
        validation.get("order3_gain", 0.0) > 0
        and validation.get("order3_path_gain", 0.0) > 0
    )
    anchor_delta = (
        metric["anchor_js_peak"]["auprc"] - metric["direct_role"]["auprc"]
    )
    anchor_candidate = (
        anchor_delta >= 0.01
        and metric["anchor_js_excess"]["auprc"]
        > metric["anchor_js_excess"]["prevalence"]
    )
    escape_candidate = (
        metric["evidence_escape"]["difference_ci_high"] < 0
        and paired["evidence_escape"]["paired_dz"] <= -0.2
    )
    lock_candidate = (
        metric["lock_in"]["auprc"] - metric["lock_in"]["prevalence"] >= 0.01
        and temporal["lock_in"]["onset_minus_pseudo"] > 0
    )

    anchor_status = (
        "CANDIDATE_EVIDENCE_ANCHOR"
        if evidence_anchors
        else "CANDIDATE_PROMPT_CHUNK_PROXY"
    )
    return [
        {
            "hypothesis": "H1_non_markov_path_memory",
            "status": (
                "CANDIDATE_ORDER2_AND_ORDER3"
                if order2 and order3
                else "CANDIDATE_ORDER2"
                if order2
                else "INCONCLUSIVE"
            ),
            "evidence": (
                f"validation order2_gain="
                f"{validation.get('order2_gain', float('nan')):.6g}, "
                f"order2_path_gain="
                f"{validation.get('order2_path_gain', float('nan')):.6g}, "
                f"order3_gain="
                f"{validation.get('order3_gain', float('nan')):.6g}, "
                f"order3_path_gain="
                f"{validation.get('order3_path_gain', float('nan')):.6g}"
            ),
        },
        {
            "hypothesis": "H2_anchor_path_congruence",
            "status": anchor_status if anchor_candidate else "INCONCLUSIVE",
            "evidence": (
                f"anchor_js_peak minus direct_role AUPRC={anchor_delta:.6g}"
            ),
        },
        {
            "hypothesis": "H3_evidence_audit_escape",
            "status": anchor_status if escape_candidate else "INCONCLUSIVE",
            "evidence": (
                f"escape CI high="
                f"{metric['evidence_escape']['difference_ci_high']:.6g}, "
                f"matched dz={paired['evidence_escape']['paired_dz']:.6g}"
            ),
        },
        {
            "hypothesis": "H4_response_walk_lock_in",
            "status": "CANDIDATE" if lock_candidate else "INCONCLUSIVE",
            "evidence": (
                f"lock-in AUPRC gain="
                f"{metric['lock_in']['auprc'] - metric['lock_in']['prevalence']:.6g}, "
                f"onset-pseudo="
                f"{temporal['lock_in']['onset_minus_pseudo']:.6g}"
            ),
        },
        {
            "hypothesis": "H5_base_model_causal_effect",
            "status": "NOT_IMPLEMENTED_REQUIRES_NEW_MODEL_RUNS",
            "evidence": (
                "cached attention cannot establish Q/K/V, residual, MLP, "
                "or output causality"
            ),
        },
    ]


def evaluate_walk_audit(
    *,
    split_root,
    score_dir,
    output_dir,
    bootstrap_replicates=None,
    permutation_replicates=None,
):
    manifest = read_json(Path(score_dir) / "manifest.json")
    config = WalkAuditConfig(**manifest["config"])
    if bootstrap_replicates is not None:
        config = WalkAuditConfig(
            **(
                config.to_dict()
                | {"bootstrap_replicates": bootstrap_replicates}
            )
        )
    if permutation_replicates is not None:
        config = WalkAuditConfig(
            **(
                config.to_dict()
                | {"permutation_replicates": permutation_replicates}
            )
        )

    samples = _load_samples(split_root, score_dir, manifest)
    metrics = _metric_rows(samples, config)
    matched = _matched_rows(samples)
    onset = _onset_rows(samples, config.score_horizon)
    decisions = _decisions(metrics, matched, onset, manifest)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "matched_effects.csv", matched)
    write_csv(output_dir / "onset_profiles.csv", onset)
    write_csv(output_dir / "decision_table.csv", decisions)
    labels = np.concatenate([item["labels"] for item in samples])
    anchor_modes = sorted({row["anchor_mode"] for row in manifest["samples"]})
    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "artifact_validation_passed": True,
            "samples": len(samples),
            "tokens": len(labels),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "anchor_modes": anchor_modes,
            "validation": manifest["validation"],
            "decisions": decisions,
            "claim_scope": (
                "attention-derived evidence-anchor lineage and causal-walk proxy; "
                "not model-computation ancestry or causal grounding"
                if anchor_modes == ["manifest"]
                else "attention-derived prompt-chunk lineage and causal-walk proxy; "
                "not evidence grounding, model-computation ancestry, or causality"
            ),
        },
    )


__all__ = ["SCORE_DIRECTION", "evaluate_walk_audit"]
