"""Join frozen full-stream scores to character annotations only at evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from control_graph.metrics import binary_detection_metrics
from route_graph.data import read_jsonl, text_digest


class RouteEvaluator:
    def __init__(
        self,
        scores_path: Path,
        labels_path: Path,
        output_path: Path,
        bootstrap: int = 1000,
        seed: int = 20260911,
    ) -> None:
        self.scores_path, self.labels_path, self.output_path = (
            scores_path,
            labels_path,
            output_path,
        )
        self.bootstrap, self.seed = bootstrap, seed

    def run(self) -> dict:
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")
        rows = read_jsonl(self.scores_path)
        annotations = read_jsonl(self.labels_path)
        labels_by_id = {str(row["id"]): row for row in annotations}
        if len(labels_by_id) != len(annotations):
            raise ValueError("duplicate annotation response IDs")
        grouped = defaultdict(list)
        score_names = set(rows[0]["scores"])
        for row in rows:
            if (
                row["schema"] != "route-graph/score@1"
                or set(row["scores"]) != score_names
            ):
                raise ValueError("score schema or columns differ")
            if not np.isfinite(list(row["scores"].values())).all():
                raise ValueError("scores must be finite")
            if (
                type(row["reference_active_features"]) is not int
                or row["reference_active_features"] < 0
            ):
                raise ValueError(
                    "reference_active_features must be a nonnegative integer"
                )
            grouped[row["response_id"]].append(row)
        ordered, labels, onsets, first_errors, continuations = [], [], [], [], []
        for response_id, tokens in grouped.items():
            tokens.sort(key=lambda row: row["token_index"])
            count = tokens[0]["token_count"]
            if [row["token_index"] for row in tokens] != list(range(count)) or any(
                row["token_count"] != count for row in tokens
            ):
                raise ValueError(
                    "evaluation requires the complete token stream, including token zero"
                )
            if response_id not in labels_by_id:
                raise ValueError(f"missing annotations for response {response_id}")
            annotation = labels_by_id[response_id]
            text = annotation["response"]
            for span in annotation["labels"]:
                if not 0 <= span["start"] < span["end"] <= len(text):
                    raise ValueError("annotation character interval outside response")
            previous, seen_error = 0, False
            for row in tokens:
                if row["source_id"] != str(annotation["source_id"]) or row[
                    "response_sha256"
                ] != text_digest(text):
                    raise ValueError(
                        "score/annotation source or response digest mismatch"
                    )
                left, right = row["char_span"]
                if not 0 <= left < right <= len(text):
                    raise ValueError("invalid token character interval")
                label = int(
                    any(
                        left < span["end"] and right > span["start"]
                        for span in annotation["labels"]
                    )
                )
                ordered.append(row)
                labels.append(label)
                onsets.append(bool(label and not previous))
                first_errors.append(bool(label and not seen_error))
                continuations.append(bool(label and previous))
                previous = label
                seen_error = seen_error or bool(label)
        y = np.asarray(labels)
        sources = np.array([row["source_id"] for row in ordered])
        active = np.array([row["reference_active_features"] > 0 for row in ordered])
        scores = {
            name: np.array([row["scores"][name] for row in ordered])
            for name in sorted(score_names)
        }
        masks = {
            "all_tokens": np.ones(len(y), dtype=bool),
            "span_onsets": (y == 0) | onsets,
            "first_error": (y == 0) | first_errors,
            "continuations": (y == 0) | continuations,
        }
        subsets = {}
        for name, mask in masks.items():
            metrics = {}
            for score_name, values in scores.items():
                metrics[score_name] = (
                    binary_detection_metrics(
                        y[mask],
                        values[mask],
                        sources[mask],
                        bootstrap=self.bootstrap,
                        seed=self.seed,
                        source_balanced=True,
                    )
                    if len(np.unique(y[mask])) == 2
                    else None
                )
            subsets[name] = {
                "tokens": int(mask.sum()),
                "positives": int(y[mask].sum()),
                "sources": len(np.unique(sources[mask])),
                "inactive_reference_tokens": int((mask & ~active).sum()),
                "metrics": metrics,
            }
        paired = {}
        for baseline in (
            "null",
            "signal",
            "observed",
            "negative_margin",
            "entropy",
            "position",
        ):
            if baseline in scores and len(np.unique(y)) == 2:
                paired[f"residual_minus_{baseline}"] = _paired_gain(
                    y,
                    scores["residual"],
                    scores[baseline],
                    sources,
                    self.bootstrap,
                    self.seed,
                )
        report = {
            "schema": "route-graph/evaluation@1",
            "labels_used_stage": "evaluation_only",
            "point_weighting": "equal_total_weight_per_source",
            "bootstrap_unit": "source_id",
            "bootstrap": self.bootstrap,
            "seed": self.seed,
            "subsets": subsets,
            "paired_auroc_gain": paired,
            "reference_coverage": {
                "active_tokens": int(active.sum()),
                "inactive_tokens": int((~active).sum()),
                "active_token_fraction": float(active.mean()),
                "sources_with_inactive_tokens": len(np.unique(sources[~active])),
                "inactive_score_meaning": "zero score means no reference variation, not trusted normal",
            },
            "generated_token_in_candidates": float(
                np.mean([row["token_id"] in row["candidate_ids"] for row in ordered])
            ),
            "correct_answer_candidate_coverage": None,
            "scope": "full scored responses; onset subsets reuse all normal tokens; no alarm threshold",
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return report


def _paired_gain(labels, first, second, sources, bootstrap, seed) -> dict:
    """Source-weighted AUROC difference, jointly resampled for both methods."""
    groups, inverse, counts = np.unique(
        sources, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    by_source = [np.flatnonzero(inverse == index) for index in range(len(groups))]

    def difference(indices):
        return float(
            roc_auc_score(
                labels[indices], first[indices], sample_weight=weights[indices]
            )
            - roc_auc_score(
                labels[indices], second[indices], sample_weight=weights[indices]
            )
        )

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        indices = np.concatenate(
            [by_source[index] for index in rng.integers(len(groups), size=len(groups))]
        )
        if len(np.unique(labels[indices])) == 2:
            estimates.append(difference(indices))
    return {
        "estimate": difference(np.arange(len(labels))),
        "confidence_interval": np.quantile(estimates, [0.025, 0.975]).tolist()
        if estimates
        else None,
        "valid_bootstrap_replicates": len(estimates),
    }
