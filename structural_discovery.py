"""Label-open diagnostics for selecting a graph representation before modelling.

This module is deliberately not part of unsupervised training. It opens labels
only to test whether a label-free representation contains reproducible signal.
Cross-validation is grouped by RAGTruth source_id to prevent source leakage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evidence_graph import EvidenceGraphConfig, RESPONSE_STATE_NAMES, build_evidence_graph
from graph_features import (
    BASIC_FEATURE_NAMES,
    RESPONSE_FEATURE_NAMES,
    basic_structural_features,
    response_graph_features,
)
from research_dataset import ResearchDataset


REPRESENTATION_MODES = (
    "basic12",
    "response32",
    "evidence22",
    "response32+evidence22",
)


def response_representation(sample, mode: str, *, mass_cover=0.80, relay_discount=0.85):
    """Return one label-free vector per response token and its feature names."""
    if mode == "basic12":
        values = (
            basic_structural_features(sample.attention(), sample.relation_edges())
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        return values, BASIC_FEATURE_NAMES
    if mode == "response32":
        return response_graph_features(sample), RESPONSE_FEATURE_NAMES

    evidence = (
        build_evidence_graph(
            sample.attention(),
            EvidenceGraphConfig(
                mass_cover=float(mass_cover), relay_discount=float(relay_discount)
            ),
        )
        .response_state.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if mode == "evidence22":
        return evidence, RESPONSE_STATE_NAMES
    if mode == "response32+evidence22":
        base = response_graph_features(sample)
        return (
            np.concatenate((base, evidence), axis=1),
            RESPONSE_FEATURE_NAMES + RESPONSE_STATE_NAMES,
        )
    raise ValueError(f"unknown representation mode: {mode}")


@dataclass(frozen=True)
class DiscoveryResult:
    representation: str
    feature_names: tuple[str, ...]
    samples: int
    tokens: int
    positives: int
    folds: int
    oof_auroc: float
    oof_average_precision: float
    positive_fraction: float
    univariate: list[dict[str, float | str]]

    def to_dict(self):
        return {
            "representation": self.representation,
            "feature_names": list(self.feature_names),
            "samples": self.samples,
            "tokens": self.tokens,
            "positives": self.positives,
            "folds": self.folds,
            "oof_auroc": self.oof_auroc,
            "oof_average_precision": self.oof_average_precision,
            "positive_fraction": self.positive_fraction,
            "univariate": self.univariate,
            "scope": (
                "label-open source-grouped representation diagnostic; "
                "not an unsupervised result"
            ),
        }


class StructuralDiscovery:
    """Compare candidate label-free node representations with grouped OOF probes."""

    def __init__(
        self,
        split_root,
        *,
        device="cpu",
        verify_hashes=False,
        mass_cover=0.80,
        relay_discount=0.85,
        seed=0,
    ):
        self.dataset = ResearchDataset(
            split_root, device=device, verify_hashes=verify_hashes
        )
        self.mass_cover = float(mass_cover)
        self.relay_discount = float(relay_discount)
        self.seed = int(seed)

    def collect(self, mode, *, max_samples=None, task_type=None, generator_model=None):
        if mode not in REPRESENTATION_MODES:
            raise ValueError(f"mode must be one of {REPRESENTATION_MODES}")
        labels = self.dataset.labels()
        features, targets, groups = [], [], []
        sample_count = 0
        feature_names = None
        for sample_id in self.dataset.sample_ids:
            sample = self.dataset[sample_id]
            if task_type is not None and sample.task_type != task_type:
                continue
            if generator_model is not None and sample.generator_model != generator_model:
                continue
            matrix, names = response_representation(
                sample,
                mode,
                mass_cover=self.mass_cover,
                relay_discount=self.relay_discount,
            )
            y = labels.response_labels(sample).detach().cpu().numpy().astype(np.int64)
            if len(matrix) != len(y):
                raise ValueError("representation and labels do not align")
            features.append(matrix)
            targets.append(y)
            groups.append(np.full(len(y), sample.source_id, dtype=object))
            feature_names = tuple(names)
            sample_count += 1
            sample.release_attention()
            if max_samples is not None and sample_count >= int(max_samples):
                break
        if not features:
            raise ValueError("no matching samples")
        return (
            np.concatenate(features),
            np.concatenate(targets),
            np.concatenate(groups),
            feature_names,
            sample_count,
        )

    def evaluate(
        self,
        mode,
        *,
        folds=5,
        max_samples=None,
        task_type=None,
        generator_model=None,
    ) -> DiscoveryResult:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler

        X, y, groups, names, sample_count = self.collect(
            mode,
            max_samples=max_samples,
            task_type=task_type,
            generator_model=generator_model,
        )
        if np.unique(y).size < 2:
            raise ValueError("diagnostic requires both token labels")
        unique_groups = np.unique(groups)
        actual_folds = min(int(folds), len(unique_groups))
        if actual_folds < 2:
            raise ValueError("at least two source groups are required")

        splitter = GroupKFold(n_splits=actual_folds)
        score = np.full(len(y), np.nan, dtype=np.float64)
        for train, test in splitter.split(X, y, groups):
            if np.unique(y[train]).size < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=self.seed,
                ),
            )
            model.fit(X[train], y[train])
            score[test] = model.predict_proba(X[test])[:, 1]
        valid = np.isfinite(score)
        if np.unique(y[valid]).size < 2:
            raise ValueError("grouped folds produced no valid two-class predictions")

        univariate = []
        for index, name in enumerate(names):
            values = np.asarray(X[:, index], dtype=np.float64)
            finite = np.isfinite(values)
            if finite.sum() == 0 or np.unique(y[finite]).size < 2:
                auc = oriented = 0.5
            else:
                auc = float(roc_auc_score(y[finite], values[finite]))
                oriented = max(auc, 1.0 - auc)
            univariate.append(
                {
                    "feature": name,
                    "auroc": auc,
                    "orientation_free_auroc": oriented,
                }
            )
        univariate.sort(
            key=lambda row: float(row["orientation_free_auroc"]), reverse=True
        )

        return DiscoveryResult(
            representation=mode,
            feature_names=tuple(names),
            samples=sample_count,
            tokens=len(y),
            positives=int(y.sum()),
            folds=actual_folds,
            oof_auroc=float(roc_auc_score(y[valid], score[valid])),
            oof_average_precision=float(average_precision_score(y[valid], score[valid])),
            positive_fraction=float(y[valid].mean()),
            univariate=univariate,
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", default=",".join(REPRESENTATION_MODES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mass-cover", type=float, default=0.80)
    parser.add_argument("--relay-discount", type=float, default=0.85)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--task-type")
    parser.add_argument("--generator-model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    diagnostic = StructuralDiscovery(
        args.canonical_split,
        device=args.device,
        mass_cover=args.mass_cover,
        relay_discount=args.relay_discount,
        seed=args.seed,
    )
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    reports = [
        diagnostic.evaluate(
            mode,
            folds=args.folds,
            max_samples=args.max_samples,
            task_type=args.task_type,
            generator_model=args.generator_model,
        ).to_dict()
        for mode in modes
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"reports": reports}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"reports": reports}, indent=2))
    return reports


if __name__ == "__main__":
    main()
