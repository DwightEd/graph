"""Label-blind spectral representation extraction and anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .representations import SpectralConfig, spectral_token_representation


REPRESENTATION_SCHEMA = "spectral-feasibility-token-representation-v1"
SCORE_SCHEMA = "spectral-feasibility-robust-mahalanobis-v1"


@dataclass(frozen=True)
class RobustReference:
    center: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    trim_fraction: float
    ridge: float

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("features do not match the robust reference dimension")
        standardized = (values - self.center) / self.scale
        squared = np.einsum(
            "ni,ij,nj->n",
            standardized,
            self.precision,
            standardized,
            optimize=True,
        )
        return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)


def collect_representations(
    dataset,
    *,
    config: SpectralConfig | None = None,
    sample_ids=None,
    limit=None,
):
    """Collect spectral response-token vectors without opening labels."""

    config = SpectralConfig() if config is None else config
    config.validate()
    ids = list(dataset.sample_ids if sample_ids is None else map(str, sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        ids = ids[:limit]
    if not ids:
        raise ValueError("no samples selected")

    feature_blocks = []
    sample_column = []
    source_column = []
    token_column = []
    task_column = []
    data_source_column = []
    generator_column = []
    response_length_column = []
    feature_names = None

    for sample_id in ids:
        sample = dataset[sample_id]
        try:
            features, names = spectral_token_representation(sample, config)
            if feature_names is None:
                feature_names = tuple(names)
            elif tuple(names) != feature_names:
                raise RuntimeError("spectral feature names changed across samples")
            count = features.shape[0]
            feature_blocks.append(features)
            sample_column.append(np.full(count, str(sample.sample_id), dtype=str))
            source_column.append(np.full(count, str(sample.source_id), dtype=str))
            token_column.append(np.arange(count, dtype=np.int32))
            task_column.append(np.full(count, str(sample.task_type), dtype=str))
            data_source_column.append(np.full(count, str(sample.data_source), dtype=str))
            generator_column.append(np.full(count, str(sample.generator_model), dtype=str))
            response_length_column.append(np.full(count, count, dtype=np.int32))
        finally:
            sample.release_attention()

    return {
        "schema": REPRESENTATION_SCHEMA,
        "features": np.concatenate(feature_blocks, axis=0),
        "feature_names": np.asarray(feature_names, dtype=str),
        "sample_id": np.concatenate(sample_column),
        "source_id": np.concatenate(source_column),
        "token_index": np.concatenate(token_column),
        "task_type": np.concatenate(task_column),
        "data_source": np.concatenate(data_source_column),
        "generator_model": np.concatenate(generator_column),
        "response_length": np.concatenate(response_length_column),
        "sample_count": len(ids),
        "heat_scales": np.asarray(config.heat_scales, dtype=np.float32),
        "svd_bands": np.asarray(config.svd_bands, dtype=np.int32),
        "block_rows": int(config.block_rows),
    }


def save_representation_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            key: np.asarray(value)
            for key, value in artifact.items()
            if key != "sample_count"
        },
        sample_count=np.asarray(int(artifact["sample_count"]), dtype=np.int32),
    )
    return str(path)


def load_representation_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        required = {
            "schema",
            "features",
            "feature_names",
            "sample_id",
            "source_id",
            "token_index",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"representation artifact is missing {sorted(missing)}")
        schema = str(np.asarray(arrays["schema"]).item())
        if schema != REPRESENTATION_SCHEMA:
            raise ValueError("unsupported spectral representation artifact")
        output = {name: arrays[name].copy() for name in arrays.files}
    if output["features"].ndim != 2:
        raise ValueError("representation features must be a matrix")
    if output["features"].shape[1] != len(output["feature_names"]):
        raise ValueError("feature_names do not match representation dimension")
    return output


def fit_robust_reference(
    features,
    *,
    trim_fraction=0.90,
    ridge=1e-3,
):
    """Fit a label-free trimmed robust Gaussian reference in spectral space."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("robust reference requires at least three feature vectors")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    if not 0.5 <= float(trim_fraction) < 1.0:
        raise ValueError("trim_fraction must be in [0.5,1)")
    if float(ridge) <= 0.0:
        raise ValueError("ridge must be positive")

    center = np.median(values, axis=0)
    absolute = np.abs(values - center)
    mad = 1.4826 * np.median(absolute, axis=0)
    std = values.std(axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    standardized = (values - center) / scale
    radius = np.square(standardized).sum(axis=1)
    cutoff = np.quantile(radius, float(trim_fraction))
    inliers = standardized[radius <= cutoff]
    if len(inliers) < 2:
        raise ValueError("trimmed reference retained fewer than two tokens")
    covariance = np.cov(inliers, rowvar=False)
    covariance = np.atleast_2d(covariance).astype(np.float64, copy=False)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += float(ridge) * np.eye(covariance.shape[0], dtype=np.float64)
    precision = np.linalg.pinv(covariance, hermitian=True)
    return RobustReference(
        center=center,
        scale=scale,
        precision=precision,
        trim_fraction=float(trim_fraction),
        ridge=float(ridge),
    )


def score_representation_artifacts(
    train_path,
    test_path,
    output_path,
    *,
    trim_fraction=0.90,
    ridge=1e-3,
):
    """Fit on unlabeled train vectors and score unlabeled test vectors."""

    train = load_representation_artifact(train_path)
    test = load_representation_artifact(test_path)
    if not np.array_equal(train["feature_names"], test["feature_names"]):
        raise ValueError("train/test spectral feature definitions differ")
    reference = fit_robust_reference(
        train["features"],
        trim_fraction=trim_fraction,
        ridge=ridge,
    )
    score = reference.score(test["features"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        features=test["features"].astype(np.float32, copy=False),
        feature_names=test["feature_names"],
        sample_id=test["sample_id"],
        source_id=test["source_id"],
        token_index=test["token_index"].astype(np.int32, copy=False),
        task_type=test.get("task_type", np.asarray([], dtype=str)),
        data_source=test.get("data_source", np.asarray([], dtype=str)),
        generator_model=test.get("generator_model", np.asarray([], dtype=str)),
        score=score,
        reference_center=reference.center.astype(np.float32),
        reference_scale=reference.scale.astype(np.float32),
        reference_precision=reference.precision.astype(np.float32),
        trim_fraction=np.asarray(reference.trim_fraction, dtype=np.float32),
        ridge=np.asarray(reference.ridge, dtype=np.float32),
    )
    return {
        "output": str(output_path),
        "train_tokens": int(len(train["features"])),
        "test_tokens": int(len(test["features"])),
        "feature_dim": int(test["features"].shape[1]),
    }


def load_score_artifact(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        required = {
            "schema",
            "features",
            "feature_names",
            "sample_id",
            "token_index",
            "score",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"score artifact is missing {sorted(missing)}")
        if str(np.asarray(arrays["schema"]).item()) != SCORE_SCHEMA:
            raise ValueError("unsupported spectral score artifact")
        return {name: arrays[name].copy() for name in arrays.files}


def _label_store_for_evaluation(dataset):
    """Open labels only in the explicit evaluation stage.

    Formal caches seal embedded labels until every sample has passed through the
    research dataset. Canonical sidecar labels are immediately available here.
    """

    try:
        return dataset.labels()
    except RuntimeError as error:
        if "every attention sample" not in str(error):
            raise
        for sample_id in dataset.sample_ids:
            sample = dataset[sample_id]
            sample.attention()
            sample.release_attention()
        return dataset.labels()


def evaluate_score_artifact(dataset, score_path, output_path):
    """Post-hoc evaluation after spectral features and anomaly scores are frozen."""

    artifact = load_score_artifact(score_path)
    labels = _label_store_for_evaluation(dataset)
    label_cache = {}
    y = np.empty(len(artifact["score"]), dtype=np.int64)
    for index, (sample_id, token_index) in enumerate(
        zip(artifact["sample_id"], artifact["token_index"], strict=True)
    ):
        sample_id = str(sample_id)
        if sample_id not in label_cache:
            sample = dataset[sample_id]
            label_cache[sample_id] = labels.response_labels(sample).cpu().numpy()
            sample.release_attention()
        y[index] = int(label_cache[sample_id][int(token_index)])

    prevalence = float(y.mean())
    if np.unique(y).size < 2:
        raise ValueError("evaluation requires both normal and hallucination tokens")
    score = artifact["score"].astype(np.float64, copy=False)
    metrics = {
        "tokens": int(len(y)),
        "positive_tokens": int(y.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "auprc_random_baseline": prevalence,
    }

    feature_metrics = {}
    for column, name in enumerate(artifact["feature_names"].tolist()):
        value = artifact["features"][:, column].astype(np.float64, copy=False)
        auc = float(roc_auc_score(y, value))
        feature_metrics[str(name)] = {
            "auroc": auc,
            "separability": max(auc, 1.0 - auc),
            "median_normal": float(np.median(value[y == 0])),
            "median_hallucination": float(np.median(value[y == 1])),
        }

    report = {
        "schema": "spectral-feasibility-evaluation-v1",
        "metrics": metrics,
        "feature_metrics": feature_metrics,
        "labels_used_during": "posthoc_evaluation_only",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
