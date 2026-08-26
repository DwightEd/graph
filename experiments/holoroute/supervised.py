"""Label-aware linear probe for auditing routing-fingerprint quality.

This module is deliberately separate from the unsupervised detector. It uses
train token labels to measure how much correctness information is present in the
same nuisance-controlled node features. Test labels remain sealed until the
standard evaluation command is run.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm.auto import tqdm

from .artifacts import SCORE_SCHEMA, load_npz, save_npz, sha256
from .detection import CONDITION_NAMES, Reservoir, token_conditions
from .features import build_node_features
from .graph import build_graph
from .pipeline import ScoreRows, load_method, merge_rows, select_samples

PROBE_SCHEMA = "routing-fingerprint-linear-probe-v1"
MODEL_TYPE = "routing_fingerprint_linear_probe"
RESIDUAL_NAMES = ("linear_probe_logit",)


@dataclass(frozen=True)
class LinearProbe:
    coefficient: np.ndarray
    intercept: float
    fit_tokens: int
    positive_tokens: int
    negative_tokens: int

    def decision_function(self, feature: np.ndarray) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float32)
        return (value @ self.coefficient + self.intercept).astype(np.float32)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "probe_coefficient": self.coefficient.astype(np.float32),
            "probe_intercept": np.asarray(self.intercept, dtype=np.float32),
            "probe_fit_tokens": np.asarray(self.fit_tokens, dtype=np.int64),
            "probe_positive_tokens": np.asarray(self.positive_tokens, dtype=np.int64),
            "probe_negative_tokens": np.asarray(self.negative_tokens, dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "LinearProbe":
        return cls(
            coefficient=np.asarray(arrays["probe_coefficient"], dtype=np.float32),
            intercept=float(np.asarray(arrays["probe_intercept"]).item()),
            fit_tokens=int(np.asarray(arrays["probe_fit_tokens"]).item()),
            positive_tokens=int(np.asarray(arrays["probe_positive_tokens"]).item()),
            negative_tokens=int(np.asarray(arrays["probe_negative_tokens"]).item()),
        )


def fit_linear_probe(
    feature: np.ndarray,
    label: np.ndarray,
    seed: int,
) -> LinearProbe:
    value = np.asarray(feature, dtype=np.float32)
    target = np.asarray(label, dtype=np.int64).reshape(-1)
    if np.unique(target).size != 2:
        raise ValueError("the supervised probe needs both token classes")

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-4,
        average=True,
        random_state=seed,
    )
    classifier.fit(value, target)
    return LinearProbe(
        coefficient=classifier.coef_[0].astype(np.float32),
        intercept=float(classifier.intercept_[0]),
        fit_tokens=len(target),
        positive_tokens=int(target.sum()),
        negative_tokens=int((target == 0).sum()),
    )


def response_labels(label_store, sample) -> np.ndarray:
    return label_store.response_labels(sample).cpu().numpy().astype(np.int64)


def fit_supervised_probe(
    dataset,
    checkpoint_path,
    reference_path,
    probe_path,
    task: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    config, reference, _ = load_method(checkpoint_path, reference_path)
    sample_ids = select_samples(dataset, task, limit)
    label_store = dataset.prepare_evaluation_labels()

    positive = Reservoir(config.detection.reservoir_rows // 2, config.detection.seed + 11)
    negative = Reservoir(config.detection.reservoir_rows // 2, config.detection.seed + 12)

    for sample_id in tqdm(sample_ids, desc="fit labeled linear probe", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            features = build_node_features(graph, config.feature)
            condition = token_conditions(graph)
            sample_task = np.repeat(graph.task_type, graph.response_count)
            standardized = reference.standardize(
                features.node.cpu().numpy().astype(np.float32),
                condition,
                sample_task,
            )
            label = response_labels(label_store, sample)
            if bool((label == 1).any()):
                positive.add(feature=standardized[label == 1])
            if bool((label == 0).any()):
                negative.add(feature=standardized[label == 0])
        finally:
            sample.release_attention()

    positive_rows = positive.values()["feature"]
    negative_rows = negative.values()["feature"]
    feature = np.concatenate((positive_rows, negative_rows), axis=0)
    label = np.concatenate(
        (
            np.ones(len(positive_rows), dtype=np.int64),
            np.zeros(len(negative_rows), dtype=np.int64),
        )
    )
    probe = fit_linear_probe(feature, label, config.detection.seed)

    save_npz(
        probe_path,
        schema=np.asarray(PROBE_SCHEMA),
        model_type=np.asarray(MODEL_TYPE),
        labels_used=np.asarray(True),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        task=np.asarray(task),
        **probe.arrays(),
    )
    return {
        "probe": str(Path(probe_path).resolve()),
        "samples": len(sample_ids),
        "fit_tokens": probe.fit_tokens,
        "positive_tokens": probe.positive_tokens,
        "negative_tokens": probe.negative_tokens,
        "labels_read": True,
    }


def load_probe(probe_path, checkpoint_path, reference_path) -> LinearProbe:
    arrays = load_npz(probe_path)
    if str(arrays["schema"].item()) != PROBE_SCHEMA:
        raise ValueError("unsupported routing-fingerprint probe")
    if sha256(checkpoint_path) != str(arrays["checkpoint_sha256"].item()):
        raise ValueError("probe and feature checkpoint differ")
    if sha256(reference_path) != str(arrays["reference_sha256"].item()):
        raise ValueError("probe and feature reference differ")
    return LinearProbe.from_arrays(arrays)


def probability_from_logit(logit: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logit, dtype=np.float32), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def make_probe_rows(graph, feature, reference, probe: LinearProbe) -> ScoreRows:
    condition = token_conditions(graph)
    task = np.repeat(graph.task_type, graph.response_count)
    standardized = reference.standardize(
        feature.node.cpu().numpy().astype(np.float32),
        condition,
        task,
    )
    logit = probe.decision_function(standardized)
    tokens = graph.response_count
    return ScoreRows(
        sample_id=np.repeat(graph.sample_id, tokens),
        source_id=np.repeat(graph.source_id, tokens),
        task_type=task,
        token_index=np.arange(tokens, dtype=np.int32),
        response_length=np.full(tokens, tokens, dtype=np.int32),
        response_token_id=graph.response_token_ids.cpu().numpy().astype(np.int64),
        score=probability_from_logit(logit),
        residual=logit[:, None],
        standardized=logit[:, None],
        coverage=np.ones((tokens, 1), dtype=np.float32),
        condition=condition,
    )


def score_supervised_probe(
    dataset,
    checkpoint_path,
    reference_path,
    probe_path,
    output_path,
    task: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    config, reference, _ = load_method(checkpoint_path, reference_path)
    probe = load_probe(probe_path, checkpoint_path, reference_path)
    sample_ids = select_samples(dataset, task, limit)
    rows: list[ScoreRows] = []

    for sample_id in tqdm(sample_ids, desc="score labeled linear probe", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            feature = build_node_features(graph, config.feature)
            rows.append(make_probe_rows(graph, feature, reference, probe))
        finally:
            sample.release_attention()

    arrays = merge_rows(rows)
    save_npz(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        model_type=np.asarray(MODEL_TYPE),
        labels_included=np.asarray(False),
        labels_used_to_fit_model=np.asarray(True),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        probe_path=np.asarray(str(Path(probe_path).resolve())),
        probe_sha256=np.asarray(sha256(probe_path)),
        residual_names=np.asarray(RESIDUAL_NAMES),
        condition_names=np.asarray(CONDITION_NAMES),
        **arrays,
    )
    return {
        "scores": str(Path(output_path).resolve()),
        "probe": str(Path(probe_path).resolve()),
        "samples": len(sample_ids),
        "tokens": len(arrays["score"]),
        "labels_read": False,
    }
