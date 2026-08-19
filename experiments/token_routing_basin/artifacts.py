"""Strict NPZ persistence for routing-basin references and token scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_protocol import (
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)

from .detector import COMPONENT_NAMES, TokenRoutingDetector, TokenScoreTable


REFERENCE_SCHEMA = "token-routing-basin-reference-v2"
SCORE_SCHEMA = "token-routing-basin-scores-v2"


def save_reference(
    detector: TokenRoutingDetector, path, *, train_manifest_sha256: str | None = None
) -> Path:
    output = _new_file(path)
    state = detector.state()
    if train_manifest_sha256 is not None:
        if len(str(train_manifest_sha256)) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in str(train_manifest_sha256)
        ):
            raise ValueError("train_manifest_sha256 must be a SHA-256 digest")
        state["train_dataset_manifest_sha256"] = np.asarray(
            train_manifest_sha256
        )
    np.savez_compressed(output, **state)
    return output


def load_reference(path) -> TokenRoutingDetector:
    with np.load(Path(path), allow_pickle=False) as artifact:
        state = {name: artifact[name] for name in artifact.files}
    return TokenRoutingDetector.from_state(state)


def save_scores(
    table: TokenScoreTable,
    path,
    *,
    dataset_manifest_sha256: str,
    reference_sha256: str,
    audit_scope: str,
    reserved_source_ids,
    test_source_ids,
    test_sample_ids,
) -> Path:
    if audit_scope not in {"complete_split", "selected_samples"}:
        raise ValueError("audit_scope must describe a complete or selected split")
    for name, digest in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("reference_sha256", reference_sha256),
    ):
        if len(str(digest)) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in str(digest)
        ):
            raise ValueError(f"{name} must be a SHA-256 digest")
    validate_complete_token_rows(
        table.sample_id,
        table.source_id,
        table.token_index,
        table.response_length,
    )
    validate_source_audit(
        reserved_source_ids=reserved_source_ids,
        test_source_ids=test_source_ids,
        test_sample_ids=test_sample_ids,
        row_sample_ids=table.sample_id,
        row_source_ids=table.source_id,
        audit_scope=audit_scope,
    )
    rows = len(table.score)
    if table.features.shape != (rows, len(table.feature_names)):
        raise ValueError("score feature matrix has invalid shape")
    if table.controls.shape != (rows, len(table.control_names)):
        raise ValueError("score control matrix has invalid shape")
    if any(
        np.asarray(values).shape != (rows,)
        for values in (
            table.valid,
            table.winning_component,
            *table.component_score.values(),
            *table.component_raw.values(),
        )
    ):
        raise ValueError("score columns must align with token rows")

    payload = {
        "schema": np.asarray(SCORE_SCHEMA),
        "audit_scope": np.asarray(audit_scope),
        "dataset_manifest_sha256": np.asarray(dataset_manifest_sha256),
        "reference_sha256": np.asarray(reference_sha256),
        "reserved_source_ids": np.asarray(tuple(reserved_source_ids)),
        "test_source_ids": np.asarray(tuple(test_source_ids)),
        "test_sample_ids": np.asarray(tuple(test_sample_ids)),
        "online_causal_score": np.asarray(table.online_causal_score),
        "alignment": np.asarray(table.alignment),
        "threshold": np.asarray(table.threshold, dtype=np.float32),
        "score": np.asarray(table.score, dtype=np.float32),
        "valid": np.asarray(table.valid, dtype=np.bool_),
        "winning_component": np.asarray(table.winning_component),
        "sample_id": np.asarray(table.sample_id),
        "source_id": np.asarray(table.source_id),
        "task_type": np.asarray(table.task_type),
        "data_source": np.asarray(table.data_source),
        "token_index": np.asarray(table.token_index, dtype=np.int32),
        "response_length": np.asarray(table.response_length, dtype=np.int32),
        "feature_names": np.asarray(table.feature_names),
        "features": np.asarray(table.features, dtype=np.float32),
        "control_names": np.asarray(table.control_names),
        "controls": np.asarray(table.controls, dtype=np.float32),
    }
    for name in COMPONENT_NAMES:
        payload[f"component_score/{name}"] = np.asarray(
            table.component_score[name], dtype=np.float32
        )
        payload[f"component_raw/{name}"] = np.asarray(
            table.component_raw[name], dtype=np.float32
        )
    output = _new_file(path)
    np.savez_compressed(output, **payload)
    return output


def load_scores(path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as artifact:
        rows = {name: artifact[name] for name in artifact.files}
    if _scalar_text(rows, "schema") != SCORE_SCHEMA:
        raise ValueError("unsupported token routing score schema")
    sha256_text(rows, "dataset_manifest_sha256")
    sha256_text(rows, "reference_sha256")
    validate_complete_token_rows(
        rows.get("sample_id"),
        rows.get("source_id"),
        rows.get("token_index"),
        rows.get("response_length"),
    )
    validate_source_audit(
        reserved_source_ids=rows.get("reserved_source_ids"),
        test_source_ids=rows.get("test_source_ids"),
        test_sample_ids=rows.get("test_sample_ids"),
        row_sample_ids=rows.get("sample_id"),
        row_source_ids=rows.get("source_id"),
        audit_scope=_scalar_text(rows, "audit_scope"),
    )
    score = np.asarray(rows.get("score"))
    valid = np.asarray(rows.get("valid"))
    if score.ndim != 1 or valid.shape != score.shape or valid.dtype.kind != "b":
        raise ValueError("score artifact has invalid score/valid columns")
    if not np.isfinite(score[valid]).all() or not np.isnan(score[~valid]).all():
        raise ValueError("score artifact validity mask disagrees with scores")
    feature_names = _string_vector(rows, "feature_names")
    control_names = _string_vector(rows, "control_names")
    features = np.asarray(rows.get("features"))
    controls = np.asarray(rows.get("controls"))
    if features.shape != (len(score), len(feature_names)) or controls.shape != (
        len(score),
        len(control_names),
    ):
        raise ValueError("score artifact feature rows do not align")
    if not np.isfinite(features).all() or not np.isfinite(controls).all():
        raise ValueError("score artifact features and controls must be finite")
    if any(
        np.asarray(rows.get(name)).shape != score.shape
        for name in ("task_type", "data_source")
    ):
        raise ValueError("score artifact metadata rows do not align")
    for name in COMPONENT_NAMES:
        for prefix in ("component_score", "component_raw"):
            values = np.asarray(rows.get(f"{prefix}/{name}"))
            if values.shape != score.shape:
                raise ValueError(f"score artifact misses aligned {prefix}/{name}")
            if np.isinf(values).any():
                raise ValueError(f"score artifact has infinite {prefix}/{name}")
    winning = np.asarray(rows.get("winning_component"))
    if winning.shape != score.shape or winning.dtype.kind not in {"U", "S"}:
        raise ValueError("score artifact has invalid winning components")
    if not set(map(str, winning.tolist())).issubset(
        {*COMPONENT_NAMES, "invalid"}
    ):
        raise ValueError("score artifact has unknown winning components")
    causal = np.asarray(rows.get("online_causal_score"))
    if causal.ndim != 0 or causal.dtype.kind != "b" or not causal.item():
        raise ValueError("token routing artifact must declare online causal scoring")
    if _scalar_text(rows, "alignment") != "post_token_query_at_same_position":
        raise ValueError("score artifact has unsupported attention alignment")
    threshold = np.asarray(rows.get("threshold"))
    if threshold.ndim != 0 or not np.isfinite(threshold.item()) or threshold.item() <= 0:
        raise ValueError("score artifact has an invalid threshold")
    return rows


def _new_file(path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _scalar_text(mapping, name):
    value = np.asarray(mapping.get(name))
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"artifact field {name!r} must be scalar text")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _string_vector(mapping, name):
    value = np.asarray(mapping.get(name))
    if (
        value.ndim != 1
        or not len(value)
        or value.dtype.kind not in {"U", "S"}
        or len(set(map(str, value.tolist()))) != len(value)
    ):
        raise ValueError(f"artifact field {name!r} must be a unique string vector")
    return tuple(map(str, value.tolist()))
