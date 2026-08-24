"""Strict bridge to the pre-registered causal RR residual score.

The bridge is an experiment interface, not part of the path mechanism. It
freezes both input score artifacts before parsing, aligns complete token rows,
and writes enough dataset, source-split, and byte provenance for independent
post-hoc evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from experiment_protocol import (
    FrozenFile,
    scalar_text,
    sha256_text,
    validate_complete_token_rows,
    validate_source_audit,
)
from experiments.rr_signal_audit.artifacts import (
    load_score_artifact as load_rr_score_artifact,
    verify_score_provenance as verify_rr_score_provenance,
)

from .artifacts import (
    load_score_artifact as load_path_score_artifact,
    verify_score_provenance as verify_path_score_provenance,
)


ALLOWED_RR_SCORE = "received_topk.causal.residual_tail"
HYBRID_SCHEMA = "causal-typed-path-rr-cauchy-hybrid-v1"

_ROW_TEXT_FIELDS = (
    "sample_id",
    "source_id",
    "task_type",
    "data_source",
    "generator_model",
)
_ROW_INTEGER_FIELDS = ("token_index", "response_length")
_ROW_FLOAT_FIELDS = (
    "score",
    "path_only",
    "rr_only",
    "fusion_stat",
    "global_p_value",
)
_PATH_GROUP_FIELDS = (
    "fit_group_id",
    "channel_calibration_group_id",
    "fusion_calibration_group_id",
    "calibration_group_id",
)
_DIGEST_FIELDS = (
    "dataset_manifest_sha256",
    "path_score_sha256",
    "rr_score_sha256",
    "path_reference_sha256",
    "rr_reference_sha256",
    "path_train_dataset_manifest_sha256",
    "rr_train_dataset_manifest_sha256",
)
_PATH_FIELDS = (
    "path_score_path",
    "rr_score_path",
    "path_reference_path",
    "rr_reference_path",
)
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "labels_included",
        "primary_detector",
        "online_causal_score",
        "rr_score_name",
        "audit_scope",
        "test_group_id",
        "test_sample_id",
        "rr_fit_group_id",
        "rr_calibration_group_id",
        "rr_test_group_id",
        "rr_test_sample_id",
        "rr_audit_scope",
        *_ROW_TEXT_FIELDS,
        *_ROW_INTEGER_FIELDS,
        *_ROW_FLOAT_FIELDS,
        *_PATH_GROUP_FIELDS,
        *_DIGEST_FIELDS,
        *_PATH_FIELDS,
    }
)
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "label",
        "labels",
        "token_label",
        "token_labels",
        "response_label",
        "response_labels",
        "hallucination_label",
        "hallucination_labels",
    }
)
_FORBIDDEN_LABEL_ALIASES = frozenset(
    {
        "gold",
        "gold_label",
        "gold_labels",
        "ground_truth",
        "ground_truth_label",
        "ground_truth_labels",
        "is_hallucination",
        "hallucinated",
        *_FORBIDDEN_LABEL_FIELDS,
    }
)


def _key_rows(artifact) -> list[tuple[str, int]]:
    return list(
        zip(
            np.asarray(artifact["sample_id"], dtype=str).tolist(),
            np.asarray(artifact["token_index"], dtype=np.int64).tolist(),
            strict=True,
        )
    )


def _verify_rr_reference_frozen(rr_artifact) -> dict:
    """Close the legacy RR verifier's reference-file TOCTOU window."""

    reference_path = Path(scalar_text(rr_artifact, "reference_path")).resolve()
    reference_file = FrozenFile.capture(reference_path)
    if reference_file.sha256 != sha256_text(rr_artifact, "reference_sha256"):
        raise ValueError("RR score reference digest differs from its artifact")
    _reject_label_payload(rr_artifact, description="RR score")
    reference = verify_rr_score_provenance(rr_artifact)
    _reject_label_payload(reference, description="RR reference")
    reference_file.verify(reference_path)
    return reference


def _reject_label_payload(mapping, *, description: str) -> None:
    """Reject common canonical-label aliases from legacy RR artifacts."""

    forbidden = []
    for name in mapping:
        normalized = str(name).strip().lower()
        if (
            normalized in _FORBIDDEN_LABEL_ALIASES
            or normalized.startswith("label_")
            or normalized.endswith("_label")
            or normalized.endswith("_labels")
        ):
            forbidden.append(str(name))
    if forbidden:
        raise ValueError(
            f"{description} contains forbidden label fields: {sorted(forbidden)}"
        )


def _scalar_boolean(mapping, name: str) -> bool:
    if name not in mapping:
        raise ValueError(f"hybrid artifact is missing field {name!r}")
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.bool_):
        raise ValueError(f"hybrid field {name!r} must be a scalar boolean")
    return bool(value.item())


def _text_vector(mapping, name: str) -> np.ndarray:
    if name not in mapping:
        raise ValueError(f"hybrid artifact is missing field {name!r}")
    values = np.asarray(mapping[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"hybrid field {name!r} must be a text vector")
    text = values.astype(str, copy=False)
    if (
        len(text) < 1
        or any(not value.strip() for value in text.tolist())
        or len(set(text.tolist())) != len(text)
    ):
        raise ValueError(f"hybrid field {name!r} has invalid identifiers")
    return text


def _same_identifier_set(left, right) -> bool:
    return set(map(str, np.asarray(left, dtype=str).tolist())) == set(
        map(str, np.asarray(right, dtype=str).tolist())
    )


def _cauchy_two(
    p_left: np.ndarray,
    p_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = np.finfo(np.float64).eps
    values = []
    for probability in (p_left, p_right):
        clipped = np.clip(
            np.asarray(probability, dtype=np.float64),
            epsilon,
            1.0 - epsilon,
        )
        values.append(np.tan(np.pi * (0.5 - clipped)))
    statistic = 0.5 * (values[0] + values[1])
    probability = 0.5 - np.arctan(statistic) / np.pi
    return statistic.astype(np.float32), probability.astype(np.float32)


def _validate_hybrid_arrays(artifact) -> None:
    missing = _REQUIRED_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"RR hybrid artifact misses fields: {sorted(missing)}")
    unexpected = set(artifact).difference(_REQUIRED_FIELDS)
    if unexpected:
        raise ValueError(
            "RR hybrid artifact contains unsupported fields: "
            f"{sorted(unexpected)}"
        )
    if scalar_text(artifact, "schema") != HYBRID_SCHEMA:
        raise ValueError("unsupported causal typed-path/RR hybrid schema")
    if _scalar_boolean(artifact, "labels_included"):
        raise ValueError("RR hybrid artifacts must set labels_included=false")
    forbidden = _FORBIDDEN_LABEL_FIELDS.intersection(artifact)
    if forbidden:
        raise ValueError(
            "RR hybrid artifact contains forbidden labels: " f"{sorted(forbidden)}"
        )
    if scalar_text(artifact, "primary_detector") != "score":
        raise ValueError("RR hybrid primary_detector must be 'score'")
    if not _scalar_boolean(artifact, "online_causal_score"):
        raise ValueError("RR hybrid must be explicitly online causal")
    if scalar_text(artifact, "rr_score_name") != ALLOWED_RR_SCORE:
        raise ValueError("RR hybrid contains a non-allowlisted RR score")

    score = np.asarray(artifact["score"])
    if score.ndim != 1 or len(score) < 1:
        raise ValueError("RR hybrid score must be a non-empty vector")
    rows = len(score)
    for name in _ROW_TEXT_FIELDS:
        values = np.asarray(artifact[name])
        if (
            values.ndim != 1
            or len(values) != rows
            or values.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(f"RR hybrid row field {name!r} must contain text")
    for name in _ROW_INTEGER_FIELDS:
        values = np.asarray(artifact[name])
        if (
            values.ndim != 1
            or len(values) != rows
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(f"RR hybrid row field {name!r} must contain integers")
    for name in _ROW_FLOAT_FIELDS:
        values = np.asarray(artifact[name])
        if (
            values.ndim != 1
            or len(values) != rows
            or not np.issubdtype(values.dtype, np.floating)
            or not bool(np.isfinite(values).all())
        ):
            raise ValueError(f"RR hybrid row field {name!r} must be finite floats")
    if bool((score < 0.0).any()):
        raise ValueError("RR hybrid primary score must be non-negative")
    if bool(
        (np.asarray(artifact["path_only"]) < 0.0).any()
        or (np.asarray(artifact["rr_only"]) < 0.0).any()
    ):
        raise ValueError("RR hybrid component scores must be non-negative")
    probability = np.asarray(artifact["global_p_value"], dtype=np.float32)
    if bool(((probability <= 0.0) | (probability > 1.0)).any()):
        raise ValueError("RR hybrid global_p_value must lie in (0, 1]")

    validate_complete_token_rows(
        artifact["sample_id"],
        artifact["source_id"],
        artifact["token_index"],
        artifact["response_length"],
    )

    path_groups = {
        name: _text_vector(artifact, name) for name in _PATH_GROUP_FIELDS
    }
    if set(path_groups["calibration_group_id"].tolist()) != (
        set(path_groups["channel_calibration_group_id"].tolist())
        | set(path_groups["fusion_calibration_group_id"].tolist())
    ):
        raise ValueError(
            "hybrid calibration_group_id must equal the channel/fusion union"
        )
    separated = (
        set(path_groups["fit_group_id"].tolist()),
        set(path_groups["channel_calibration_group_id"].tolist()),
        set(path_groups["fusion_calibration_group_id"].tolist()),
    )
    if any(
        separated[left] & separated[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("hybrid path fit/channel/fusion groups overlap")

    test_groups = _text_vector(artifact, "test_group_id")
    test_samples = _text_vector(artifact, "test_sample_id")
    audit_scope = scalar_text(artifact, "audit_scope")
    validate_source_audit(
        reserved_source_ids=np.concatenate(
            (
                path_groups["fit_group_id"],
                path_groups["channel_calibration_group_id"],
                path_groups["fusion_calibration_group_id"],
            )
        ),
        test_source_ids=test_groups,
        test_sample_ids=test_samples,
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=audit_scope,
    )

    rr_fit = _text_vector(artifact, "rr_fit_group_id")
    rr_calibration = _text_vector(artifact, "rr_calibration_group_id")
    rr_test_groups = _text_vector(artifact, "rr_test_group_id")
    rr_test_samples = _text_vector(artifact, "rr_test_sample_id")
    rr_scope = scalar_text(artifact, "rr_audit_scope")
    if set(rr_fit.tolist()) & set(rr_calibration.tolist()):
        raise ValueError("hybrid RR fit/calibration groups overlap")
    validate_source_audit(
        reserved_source_ids=np.concatenate((rr_fit, rr_calibration)),
        test_source_ids=rr_test_groups,
        test_sample_ids=rr_test_samples,
        row_sample_ids=artifact["sample_id"],
        row_source_ids=artifact["source_id"],
        audit_scope=rr_scope,
    )
    if (
        audit_scope != rr_scope
        or not _same_identifier_set(test_groups, rr_test_groups)
        or not _same_identifier_set(test_samples, rr_test_samples)
    ):
        raise ValueError("path and RR held-out scopes differ in the hybrid")

    for name in _DIGEST_FIELDS:
        sha256_text(artifact, name)
    for name in _PATH_FIELDS:
        value = Path(scalar_text(artifact, name))
        if not value.is_absolute():
            raise ValueError(f"hybrid provenance path {name!r} must be absolute")

    path_only = np.asarray(artifact["path_only"], dtype=np.float64)
    rr_only = np.asarray(artifact["rr_only"], dtype=np.float64)
    expected_stat, expected_probability = _cauchy_two(
        np.exp(-path_only),
        np.exp(-rr_only),
    )
    expected_score = -np.log(
        np.maximum(expected_probability, np.finfo(np.float32).tiny)
    ).astype(np.float32)
    for name, expected in (
        ("fusion_stat", expected_stat),
        ("global_p_value", expected_probability),
        ("score", expected_score),
    ):
        if not bool(
            np.allclose(
                np.asarray(artifact[name], dtype=np.float32),
                expected,
                rtol=2e-6,
                atol=2e-7,
            )
        ):
            raise ValueError(f"RR hybrid field {name!r} disagrees with its inputs")


def load_rr_hybrid(path) -> dict[str, np.ndarray]:
    """Load and structurally validate one evaluation-alignable hybrid artifact."""

    with np.load(path, allow_pickle=False) as arrays:
        artifact = {name: arrays[name].copy() for name in arrays.files}
    _validate_hybrid_arrays(artifact)
    return artifact


def verify_rr_hybrid_provenance(artifact) -> tuple[dict, dict]:
    """Re-open both component scores and verify their fitted references."""

    _validate_hybrid_arrays(artifact)
    path_score_file = FrozenFile(
        Path(scalar_text(artifact, "path_score_path")),
        sha256_text(artifact, "path_score_sha256"),
    )
    rr_score_file = FrozenFile(
        Path(scalar_text(artifact, "rr_score_path")),
        sha256_text(artifact, "rr_score_sha256"),
    )
    path_score_file.verify(path_score_file.path)
    rr_score_file.verify(rr_score_file.path)
    path_artifact = load_path_score_artifact(path_score_file.path)
    rr_artifact = load_rr_score_artifact(rr_score_file.path)
    path_reference = verify_path_score_provenance(path_artifact)
    rr_reference = _verify_rr_reference_frozen(rr_artifact)
    path_score_file.verify(path_score_file.path)
    rr_score_file.verify(rr_score_file.path)

    hybrid_manifest = sha256_text(artifact, "dataset_manifest_sha256")
    if (
        sha256_text(path_artifact, "dataset_manifest_sha256") != hybrid_manifest
        or sha256_text(rr_artifact, "dataset_manifest_sha256") != hybrid_manifest
    ):
        raise ValueError("hybrid dataset manifest differs from a component score")

    hybrid_keys = _key_rows(artifact)
    path_keys = _key_rows(path_artifact)
    rr_keys = _key_rows(rr_artifact)
    if hybrid_keys != path_keys or set(path_keys) != set(rr_keys):
        raise ValueError("hybrid token rows differ from its component scores")
    rr_index = {key: index for index, key in enumerate(rr_keys)}
    if len(rr_index) != len(rr_keys):
        raise ValueError("hybrid RR component contains duplicate token rows")
    reorder = np.asarray([rr_index[key] for key in path_keys], dtype=np.int64)
    for name in (*_ROW_TEXT_FIELDS, *_ROW_INTEGER_FIELDS):
        hybrid_values = np.asarray(artifact[name])
        path_values = np.asarray(path_artifact[name])
        rr_values = np.asarray(rr_artifact[name])[reorder]
        if hybrid_values.dtype.kind in {"U", "S"}:
            aligned = np.array_equal(
                hybrid_values.astype(str), path_values.astype(str)
            ) and np.array_equal(hybrid_values.astype(str), rr_values.astype(str))
        else:
            aligned = np.array_equal(hybrid_values, path_values) and np.array_equal(
                hybrid_values, rr_values
            )
        if not aligned:
            raise ValueError(f"hybrid row field {name!r} differs from a component")

    rr_names = tuple(
        map(str, np.asarray(rr_artifact["score_names"], dtype=str).tolist())
    )
    rr_name = scalar_text(artifact, "rr_score_name")
    if rr_name not in rr_names:
        raise ValueError("hybrid RR score name disappeared from its component")
    if not np.array_equal(
        np.asarray(artifact["path_only"], dtype=np.float32),
        np.asarray(path_artifact["score"], dtype=np.float32),
    ) or not np.array_equal(
        np.asarray(artifact["rr_only"], dtype=np.float32),
        np.asarray(rr_artifact["scores"], dtype=np.float32)[
            reorder, rr_names.index(rr_name)
        ],
    ):
        raise ValueError("hybrid component score columns differ from their sources")

    for name in _PATH_GROUP_FIELDS:
        if not np.array_equal(
            np.asarray(artifact[name], dtype=str),
            np.asarray(path_artifact[name], dtype=str),
        ):
            raise ValueError(f"hybrid path source field {name!r} changed")
    rr_group_pairs = (
        ("rr_fit_group_id", "fit_group_id"),
        ("rr_calibration_group_id", "calibration_group_id"),
        ("rr_test_group_id", "test_group_id"),
        ("rr_test_sample_id", "test_sample_id"),
    )
    for hybrid_name, rr_name_field in rr_group_pairs:
        if not np.array_equal(
            np.asarray(artifact[hybrid_name], dtype=str),
            np.asarray(rr_artifact[rr_name_field], dtype=str),
        ):
            raise ValueError(f"hybrid RR source field {hybrid_name!r} changed")
    checks = (
        (
            "path_reference_path",
            str(Path(scalar_text(path_artifact, "reference_path")).resolve()),
        ),
        (
            "rr_reference_path",
            str(Path(scalar_text(rr_artifact, "reference_path")).resolve()),
        ),
        ("path_reference_sha256", sha256_text(path_artifact, "reference_sha256")),
        ("rr_reference_sha256", sha256_text(rr_artifact, "reference_sha256")),
        (
            "path_train_dataset_manifest_sha256",
            sha256_text(path_reference, "train_dataset_manifest_sha256"),
        ),
        (
            "rr_train_dataset_manifest_sha256",
            sha256_text(rr_reference, "train_dataset_manifest_sha256"),
        ),
    )
    for name, expected in checks:
        if scalar_text(artifact, name) != expected:
            raise ValueError(f"hybrid provenance field {name!r} changed")
    return path_reference, rr_reference


def build_rr_hybrid(
    path_score_path,
    rr_score_path,
    output_path,
    *,
    rr_score_name: str = ALLOWED_RR_SCORE,
) -> dict[str, object]:
    """Create one frozen path/RR hybrid without opening evaluation labels."""

    # Freeze both identities before any semantic validation or parsing.
    path_score_file = FrozenFile.capture(path_score_path)
    rr_score_file = FrozenFile.capture(rr_score_path)
    if rr_score_name != ALLOWED_RR_SCORE:
        raise ValueError(
            f"only the online-causal RR score {ALLOWED_RR_SCORE!r} is allowed"
        )
    path_artifact = load_path_score_artifact(path_score_file.path)
    path_reference = verify_path_score_provenance(path_artifact)
    rr_artifact = load_rr_score_artifact(rr_score_file.path)
    rr_reference = _verify_rr_reference_frozen(rr_artifact)

    path_manifest = sha256_text(path_artifact, "dataset_manifest_sha256")
    rr_manifest = sha256_text(rr_artifact, "dataset_manifest_sha256")
    if path_manifest != rr_manifest:
        raise ValueError("path and RR scores belong to different dataset manifests")

    names = tuple(map(str, np.asarray(rr_artifact["score_names"], dtype=str).tolist()))
    if rr_score_name not in names:
        raise ValueError("RR score artifact does not contain the required causal field")
    rr_column = names.index(rr_score_name)
    rr_keys = _key_rows(rr_artifact)
    path_keys = _key_rows(path_artifact)
    if len(set(rr_keys)) != len(rr_keys) or len(set(path_keys)) != len(path_keys):
        raise ValueError("a component score artifact contains duplicate token rows")
    if set(path_keys) != set(rr_keys):
        raise ValueError("path and RR artifacts do not cover identical token rows")
    rr_index = {key: index for index, key in enumerate(rr_keys)}
    reorder = np.asarray([rr_index[key] for key in path_keys], dtype=np.int64)

    for field in (
        "source_id",
        "response_length",
        "task_type",
        "data_source",
        "generator_model",
    ):
        left = np.asarray(path_artifact[field])
        right = np.asarray(rr_artifact[field])[reorder]
        equal = (
            np.array_equal(left.astype(str), right.astype(str))
            if left.dtype.kind in {"U", "S"}
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"path and RR artifacts disagree on {field}")

    path_scope = scalar_text(path_artifact, "audit_scope")
    rr_scope = scalar_text(rr_artifact, "audit_scope")
    if (
        path_scope != rr_scope
        or not _same_identifier_set(
            path_artifact["test_group_id"], rr_artifact["test_group_id"]
        )
        or not _same_identifier_set(
            path_artifact["test_sample_id"], rr_artifact["test_sample_id"]
        )
    ):
        raise ValueError("path and RR artifacts have different held-out scopes")

    path_score = np.asarray(path_artifact["score"], dtype=np.float64)
    rr_score = np.asarray(rr_artifact["scores"], dtype=np.float64)[reorder, rr_column]
    if not np.isfinite(path_score).all() or not np.isfinite(rr_score).all():
        raise ValueError("hybrid inputs must be finite")
    statistic, probability = _cauchy_two(np.exp(-path_score), np.exp(-rr_score))
    hybrid_score = -np.log(
        np.maximum(probability, np.finfo(np.float32).tiny)
    ).astype(np.float32)

    payload = {
        "schema": np.asarray(HYBRID_SCHEMA),
        "labels_included": np.asarray(False, dtype=np.bool_),
        "primary_detector": np.asarray("score"),
        "online_causal_score": np.asarray(True, dtype=np.bool_),
        "sample_id": np.asarray(path_artifact["sample_id"]),
        "source_id": np.asarray(path_artifact["source_id"]),
        "token_index": np.asarray(path_artifact["token_index"]),
        "response_length": np.asarray(path_artifact["response_length"]),
        "task_type": np.asarray(path_artifact["task_type"]),
        "data_source": np.asarray(path_artifact["data_source"]),
        "generator_model": np.asarray(path_artifact["generator_model"]),
        "score": hybrid_score,
        "path_only": path_score.astype(np.float32),
        "rr_only": rr_score.astype(np.float32),
        "fusion_stat": statistic,
        "global_p_value": probability,
        "rr_score_name": np.asarray(rr_score_name),
        "dataset_manifest_sha256": np.asarray(path_manifest),
        "audit_scope": np.asarray(path_scope),
        "test_group_id": np.asarray(path_artifact["test_group_id"], dtype=str),
        "test_sample_id": np.asarray(path_artifact["test_sample_id"], dtype=str),
        **{
            name: np.asarray(path_artifact[name], dtype=str)
            for name in _PATH_GROUP_FIELDS
        },
        "rr_fit_group_id": np.asarray(rr_artifact["fit_group_id"], dtype=str),
        "rr_calibration_group_id": np.asarray(
            rr_artifact["calibration_group_id"], dtype=str
        ),
        "rr_test_group_id": np.asarray(rr_artifact["test_group_id"], dtype=str),
        "rr_test_sample_id": np.asarray(rr_artifact["test_sample_id"], dtype=str),
        "rr_audit_scope": np.asarray(rr_scope),
        "path_score_path": np.asarray(str(path_score_file.path)),
        "path_score_sha256": np.asarray(path_score_file.sha256),
        "rr_score_path": np.asarray(str(rr_score_file.path)),
        "rr_score_sha256": np.asarray(rr_score_file.sha256),
        "path_reference_path": np.asarray(
            str(Path(scalar_text(path_artifact, "reference_path")).resolve())
        ),
        "path_reference_sha256": np.asarray(
            sha256_text(path_artifact, "reference_sha256")
        ),
        "rr_reference_path": np.asarray(
            str(Path(scalar_text(rr_artifact, "reference_path")).resolve())
        ),
        "rr_reference_sha256": np.asarray(
            sha256_text(rr_artifact, "reference_sha256")
        ),
        "path_train_dataset_manifest_sha256": np.asarray(
            sha256_text(path_reference, "train_dataset_manifest_sha256")
        ),
        "rr_train_dataset_manifest_sha256": np.asarray(
            sha256_text(rr_reference, "train_dataset_manifest_sha256")
        ),
    }

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_artifact = load_rr_hybrid(temporary)
        verify_rr_hybrid_provenance(temporary_artifact)
        path_score_file.verify(path_score_file.path)
        rr_score_file.verify(rr_score_file.path)
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        final_artifact = load_rr_hybrid(destination)
        verify_rr_hybrid_provenance(final_artifact)
        path_score_file.verify(path_score_file.path)
        rr_score_file.verify(rr_score_file.path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "hybrid_scores": str(destination),
        "hybrid_scores_sha256": FrozenFile.capture(destination).sha256,
        "dataset_manifest_sha256": path_manifest,
        "tokens": int(len(hybrid_score)),
        "rr_score_name": rr_score_name,
        "labels_read": False,
    }


__all__ = [
    "ALLOWED_RR_SCORE",
    "HYBRID_SCHEMA",
    "build_rr_hybrid",
    "load_rr_hybrid",
    "verify_rr_hybrid_provenance",
]
