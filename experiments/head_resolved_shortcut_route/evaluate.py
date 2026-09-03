"""Freeze the three shortcut-route axes before opening RAGTruth labels.

Evaluation is the only phase that requests labels from the dataset interface.
Every artifact, coordinate, identity, score direction, and validity mask is
checked and written to ``frozen_axes.npz`` before the dataset is reopened with
embedded labels enabled.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .data import TASK_TYPES, canonical_task_type
from .route_artifact import SCHEMA as ARTIFACT_SCHEMA
from .route_artifact import RouteArtifact, load_route_artifact, validate_artifact
from .route_shortcut import SUPPORT, VETO

SCHEMA = "shortcut-route-collection-v1"
VERSION = 1
REPORT_SCHEMA = "shortcut-route-evaluation-v1"
POSITION_BIN = 16
SURPRISAL_BIN_WIDTH = 0.5
MIN_BOOTSTRAP_SUCCESS_RATE = 0.9

SCORE_ORDER = (
    "carrier_drift_support",
    "negative_prompt_source_dispersion_support",
    "response_born_takeover_support",
)
VETO_ORDER = (
    "carrier_drift_veto",
    "prompt_source_dispersion_veto",
    "response_born_takeover_veto",
)
CONTROL_ORDER = (
    "absolute_response_position",
    "relative_response_position",
    "response_length",
    "observer_target_surprisal",
)
SCORE_DEFINITIONS = {
    "carrier_drift_support": {
        "artifact_value": "carrier_drift[:, support]",
        "direction": "higher_is_more_shortcut_like",
    },
    "negative_prompt_source_dispersion_support": {
        "artifact_value": "-prompt_source_dispersion[:, support]",
        "direction": "higher_score_means_lower_entropy_and_more_shortcut_like",
    },
    "response_born_takeover_support": {
        "artifact_value": "response_born_takeover[:, support]",
        "direction": "higher_is_more_shortcut_like",
    },
}
VETO_DEFINITIONS = {
    "carrier_drift_veto": "carrier_drift[:, veto]",
    "prompt_source_dispersion_veto": "prompt_source_dispersion[:, veto]",
    "response_born_takeover_veto": "response_born_takeover[:, veto]",
}
CONTROL_DEFINITIONS = {
    "absolute_response_position": "zero-based response token index; higher is later",
    "relative_response_position": "response midpoint fraction; higher is later",
    "response_length": "tokenized response length; higher is longer",
    "observer_target_surprisal": (
        "negative target log-probability under the frozen observer; not generator "
        "confidence when observer and generator differ"
    ),
}


def _array(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _json_identity(value: Any) -> str:
    """Give nested manifest identities a stable, directly comparable form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _load_manifest(
    root: Path, task_type: str, *, allow_partial: bool = False
) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "version",
        "artifact_schema",
        "dataset_identity",
        "source_identity",
        "observer_identity",
        "model_dtype",
        "top_k",
        "cover_mass",
        "task_types",
        "index",
        "samples",
        "complete",
        "labels_used",
        "dataset_candidates",
    }
    if not required.issubset(manifest):
        missing = ", ".join(sorted(required - set(manifest)))
        raise ValueError(f"collection manifest is missing: {missing}")
    if (
        manifest["schema"] != SCHEMA
        or manifest["version"] != VERSION
        or manifest["artifact_schema"] != ARTIFACT_SCHEMA
    ):
        raise ValueError("collection manifest schema does not match shortcut-route v1")
    if manifest["labels_used"] is not False:
        raise ValueError("evaluation requires a label-free collection")
    if not isinstance(manifest["complete"], bool):
        raise TypeError("collection manifest complete must be boolean")
    if manifest["complete"] is not True and not allow_partial:
        raise ValueError(
            "partial collection requires allow_partial=True and is not a formal run"
        )
    tasks = tuple(canonical_task_type(value) for value in manifest["task_types"])
    if task_type not in tasks:
        raise ValueError(f"collection has no declared {task_type} task")
    if (
        not isinstance(manifest["top_k"], int)
        or isinstance(manifest["top_k"], bool)
        or int(manifest["top_k"]) < 0
    ):
        raise ValueError("manifest top_k must be a nonnegative integer")
    cover_mass = float(manifest["cover_mass"])
    if not math.isfinite(cover_mass) or not 0 < cover_mass <= 1:
        raise ValueError("manifest cover_mass must be in (0, 1]")
    if any(
        not isinstance(manifest[name], int) or isinstance(manifest[name], bool)
        for name in ("samples", "dataset_candidates")
    ):
        raise ValueError("manifest sample counts must be integers")
    samples = int(manifest["samples"])
    candidates = int(manifest["dataset_candidates"])
    if samples < 0 or candidates < samples:
        raise ValueError("manifest sample counts are inconsistent")
    if manifest["complete"] and samples != candidates:
        raise ValueError("complete manifest does not cover every dataset candidate")
    return manifest


def _load_index(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    index_path = Path(str(manifest["index"]))
    if index_path.is_absolute() or ".." in index_path.parts:
        raise ValueError("manifest index must stay inside the collection")
    rows: list[dict[str, Any]] = []
    with (root / index_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != int(manifest["samples"]):
        raise ValueError("manifest and index sample counts disagree")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError("collection index sample IDs must be nonempty and unique")
    return rows


def _artifact_path(root: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must stay inside the collection")
    if relative.parts[:1] == ("samples",):
        return root / relative
    return root / "samples" / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_records(
    inputs: Iterable[tuple[str | Path, str | Path]],
    task_type: str,
    *,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only label-free manifests and indexes, preserving journal order."""

    task_type = canonical_task_type(task_type)
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    shared_identity: tuple[str, ...] | None = None
    seen_samples: set[str] = set()
    for shard, (collection_value, split_value) in enumerate(inputs):
        root = Path(collection_value)
        manifest = _load_manifest(root, task_type, allow_partial=allow_partial)
        dataset_path = manifest["dataset_identity"].get("path")
        if (
            dataset_path is None
            or Path(dataset_path).resolve() != Path(split_value).resolve()
        ):
            raise ValueError("dataset path does not match the collection identity")
        manifest_file = manifest["dataset_identity"].get("manifest_file", {})
        if "sha256" in manifest_file:
            current_manifest = Path(split_value) / "manifest.json"
            if (
                not current_manifest.is_file()
                or current_manifest.stat().st_size != int(manifest_file["size"])
                or _sha256(current_manifest) != str(manifest_file["sha256"])
            ):
                raise ValueError("dataset manifest changed after route collection")
        identity = (
            str(manifest["artifact_schema"]),
            _json_identity(manifest["source_identity"]),
            _json_identity(manifest["observer_identity"]),
            str(manifest["model_dtype"]),
            str(int(manifest["top_k"])),
            repr(float(manifest["cover_mass"])),
        )
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError("shortcut-route shards have different scientific identity")
        manifests.append(manifest)
        for row in _load_index(root, manifest):
            required = {
                "sample_id",
                "source_id",
                "task_type",
                "generator_model",
                "path",
                "bytes",
                "sha256",
                "events",
                "response_start",
            }
            if not required.issubset(row):
                missing = ", ".join(sorted(required - set(row)))
                raise ValueError(f"collection index row is missing: {missing}")
            row_task = canonical_task_type(row["task_type"])
            if row_task != task_type:
                continue
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise ValueError(f"sample appears in more than one shard: {sample_id}")
            seen_samples.add(sample_id)
            path = _artifact_path(root, row["path"])
            if not path.is_file():
                raise ValueError(f"indexed artifact is missing: {sample_id}")
            if isinstance(row["bytes"], bool) or path.stat().st_size != int(
                row["bytes"]
            ):
                raise ValueError(f"indexed artifact size changed: {sample_id}")
            if _sha256(path) != str(row["sha256"]):
                raise ValueError(f"indexed artifact digest changed: {sample_id}")
            if isinstance(row["events"], bool) or int(row["events"]) <= 0:
                raise ValueError("index events must be a positive integer")
            if (
                isinstance(row["response_start"], bool)
                or int(row["response_start"]) <= 0
            ):
                raise ValueError("index response_start must be a positive integer")
            records.append(
                {
                    **row,
                    "sample_id": sample_id,
                    "source_id": str(row["source_id"]),
                    "task_type": row_task,
                    "path": path,
                    "split_root": Path(split_value),
                    "physical_shard": shard,
                    "manifest_top_k": int(manifest["top_k"]),
                    "manifest_cover_mass": float(manifest["cover_mass"]),
                }
            )
    if not records:
        raise ValueError(f"no {task_type} samples in the supplied collections")
    return records, manifests


def _validate_record_artifact(
    record: Mapping[str, Any], artifact: RouteArtifact
) -> None:
    """Bind a fully validated route artifact to its label-free journal row."""

    validate_artifact(artifact)
    event_count = len(artifact.events.query_position)
    if event_count != int(record["events"]):
        raise ValueError("artifact event count does not match its index row")
    if artifact.response_start != int(record["response_start"]):
        raise ValueError("artifact response_start does not match its index row")
    if artifact.top_k != int(record["manifest_top_k"]):
        raise ValueError("artifact top_k does not match its manifest")
    if not math.isclose(
        artifact.cover_mass,
        float(record["manifest_cover_mass"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("artifact cover_mass does not match its manifest")
    query = _array(artifact.events.query_position, np.int64)
    prediction = _array(artifact.events.prediction_position, np.int64)
    target = _array(artifact.events.target_token_id, np.int64)
    expected_prediction = np.arange(
        artifact.response_start,
        artifact.response_start + event_count,
        dtype=np.int64,
    )
    if not np.array_equal(prediction, expected_prediction):
        raise ValueError("artifact prediction positions are not the complete response")
    if not np.array_equal(query, prediction - 1):
        raise ValueError("artifact violates the frozen q -> q + 1 alignment")
    source_token_id = _array(artifact.source_token_id, np.int64)
    if len(source_token_id) + 1 != artifact.response_start + event_count:
        raise ValueError("artifact source sequence and event count do not align")
    if event_count > 1 and not np.array_equal(
        target[:-1], source_token_id[artifact.response_start :]
    ):
        raise ValueError("artifact targets do not match its teacher-forced sequence")


def _axis_arrays(artifact: RouteArtifact) -> dict[str, np.ndarray]:
    axes = artifact.axes
    values = {
        "carrier_drift_support": _array(axes.carrier_drift[:, SUPPORT], np.float32),
        "negative_prompt_source_dispersion_support": -_array(
            axes.prompt_source_dispersion[:, SUPPORT], np.float32
        ),
        "response_born_takeover_support": _array(
            axes.response_born_takeover[:, SUPPORT], np.float32
        ),
        "carrier_drift_veto": _array(axes.carrier_drift[:, VETO], np.float32),
        "prompt_source_dispersion_veto": _array(
            axes.prompt_source_dispersion[:, VETO], np.float32
        ),
        "response_born_takeover_veto": _array(
            axes.response_born_takeover[:, VETO], np.float32
        ),
    }
    masks = {
        "carrier_drift_support__valid": _array(
            axes.carrier_drift_defined[:, SUPPORT], bool
        ),
        "negative_prompt_source_dispersion_support__valid": _array(
            axes.prompt_source_dispersion_defined[:, SUPPORT], bool
        ),
        "response_born_takeover_support__valid": _array(
            axes.response_born_takeover_defined[:, SUPPORT], bool
        ),
        "carrier_drift_veto__valid": _array(axes.carrier_drift_defined[:, VETO], bool),
        "prompt_source_dispersion_veto__valid": _array(
            axes.prompt_source_dispersion_defined[:, VETO], bool
        ),
        "response_born_takeover_veto__valid": _array(
            axes.response_born_takeover_defined[:, VETO], bool
        ),
    }
    count = len(artifact.events.query_position)
    for name, value in values.items():
        if value.shape != (count,):
            raise ValueError(f"axis summary is not event-aligned: {name}")
        valid = masks[f"{name}__valid"]
        if valid.shape != (count,):
            raise ValueError(f"axis validity is not event-aligned: {name}")
        if not np.isfinite(value[valid]).all():
            raise ValueError(f"defined axis values must be finite: {name}")
    return {**values, **masks}


def _control_arrays(artifact: RouteArtifact) -> dict[str, np.ndarray]:
    """Freeze independent non-route controls on the same event axis."""

    count = len(artifact.events.query_position)
    target_logprob = _array(artifact.readout.target_logprob, np.float32)
    if target_logprob.shape != (count,) or not np.isfinite(target_logprob).all():
        raise ValueError(
            "observer target log-probability must be finite and event-aligned"
        )
    response_index = np.arange(count, dtype=np.float32)
    return {
        "target_logprob": target_logprob,
        "observer_target_surprisal": -target_logprob,
        "absolute_response_position": response_index,
        "relative_response_position": (response_index + 0.5) / count,
    }


def freeze_axes(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Validate every artifact and build the immutable, label-free token table."""

    event_fields: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "generator_model": [],
        "physical_shard": [],
        "record_index": [],
        "response_index": [],
        "response_length": [],
        "response_start": [],
        "query_position": [],
        "prediction_position": [],
        "target_token_id": [],
        "target_logprob": [],
        "observer_target_surprisal": [],
        "absolute_response_position": [],
        "relative_response_position": [],
        **{name: [] for name in SCORE_ORDER},
        **{f"{name}__valid": [] for name in SCORE_ORDER},
        **{name: [] for name in VETO_ORDER},
        **{f"{name}__valid": [] for name in VETO_ORDER},
    }
    record_fields: dict[str, list[Any]] = {
        "record_sample_id": [],
        "record_source_id": [],
        "record_task_type": [],
        "record_generator_model": [],
        "record_physical_shard": [],
        "record_response_start": [],
        "record_event_count": [],
    }
    source_ptr = [0]
    source_tokens: list[np.ndarray] = []
    for record_index, record in enumerate(records):
        artifact = load_route_artifact(record["path"])
        _validate_record_artifact(record, artifact)
        axes = _axis_arrays(artifact)
        controls = _control_arrays(artifact)
        count = len(artifact.events.query_position)
        generator = (
            "" if record["generator_model"] is None else str(record["generator_model"])
        )
        event_fields["sample_id"].append(np.repeat(str(record["sample_id"]), count))
        event_fields["source_id"].append(np.repeat(str(record["source_id"]), count))
        event_fields["task_type"].append(np.repeat(str(record["task_type"]), count))
        event_fields["generator_model"].append(np.repeat(generator, count))
        event_fields["physical_shard"].append(
            np.full(count, int(record["physical_shard"]), dtype=np.int32)
        )
        event_fields["record_index"].append(
            np.full(count, record_index, dtype=np.int32)
        )
        event_fields["response_index"].append(np.arange(count, dtype=np.int32))
        event_fields["response_length"].append(np.full(count, count, dtype=np.int32))
        event_fields["response_start"].append(
            np.full(count, artifact.response_start, dtype=np.int32)
        )
        event_fields["query_position"].append(
            _array(artifact.events.query_position, np.int64)
        )
        event_fields["prediction_position"].append(
            _array(artifact.events.prediction_position, np.int64)
        )
        event_fields["target_token_id"].append(
            _array(artifact.events.target_token_id, np.int64)
        )
        for name, value in axes.items():
            event_fields[name].append(value)
        for name, value in controls.items():
            event_fields[name].append(value)

        record_fields["record_sample_id"].append(str(record["sample_id"]))
        record_fields["record_source_id"].append(str(record["source_id"]))
        record_fields["record_task_type"].append(str(record["task_type"]))
        record_fields["record_generator_model"].append(generator)
        record_fields["record_physical_shard"].append(int(record["physical_shard"]))
        record_fields["record_response_start"].append(artifact.response_start)
        record_fields["record_event_count"].append(count)
        current_source = _array(artifact.source_token_id, np.int64)
        source_tokens.append(current_source)
        source_ptr.append(source_ptr[-1] + len(current_source))

    frozen = {name: np.concatenate(values) for name, values in event_fields.items()}
    frozen.update(
        {
            "record_sample_id": np.asarray(record_fields["record_sample_id"]),
            "record_source_id": np.asarray(record_fields["record_source_id"]),
            "record_task_type": np.asarray(record_fields["record_task_type"]),
            "record_generator_model": np.asarray(
                record_fields["record_generator_model"]
            ),
            "record_physical_shard": np.asarray(
                record_fields["record_physical_shard"], dtype=np.int32
            ),
            "record_response_start": np.asarray(
                record_fields["record_response_start"], dtype=np.int32
            ),
            "record_event_count": np.asarray(
                record_fields["record_event_count"], dtype=np.int32
            ),
            "canonical_source_ptr": np.asarray(source_ptr, dtype=np.int64),
            "canonical_source_token_id": np.concatenate(source_tokens).astype(
                np.int64, copy=False
            ),
        }
    )
    return frozen


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_frozen(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        if "label" in stored:
            raise ValueError("frozen axes must not contain labels")
        return {name: np.array(stored[name], copy=True) for name in stored.files}


def _record_rows(frozen: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(frozen["record_index"]) == index)


def _validate_sample_identity(
    record: Mapping[str, Any],
    frozen: Mapping[str, np.ndarray],
    record_index: int,
    sample: Any,
) -> tuple[Any, np.ndarray]:
    """Rebind the frozen event identity to the exact RAGTruth token sequence."""

    actual_sample_id = getattr(sample, "sample_id", record["sample_id"])
    if str(actual_sample_id) != str(record["sample_id"]):
        raise ValueError(
            "sample_id changed between frozen axes and the canonical cache"
        )
    if str(sample.source_id) != str(frozen["record_source_id"][record_index]):
        raise ValueError("source_id changed between frozen axes and labels")
    if canonical_task_type(sample.task_type) != str(
        frozen["record_task_type"][record_index]
    ):
        raise ValueError("task_type changed between frozen axes and labels")
    expected_generator = str(frozen["record_generator_model"][record_index])
    raw_generator = getattr(sample, "generator_model", None)
    actual_generator = "" if raw_generator is None else str(raw_generator)
    if actual_generator != expected_generator:
        raise ValueError("generator_model changed between frozen axes and labels")

    attention = sample.attention()
    token_ids = _array(attention.token_ids, np.int64)
    response_start = int(frozen["record_response_start"][record_index])
    if int(attention.response_idx) != response_start:
        raise ValueError("response_start changed between frozen axes and labels")
    start = int(frozen["canonical_source_ptr"][record_index])
    stop = int(frozen["canonical_source_ptr"][record_index + 1])
    source_token_id = np.asarray(
        frozen["canonical_source_token_id"][start:stop], dtype=np.int64
    )
    if token_ids.shape != (len(source_token_id) + 1,) or not np.array_equal(
        token_ids[:-1], source_token_id
    ):
        raise ValueError("token sequence changed between frozen axes and labels")

    rows = _record_rows(frozen, record_index)
    prediction = np.asarray(frozen["prediction_position"][rows], dtype=np.int64)
    query = np.asarray(frozen["query_position"][rows], dtype=np.int64)
    target = np.asarray(frozen["target_token_id"][rows], dtype=np.int64)
    if not np.array_equal(query + 1, prediction):
        raise ValueError("frozen event identity violates q -> q + 1")
    if not np.array_equal(token_ids[prediction], target):
        raise ValueError("target token IDs changed between frozen axes and labels")
    response_index = prediction - response_start
    if not np.array_equal(
        response_index, np.asarray(frozen["response_index"][rows], dtype=np.int64)
    ):
        raise ValueError(
            "frozen response_index is not prediction_position - response_start"
        )
    if str(record["sample_id"]) != str(frozen["record_sample_id"][record_index]):
        raise ValueError("record order changed after axes were frozen")
    return attention, response_index


def validate_canonical_inputs(
    records: Sequence[Mapping[str, Any]], frozen: Mapping[str, np.ndarray]
) -> None:
    """Bind every frozen event to a label-free RAGTruth cache sample."""

    shards = sorted({int(record["physical_shard"]) for record in records})
    for shard in shards:
        selected = [
            (index, record)
            for index, record in enumerate(records)
            if int(record["physical_shard"]) == shard
        ]
        dataset = open_research_dataset(
            selected[0][1]["split_root"],
            device="cpu",
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        for record_index, record in selected:
            sample = dataset[str(record["sample_id"])]
            try:
                _validate_sample_identity(record, frozen, record_index, sample)
            finally:
                sample.release_attention()


def _load_labels(
    records: Sequence[Mapping[str, Any]], frozen: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Open embedded labels after freeze and align them by ``p - P``."""

    labels = np.empty(len(frozen["sample_id"]), dtype=bool)
    shards = sorted({int(record["physical_shard"]) for record in records})
    for shard in shards:
        selected = [
            (index, record)
            for index, record in enumerate(records)
            if int(record["physical_shard"]) == shard
        ]
        dataset = open_research_dataset(
            selected[0][1]["split_root"],
            device="cpu",
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        prepared = dataset.prepare_evaluation_labels(
            [str(record["sample_id"]) for _index, record in selected]
        )
        for record_index, record in selected:
            sample = dataset[str(record["sample_id"])]
            try:
                rows = _record_rows(frozen, record_index)
                _attention, response_index = _validate_sample_identity(
                    record, frozen, record_index, sample
                )
                response_label = _array(prepared.response_labels(sample), bool)
                if response_label.ndim != 1 or (
                    len(response_index) and response_index.max() >= len(response_label)
                ):
                    raise ValueError(
                        "RAGTruth labels do not cover frozen response events"
                    )
                labels[rows] = response_label[response_index]
            finally:
                sample.release_attention()
    return labels


def _detection_context(
    label: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    valid: np.ndarray,
    eligible: np.ndarray,
    *,
    bootstrap: int,
) -> dict[str, Any]:
    source = np.asarray(arrays["source_id"])[valid]
    sample = np.asarray(arrays.get("sample_id", arrays["source_id"]))[valid]
    current_label = np.asarray(label[valid], dtype=bool)
    position_coverage: list[dict[str, Any]] = []
    if "relative_response_position" in arrays:
        relative = np.asarray(arrays["relative_response_position"], dtype=np.float64)
        decile = np.minimum(np.floor(relative * 10).astype(np.int16), 9)
        for index in range(10):
            current = eligible & (decile == index)
            current_valid = valid & current
            if current.any():
                position_coverage.append(
                    {
                        "relative_decile": index,
                        "eligible_tokens": int(current.sum()),
                        "valid_tokens": int(current_valid.sum()),
                        "eligible_positive_tokens": int((current & label).sum()),
                        "valid_positive_tokens": int((current_valid & label).sum()),
                        "eligible_negative_tokens": int((current & ~label).sum()),
                        "valid_negative_tokens": int((current_valid & ~label).sum()),
                    }
                )
    return {
        "eligible_tokens": int(eligible.sum()),
        "valid_tokens": int(valid.sum()),
        "invalid_tokens": int(eligible.sum() - valid.sum()),
        "valid_fraction": float(valid.sum() / max(eligible.sum(), 1)),
        "positive_tokens": int(current_label.sum()),
        "negative_tokens": int((~current_label).sum()),
        "prevalence": (float(current_label.mean()) if len(current_label) else None),
        "valid_samples": int(np.unique(sample).size),
        "valid_sources": int(np.unique(source).size),
        "positive_sources": int(np.unique(source[current_label]).size),
        "negative_sources": int(np.unique(source[~current_label]).size),
        "validity_by_response_decile": position_coverage,
        "auroc": None,
        "average_precision": None,
        "auroc_ci95": [None, None],
        "average_precision_ci95": [None, None],
        "bootstrap_requested": int(bootstrap),
        "bootstrap_successful": 0,
        "bootstrap_ci_reliable": None if not bootstrap else False,
    }


def _source_metric_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[list[float | None], list[float | None], int]:
    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values: list[tuple[float, float]] = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                (
                    float(roc_auc_score(label[index], score[index])),
                    float(average_precision_score(label[index], score[index])),
                )
            )
    if not values:
        return [None, None], [None, None], 0
    array = np.asarray(values, dtype=np.float64)
    return (
        [float(value) for value in np.quantile(array[:, 0], (0.025, 0.975))],
        [float(value) for value in np.quantile(array[:, 1], (0.025, 0.975))],
        len(values),
    )


def _detection_for_names(
    label: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    bootstrap: int,
    seed: int,
    eligible: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute fixed-direction token-micro metrics for named frozen values."""

    label = np.asarray(label, dtype=bool)
    base = (
        np.ones(len(label), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool).copy()
    )
    if base.shape != label.shape:
        raise ValueError("detection eligibility must share the event axis")
    result: dict[str, dict[str, Any]] = {}
    for offset, name in enumerate(names):
        score = np.asarray(arrays[name], dtype=np.float64)
        if score.shape != label.shape:
            raise ValueError(f"score does not share the event axis: {name}")
        valid = base.copy()
        validity_name = f"{name}__valid"
        if validity_name in arrays:
            current_valid = np.asarray(arrays[validity_name], dtype=bool)
            if current_valid.shape != label.shape:
                raise ValueError(f"validity does not share the event axis: {name}")
            valid &= current_valid
        valid &= np.isfinite(score)
        current_label = np.asarray(label[valid], dtype=bool)
        current_score = score[valid]
        metric = _detection_context(label, arrays, valid, base, bootstrap=bootstrap)
        if np.unique(current_label).size != 2:
            result[name] = metric
            continue
        metric["auroc"] = float(roc_auc_score(current_label, current_score))
        metric["average_precision"] = float(
            average_precision_score(current_label, current_score)
        )
        if bootstrap:
            auroc_ci, ap_ci, successful = _source_metric_bootstrap(
                current_label,
                current_score,
                np.asarray(arrays["source_id"])[valid],
                replicates=bootstrap,
                seed=seed + offset,
            )
            reliable = (
                successful / bootstrap >= MIN_BOOTSTRAP_SUCCESS_RATE
                and np.unique(np.asarray(arrays["source_id"])[valid]).size >= 2
            )
            metric.update(
                auroc_ci95=auroc_ci if reliable else [None, None],
                average_precision_ci95=ap_ci if reliable else [None, None],
            )
            metric.update(
                bootstrap_successful=successful,
                bootstrap_ci_reliable=reliable,
            )
        result[name] = metric
    return result


def detection_summary(
    label: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Compute each preregistered axis with its own frozen validity mask."""

    return _detection_for_names(
        label,
        arrays,
        SCORE_ORDER,
        bootstrap=bootstrap,
        seed=seed,
    )


def _source_difference_bootstrap(
    label: np.ndarray,
    value: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[list[float | None], int]:
    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                float(
                    value[index][label[index]].mean()
                    - value[index][~label[index]].mean()
                )
            )
    if not values:
        return [None, None], 0
    return [float(x) for x in np.quantile(values, (0.025, 0.975))], len(values)


def _raw_group_difference(
    label: np.ndarray,
    value: np.ndarray,
    source_id: np.ndarray,
    valid: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    current_label = np.asarray(label[valid], dtype=bool)
    current_value = np.asarray(value[valid], dtype=np.float64)
    current_source = np.asarray(source_id)[valid]
    correct = current_value[~current_label]
    hallucinated = current_value[current_label]
    difference = (
        float(hallucinated.mean() - correct.mean())
        if len(correct) and len(hallucinated)
        else None
    )
    interval: list[float | None] = [None, None]
    successful = 0
    reliable: bool | None = None if not bootstrap else False
    if bootstrap and difference is not None:
        interval, successful = _source_difference_bootstrap(
            current_label,
            current_value,
            current_source,
            replicates=bootstrap,
            seed=seed,
        )
        reliable = (
            successful / bootstrap >= MIN_BOOTSTRAP_SUCCESS_RATE
            and np.unique(current_source).size >= 2
        )
        if not reliable:
            interval = [None, None]
    return {
        "valid_tokens": int(valid.sum()),
        "positive_tokens": int(current_label.sum()),
        "negative_tokens": int((~current_label).sum()),
        "sources": int(np.unique(current_source).size),
        "positive_sources": int(np.unique(current_source[current_label]).size),
        "negative_sources": int(np.unique(current_source[~current_label]).size),
        "correct_mean": float(correct.mean()) if len(correct) else None,
        "hallucinated_mean": (
            float(hallucinated.mean()) if len(hallucinated) else None
        ),
        "hallucinated_minus_correct": difference,
        "difference_ci95": interval,
        "bootstrap_requested": int(bootstrap),
        "bootstrap_successful": successful,
        "bootstrap_ci_reliable": reliable,
    }


def _matched_group_difference(
    label: np.ndarray,
    value: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    valid: np.ndarray,
    *,
    match_surprisal: bool,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compare H-C within fixed, label-independent response strata."""

    sample_id = np.asarray(arrays["sample_id"])
    source_id = np.asarray(arrays["source_id"])
    response_index = np.asarray(arrays["response_index"], dtype=np.int64)
    relative = np.asarray(arrays["relative_response_position"], dtype=np.float64)
    surprisal = np.asarray(arrays["observer_target_surprisal"], dtype=np.float64)
    absolute_bin = response_index // POSITION_BIN
    relative_bin = np.minimum(np.floor(relative * 10).astype(np.int16), 9)
    surprisal_bin = np.floor(surprisal / SURPRISAL_BIN_WIDTH).astype(np.int64)

    cells: dict[tuple[Any, ...], list[int]] = {}
    for index in np.flatnonzero(valid):
        key: tuple[Any, ...] = (
            str(sample_id[index]),
            int(absolute_bin[index]),
            int(relative_bin[index]),
        )
        if match_surprisal:
            key += (int(surprisal_bin[index]),)
        cells.setdefault(key, []).append(int(index))

    by_sample: dict[str, list[tuple[float, float]]] = {}
    matched_positive: set[int] = set()
    matched_negative: set[int] = set()
    matched_cells = 0
    for key, members in cells.items():
        rows = np.asarray(members, dtype=np.int64)
        positive = rows[label[rows]]
        negative = rows[~label[rows]]
        if not len(positive) or not len(negative):
            continue
        weight = len(positive) * len(negative) / (len(positive) + len(negative))
        effect = float(value[positive].mean() - value[negative].mean())
        by_sample.setdefault(str(key[0]), []).append((effect, float(weight)))
        matched_positive.update(int(row) for row in positive)
        matched_negative.update(int(row) for row in negative)
        matched_cells += 1

    sample_source = {
        str(sample): str(source)
        for sample, source in zip(sample_id, source_id, strict=True)
    }
    by_source: dict[str, list[float]] = {}
    for sample, effects in by_sample.items():
        values = np.asarray([effect for effect, _weight in effects])
        weights = np.asarray([weight for _effect, weight in effects])
        by_source.setdefault(sample_source[sample], []).append(
            float(np.average(values, weights=weights))
        )
    source_effects = np.asarray(
        [np.mean(effects) for effects in by_source.values()], dtype=np.float64
    )
    difference = float(source_effects.mean()) if len(source_effects) else None
    interval: list[float | None] = [None, None]
    if bootstrap and len(source_effects) >= 2:
        random = np.random.default_rng(seed)
        draws = random.choice(
            source_effects,
            size=(bootstrap, len(source_effects)),
            replace=True,
        ).mean(axis=1)
        interval = [float(x) for x in np.quantile(draws, (0.025, 0.975))]
    valid_positive = int((valid & label).sum())
    return {
        "valid_tokens": int(valid.sum()),
        "hallucinated_minus_correct": difference,
        "difference_ci95": interval,
        "matched_cells": matched_cells,
        "matched_samples": len(by_sample),
        "matched_sources": len(by_source),
        "matched_positive_tokens": len(matched_positive),
        "matched_negative_tokens": len(matched_negative),
        "positive_token_coverage": len(matched_positive) / max(valid_positive, 1),
        "sources": len(by_source),
        "positive_sources": len(by_source),
        "negative_sources": len(by_source),
        "bootstrap_requested": int(bootstrap),
        "bootstrap_successful": int(bootstrap if len(source_effects) >= 2 else 0),
        "bootstrap_ci_reliable": (len(source_effects) >= 2 if bootstrap else None),
    }


def support_group_audit(
    label: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Report raw and fixed-stratum H-C differences for every support axis."""

    label = np.asarray(label, dtype=bool)
    raw: dict[str, Any] = {}
    position_matched: dict[str, Any] = {}
    position_surprisal_matched: dict[str, Any] = {}
    for offset, name in enumerate(SCORE_ORDER):
        value = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(arrays[f"{name}__valid"], dtype=bool).copy()
        valid &= np.isfinite(value)
        axis_seed = seed + 3 * offset
        raw[name] = _raw_group_difference(
            label,
            value,
            np.asarray(arrays["source_id"]),
            valid,
            bootstrap=bootstrap,
            seed=axis_seed,
        )
        position_matched[name] = _matched_group_difference(
            label,
            value,
            arrays,
            valid,
            match_surprisal=False,
            bootstrap=bootstrap,
            seed=axis_seed + 1,
        )
        position_surprisal_matched[name] = _matched_group_difference(
            label,
            value,
            arrays,
            valid,
            match_surprisal=True,
            bootstrap=bootstrap,
            seed=axis_seed + 2,
        )
    metrics = {
        name: {
            "raw": raw[name],
            "position_matched": position_matched[name],
            "position_surprisal_matched": position_surprisal_matched[name],
        }
        for name in SCORE_ORDER
    }
    return {
        "role": "posthoc_group_difference_not_score_selection",
        "aggregation": "matched_cell_then_response_then_equal_source",
        "position_matching": (
            f"same sample, absolute response bin width {POSITION_BIN}, "
            "relative response decile"
        ),
        "position_surprisal_matching": (
            f"position matching plus observer surprisal bin width "
            f"{SURPRISAL_BIN_WIDTH:g}"
        ),
        "length_and_generator_control": "exact within the same response sample",
        "metrics": metrics,
    }


def veto_audit(
    label: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Report raw veto group differences without promoting them to detectors."""

    output: dict[str, dict[str, Any]] = {}
    for offset, name in enumerate(VETO_ORDER):
        value = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(arrays[f"{name}__valid"], dtype=bool).copy()
        valid &= np.isfinite(value)
        output[name] = {
            "role": "raw_posthoc_audit_no_preregistered_hallucination_direction",
            **_raw_group_difference(
                np.asarray(label, dtype=bool),
                value,
                np.asarray(arrays["source_id"]),
                valid,
                bootstrap=bootstrap,
                seed=seed + offset,
            ),
        }
    return output


def build_report(
    *,
    task_type: str,
    arrays: Mapping[str, np.ndarray],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Build one task-specific report from already frozen axes and labels."""

    task_type = canonical_task_type(task_type)
    label = np.asarray(arrays["label"], dtype=bool)
    required_values = (*SCORE_ORDER, *VETO_ORDER, *CONTROL_ORDER)
    if any(len(np.asarray(arrays[name])) != len(label) for name in required_values):
        raise ValueError("scores and labels must share the frozen event axis")
    common_valid = np.ones(len(label), dtype=bool)
    control_by_axis: dict[str, dict[str, Any]] = {}
    for name in SCORE_ORDER:
        valid = np.asarray(arrays[f"{name}__valid"], dtype=bool).copy()
        valid &= np.isfinite(np.asarray(arrays[name], dtype=np.float64))
        common_valid &= valid
        control_by_axis[name] = _detection_for_names(
            label,
            arrays,
            CONTROL_ORDER,
            bootstrap=bootstrap,
            seed=seed + 20 + SCORE_ORDER.index(name) * len(CONTROL_ORDER),
            eligible=valid,
        )

    by_generator: dict[str, Any] = {}
    generator = np.asarray(arrays["generator_model"])
    for offset, model in enumerate(
        sorted(str(value) for value in np.unique(generator))
    ):
        eligible = generator == model
        by_generator[model or "<unknown>"] = {
            "samples": int(np.unique(np.asarray(arrays["sample_id"])[eligible]).size),
            "sources": int(np.unique(np.asarray(arrays["source_id"])[eligible]).size),
            "tokens": int(eligible.sum()),
            "hallucinated_tokens": int((label & eligible).sum()),
            "detection": _detection_for_names(
                label,
                arrays,
                SCORE_ORDER,
                bootstrap=bootstrap,
                seed=seed + 100 + offset * len(SCORE_ORDER),
                eligible=eligible,
            ),
            "control_detection": _detection_for_names(
                label,
                arrays,
                CONTROL_ORDER,
                bootstrap=bootstrap,
                seed=seed + 200 + offset * len(CONTROL_ORDER),
                eligible=eligible,
            ),
        }
    return {
        "schema": REPORT_SCHEMA,
        "task_type": task_type,
        "samples": int(np.unique(arrays["sample_id"]).size),
        "sources": int(np.unique(arrays["source_id"]).size),
        "tokens": len(label),
        "hallucinated_tokens": int(label.sum()),
        "prevalence": float(label.mean()) if len(label) else None,
        "main_axes": SCORE_DEFINITIONS,
        "detection_estimand": "token_micro",
        "detection_bootstrap_unit": "source_id_cluster",
        "bootstrap_policy": {
            "requested_replicates": int(bootstrap),
            "minimum_success_rate_for_ci": MIN_BOOTSTRAP_SUCCESS_RATE,
            "minimum_sources_for_ci": 2,
            "single_class_replicates": "excluded_and_counted_as_unsuccessful",
        },
        "detection": detection_summary(label, arrays, bootstrap=bootstrap, seed=seed),
        "control_definitions": CONTROL_DEFINITIONS,
        "control_detection": _detection_for_names(
            label,
            arrays,
            CONTROL_ORDER,
            bootstrap=bootstrap,
            seed=seed + len(SCORE_ORDER),
        ),
        "control_detection_by_axis_validity": control_by_axis,
        "support_group_audit": support_group_audit(
            label,
            arrays,
            bootstrap=bootstrap,
            seed=seed + 40,
        ),
        "common_validity_sensitivity": {
            "valid_tokens": int(common_valid.sum()),
            "role": "axis_comparison_sensitivity_not_main_estimand",
            "detection": _detection_for_names(
                label,
                arrays,
                SCORE_ORDER,
                bootstrap=bootstrap,
                seed=seed + 60,
                eligible=common_valid,
            ),
            "control_detection": _detection_for_names(
                label,
                arrays,
                CONTROL_ORDER,
                bootstrap=bootstrap,
                seed=seed + 80,
                eligible=common_valid,
            ),
        },
        "by_generator_model": by_generator,
        "veto_definitions": VETO_DEFINITIONS,
        "veto_audit": veto_audit(
            label, arrays, bootstrap=bootstrap, seed=seed + len(SCORE_ORDER)
        ),
        "labels_used_during": "posthoc_evaluation_only_after_frozen_axes",
    }


def evaluate_all(
    *,
    inputs: Iterable[tuple[str | Path, str | Path]],
    task_type: str,
    output: str | Path,
    bootstrap: int = 1000,
    seed: int = 20260828,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Freeze one task's route axes, then open and align RAGTruth labels."""

    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be boolean")
    task_type = canonical_task_type(task_type)
    records, manifests = _pool_records(inputs, task_type, allow_partial=allow_partial)
    frozen_path = Path(output).with_name("frozen_axes.npz")
    frozen_axes = freeze_axes(records)
    validate_canonical_inputs(records, frozen_axes)
    _write_npz(frozen_path, frozen_axes)

    # Reopen the immutable boundary: label-side code never receives artifacts.
    frozen = _load_frozen(frozen_path)
    label = _load_labels(records, frozen)
    merged = {**frozen, "label": label}
    report = build_report(
        task_type=task_type,
        arrays=merged,
        bootstrap=bootstrap,
        seed=seed,
    )
    report.update(
        frozen_axes=frozen_path.name,
        physical_shards=len(manifests),
        capture_complete=all(manifest["complete"] is True for manifest in manifests),
        collection_status=(
            "complete"
            if all(manifest["complete"] is True for manifest in manifests)
            else "partial"
        ),
        analysis_status=(
            "preregistered_association_evaluation"
            if all(manifest["complete"] is True for manifest in manifests)
            else "partial_smoke_not_formal"
        ),
        claims_boundary=(
            "Association report only; METHOD structural controls and independent "
            "replication are required before retaining the mechanism."
        ),
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_npz(output_path.with_name("token_results.npz"), merged)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


__all__ = [
    "ARTIFACT_SCHEMA",
    "CONTROL_DEFINITIONS",
    "CONTROL_ORDER",
    "REPORT_SCHEMA",
    "SCHEMA",
    "SCORE_DEFINITIONS",
    "SCORE_ORDER",
    "TASK_TYPES",
    "VERSION",
    "VETO_DEFINITIONS",
    "VETO_ORDER",
    "build_report",
    "detection_summary",
    "evaluate_all",
    "freeze_axes",
    "support_group_audit",
    "validate_canonical_inputs",
    "veto_audit",
]
