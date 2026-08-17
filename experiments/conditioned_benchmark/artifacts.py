"""Adapters from existing score artifacts to a common method registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import MethodScore, ScoreArtifact


SPECTRAL_FIELDS = {
    "primary": ("score", None),
    "residual_tail": ("score_rr_residual", None),
    "raw_residual_energy": ("rr_residual_energy", None),
    "in_subspace_tail": ("score_rr_latent", None),
    "ppca_tail": ("score_rr_ppca", None),
    "localized_channel_tail": ("score_rr_localized", None),
    "peak_channel": ("top_channel_score", 0),
}

TRAJECTORY_SIGNATURE = {
    "score_full",
    "score_mass_only",
    "score_dynamics_only",
    "score_route_embedding",
    "score_prompt_mass_low",
}

IDENTIFIER_FIELDS = {
    "schema",
    "sample_id",
    "sample_ids",
    "token_index",
    "position",
    "positions",
    "source_id",
    "source_ids",
    "task_type",
    "data_source",
    "generator_model",
    "labels",
    "reference_path",
    "reference_sha256",
    "topology_reference_path",
    "spectral_reference_path",
    "feature_names",
}


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: str
    adapter: str = "auto"
    protocol: str = "unknown"
    methods: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ArtifactSpec":
        if "name" not in value or "path" not in value:
            raise ValueError("each artifact requires name and path")
        return cls(
            name=str(value["name"]),
            path=str(value["path"]),
            adapter=str(value.get("adapter", "auto")),
            protocol=str(value.get("protocol", "unknown")),
            methods=tuple(value.get("methods", ())),
        )


def _scalar_text(arrays, name, default="") -> str:
    if name not in arrays:
        return default
    value = np.asarray(arrays[name])
    return str(value.item()) if value.ndim == 0 else default


def _first(arrays, names, *, required=True):
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    if required:
        raise ValueError(f"artifact misses every row field in {tuple(names)}")
    return None


def _column(arrays, field: str, column) -> np.ndarray:
    if field not in arrays:
        raise ValueError(f"configured score field {field!r} is missing")
    values = np.asarray(arrays[field])
    if column is None:
        if values.ndim != 1:
            raise ValueError(
                f"score field {field!r} needs a column, shape={values.shape}"
            )
        return values
    if values.ndim != 2:
        raise ValueError(f"column requested from non-matrix field {field!r}")
    if isinstance(column, str) and not column.lstrip("-").isdigit():
        if "feature_names" not in arrays:
            raise ValueError(f"named column {column!r} requires feature_names")
        names = np.asarray(arrays["feature_names"]).astype(str)
        matches = np.flatnonzero(names == column)
        if len(matches) != 1:
            raise ValueError(f"feature column {column!r} is missing or ambiguous")
        column = int(matches[0])
    return values[:, int(column)]


def _configured_methods(spec: ArtifactSpec, arrays) -> dict[str, MethodScore]:
    methods = {}
    for item in spec.methods:
        if "name" not in item or "field" not in item:
            raise ValueError("configured methods require name and field")
        short_name = str(item["name"])
        name = f"{spec.name}.{short_name}"
        methods[name] = MethodScore(
            name=name,
            values=_column(arrays, str(item["field"]), item.get("column")),
            direction=str(item.get("direction", "higher")),
            protocol=str(item.get("protocol", spec.protocol)),
            source_field=str(item["field"]),
            source_direction=str(item.get("direction", "higher")),
        )
    return methods


def _automatic_methods(spec: ArtifactSpec, arrays, schema: str, row_count: int):
    methods = {}
    if schema == "rr-spectral-score":
        protocol = "label_free_frozen_score"
        for short_name, (field, column) in SPECTRAL_FIELDS.items():
            if field not in arrays:
                continue
            name = f"{spec.name}.{short_name}"
            methods[name] = MethodScore(
                name=name,
                values=_column(arrays, field, column),
                protocol=protocol,
                source_field=field,
                source_direction="higher",
            )
        return methods

    if schema == "cmrp-score-v1":
        if "score" not in arrays:
            raise ValueError("CMRP score artifact misses its frozen primary score")
        name = f"{spec.name}.primary"
        return {
            name: MethodScore(
                name=name,
                values=_column(arrays, "score", None),
                protocol="label_free_frozen_score",
                source_field="score",
                source_direction="higher",
            )
        }

    score_fields = [
        name
        for name in arrays.files
        if (name == "score" or name.startswith("score_"))
        and np.asarray(arrays[name]).ndim == 1
        and len(arrays[name]) == row_count
    ]
    if not score_fields and {"labels", "sample_ids", "positions"}.issubset(
        arrays.files
    ):
        score_fields = [
            name
            for name in arrays.files
            if name not in IDENTIFIER_FIELDS
            and np.asarray(arrays[name]).ndim == 1
            and len(arrays[name]) == row_count
            and np.issubdtype(np.asarray(arrays[name]).dtype, np.number)
        ]

    inferred_protocol = spec.protocol
    if (
        inferred_protocol == "unknown"
        and TRAJECTORY_SIGNATURE.issubset(arrays.files)
    ):
        inferred_protocol = "label_free_frozen_score"
    if any(name.startswith("probe_") for name in score_fields):
        inferred_protocol = "supervised_diagnostic"
    for field in score_fields:
        short_name = field[6:] if field.startswith("score_") else field
        name = f"{spec.name}.{short_name}"
        methods[name] = MethodScore(
            name=name,
            values=np.asarray(arrays[field]),
            protocol=inferred_protocol,
            source_field=field,
            source_direction="higher",
        )
    return methods


def load_score_artifact(spec: ArtifactSpec) -> ScoreArtifact:
    """Load detector outputs only; labels in input files are never consumed."""

    path = Path(spec.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as arrays:
        sample_id = _first(arrays, ("sample_id", "sample_ids"))
        token_index = _first(arrays, ("token_index", "position", "positions"))
        schema = _scalar_text(arrays, "schema", "unversioned-npz")
        row_count = len(sample_id)
        if spec.adapter not in {"auto", "generic"}:
            raise ValueError(f"unsupported artifact adapter: {spec.adapter}")
        methods = (
            _configured_methods(spec, arrays)
            if spec.methods
            else _automatic_methods(spec, arrays, schema, row_count)
        )
        metadata = {}
        for output_name, candidates in {
            "source_id": ("source_id", "source_ids"),
            "task_type": ("task_type",),
            "data_source": ("data_source",),
            "generator_model": ("generator_model",),
        }.items():
            value = _first(arrays, candidates, required=False)
            if value is not None:
                metadata[output_name] = value
        artifact = ScoreArtifact(
            name=spec.name,
            path=str(path),
            schema=schema,
            sample_id=sample_id.copy(),
            token_index=token_index.copy(),
            methods=methods,
            metadata={name: value.copy() for name, value in metadata.items()},
        )
    return artifact.validate()
