"""Strict owner-dispatched score artifacts for the conditioned benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment_protocol import FrozenFile
from experiments.causal_multiplex_flow.artifacts import (
    load_score_artifact as load_cmrp_score,
)
from experiments.rr_topology_dynamics.artifacts import (
    load_topology_artifact,
)
from experiments.spectral_feasibility.artifacts import (
    load_score_artifact as load_spectral_score,
)

from .types import MethodScore, ScoreArtifact


@dataclass(frozen=True)
class ArtifactSpec:
    """One current v2 artifact and any explicitly oriented RR features."""

    name: str
    path: str
    column: str | None = None
    direction: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ArtifactSpec:
        allowed = {"name", "path", "column", "direction"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unsupported artifact settings: {sorted(unknown)}")
        if "name" not in value or "path" not in value:
            raise ValueError("each artifact requires name and path")
        return cls(
            name=str(value["name"]),
            path=str(value["path"]),
            column=(None if value.get("column") is None else str(value["column"])),
            direction=(
                None if value.get("direction") is None else str(value["direction"])
            ),
        )


def _scalar_text(arrays, name: str) -> str:
    if name not in arrays:
        raise ValueError(f"artifact misses field {name!r}")
    value = np.asarray(arrays[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"artifact field {name!r} must be scalar text")
    return str(value.item())


def _schema(path: Path) -> str:
    with np.load(path, allow_pickle=False) as arrays:
        return _scalar_text(arrays, "schema")


def _primary_method(spec: ArtifactSpec, arrays, field: str) -> dict[str, MethodScore]:
    if spec.column is not None or spec.direction is not None:
        raise ValueError(
            "column and direction are only valid for RR topology artifacts"
        )
    name = f"{spec.name}.primary"
    return {
        name: MethodScore(
            name=name,
            values=np.asarray(arrays[field]),
            protocol="label_free_frozen_score",
            source_field=field,
            source_direction="higher",
        )
    }


def _feature_column(arrays, column: str):
    values = np.asarray(arrays["features_z"])
    names = np.asarray(arrays["feature_names"]).astype(str)
    matches = np.flatnonzero(names == column)
    if len(matches) != 1:
        raise ValueError(f"features_z column {column!r} is missing or ambiguous")
    return values[:, int(matches[0])]


def _topology_methods(spec: ArtifactSpec, arrays) -> dict[str, MethodScore]:
    if spec.column is None or spec.direction is None:
        raise ValueError(
            "RR topology artifacts require a features_z column and direction"
        )
    if spec.direction not in {"higher", "lower"}:
        raise ValueError("RR topology direction must be higher or lower")
    name = f"{spec.name}.{spec.column}"
    return {
        name: MethodScore(
            name=name,
            values=_feature_column(arrays, spec.column),
            direction=spec.direction,
            protocol="label_free_feature_fixed_direction",
            source_field="features_z",
            source_direction=spec.direction,
        )
    }


def load_score_artifact(spec: ArtifactSpec, frozen: FrozenFile) -> ScoreArtifact:
    """Strict-load one captured current-v2 artifact through its owner contract."""

    frozen.verify(spec.path)
    schema = _schema(frozen.path)
    if schema == "cmrp-score-v2":
        arrays = load_cmrp_score(frozen.path)
        methods = _primary_method(spec, arrays, "score")
    elif schema == "rr-spectral-score-v2":
        arrays = load_spectral_score(frozen.path)
        methods = _primary_method(spec, arrays, "score_rr_residual")
    elif schema == "rr-topology-dynamics-features-v2":
        arrays = load_topology_artifact(frozen.path)
        methods = _topology_methods(spec, arrays)
    else:
        raise ValueError(f"unsupported conditioned benchmark artifact schema: {schema}")
    frozen.verify(frozen.path)

    artifact = ScoreArtifact(
        name=spec.name,
        path=str(frozen.path),
        schema=schema,
        sample_id=np.asarray(arrays["sample_id"]).copy(),
        source_id=np.asarray(arrays["source_id"]).copy(),
        token_index=np.asarray(arrays["token_index"]).copy(),
        response_length=np.asarray(arrays["response_length"]).copy(),
        dataset_manifest_sha256=_scalar_text(arrays, "dataset_manifest_sha256"),
        methods=methods,
    )
    return artifact.validate()
