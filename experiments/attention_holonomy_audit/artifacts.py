"""Compact NPZ artifacts for the attention holonomy audit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .config import AuditConfig, EvaluationConfig, GraphConfig, ReferenceConfig, TransportConfig
from .reference import NuisanceReference
from .ridge import AffineMap
from .transport import TransportReference

REFERENCE_SCHEMA = "attention-holonomy-audit-reference-v1"
SCORE_SCHEMA = "attention-holonomy-audit-score-v1"
EVALUATION_SCHEMA = "attention-holonomy-audit-evaluation-v1"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_npz(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _config_from_dict(value: dict[str, object]) -> AuditConfig:
    return AuditConfig(
        graph=GraphConfig(**value["graph"]),
        transport=TransportConfig(**value["transport"]),
        reference=ReferenceConfig(**value["reference"]),
        evaluation=EvaluationConfig(**value["evaluation"]),
    )


def transport_arrays(reference: TransportReference) -> dict[str, np.ndarray]:
    def stack(maps: tuple[AffineMap, ...], name: str) -> dict[str, np.ndarray]:
        return {
            f"{name}_weight": np.stack([value.weight for value in maps]),
            f"{name}_bias": np.stack([value.bias for value in maps]),
            f"{name}_mean": np.stack([value.target_mean for value in maps]),
            f"{name}_count": np.asarray([value.count for value in maps], dtype=np.int64),
        }

    result = {}
    result.update(stack(reference.depth, "depth"))
    relay_flat = tuple(value for pair in reference.relay for value in pair)
    relay = stack(relay_flat, "relay")
    layers = max(reference.num_layers - 1, 0)
    for name, value in relay.items():
        if name.endswith("_weight"):
            result[name] = value.reshape(layers, 2, reference.num_heads, reference.num_heads)
        elif name.endswith("_bias") or name.endswith("_mean"):
            result[name] = value.reshape(layers, 2, reference.num_heads)
        else:
            result[name] = value.reshape(layers, 2)
    result.update(stack(reference.query_local, "query_local"))
    result.update(stack(reference.query_full, "query_full"))
    return result


def _maps(arrays: dict[str, np.ndarray], name: str) -> tuple[AffineMap, ...]:
    return tuple(
        AffineMap(
            weight=arrays[f"{name}_weight"][index],
            bias=arrays[f"{name}_bias"][index],
            target_mean=arrays[f"{name}_mean"][index],
            count=int(arrays[f"{name}_count"][index]),
        )
        for index in range(len(arrays[f"{name}_count"]))
    )


def load_reference(path) -> tuple[dict[str, np.ndarray], AuditConfig, TransportReference, NuisanceReference]:
    arrays = load_npz(path)
    if str(arrays["schema"].item()) != REFERENCE_SCHEMA:
        raise ValueError("unsupported attention holonomy reference")
    config = _config_from_dict(json.loads(str(arrays["config_json"].item())))
    layers = int(arrays["num_layers"])
    heads = int(arrays["num_heads"])
    depth = _maps(arrays, "depth")
    relay_weight = arrays["relay_weight"]
    relay_bias = arrays["relay_bias"]
    relay_mean = arrays["relay_mean"]
    relay_count = arrays["relay_count"]
    relay = tuple(
        tuple(
            AffineMap(
                weight=relay_weight[layer, role],
                bias=relay_bias[layer, role],
                target_mean=relay_mean[layer, role],
                count=int(relay_count[layer, role]),
            )
            for role in range(2)
        )
        for layer in range(max(layers - 1, 0))
    )
    transport = TransportReference(
        depth=depth,
        relay=relay,
        query_local=_maps(arrays, "query_local"),
        query_full=_maps(arrays, "query_full"),
        num_layers=layers,
        num_heads=heads,
        graph_config=config.graph,
        transport_config=config.transport,
    )
    nuisance = NuisanceReference(
        task_names=tuple(arrays["task_names"].astype(str).tolist()),
        coefficient=arrays["nuisance_coefficient"],
        residual_median=arrays["residual_median"],
        residual_scale=arrays["residual_scale"],
        position_degree=int(arrays["position_degree"]),
    )
    return arrays, config, transport, nuisance
