"""File-level extraction pipeline with explicit progress and atomic manifest."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .data import load_attention_sample
from .routing import AnchorSpec, encode_route_dynamics


SCHEMA = "trajectory-geometry-route-dynamics-v1"


def extract_one(
    attention_path: str | Path,
    output_path: str | Path,
    *,
    spec: AnchorSpec,
    embedding_dim: int,
    seed: int,
    save_raw_route: bool,
) -> dict[str, object]:
    sample = load_attention_sample(attention_path)
    dynamics = encode_route_dynamics(
        sample, spec=spec, embedding_dim=embedding_dim, seed=seed
    )
    payload: dict[str, object] = {
        "schema": np.asarray(SCHEMA),
        "sample_id": np.asarray(sample.sample_id),
        "response_idx": np.asarray(sample.response_idx, dtype=np.int32),
        "token_count": np.asarray(sample.token_count, dtype=np.int32),
        "layers": np.asarray(sample.layers, dtype=np.int16),
        "heads": np.asarray(sample.heads, dtype=np.int16),
        "attention_floor": np.asarray(sample.attention_floor, dtype=np.float32),
        "anchor_names": np.asarray(dynamics.anchor_names),
        "route_embedding": dynamics.route_embedding,
        "temporal_js": dynamics.temporal_js,
        "depth_js": dynamics.depth_js,
        "head_js": dynamics.head_js,
        "route_acceleration": dynamics.route_acceleration,
        "prompt_mass": dynamics.prompt_mass,
        "history_mass": dynamics.history_mass,
        "self_mass": dynamics.self_mass,
        "unresolved_mass": dynamics.unresolved_mass,
        "mass_overflow": dynamics.mass_overflow,
        "embedding_dim": np.asarray(embedding_dim, dtype=np.int32),
        "projection_seed": np.asarray(seed, dtype=np.int64),
    }
    if save_raw_route:
        payload["raw_route_mass"] = dynamics.route_mass
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)
    return {
        "sample_id": sample.sample_id,
        "source": str(sample.path),
        "output": str(output_path.resolve()),
        "response_tokens": sample.response_tokens,
    }


def extract_many(
    files: list[Path],
    output_dir: str | Path,
    *,
    spec: AnchorSpec,
    embedding_dim: int,
    seed: int,
    save_raw_route: bool,
) -> dict[str, object]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, object]] = []
    total = len(files)
    for index, path in enumerate(files, start=1):
        output = output_dir / f"{path.stem}_route_geometry.npz"
        before = time.time()
        row = extract_one(
            path,
            output,
            spec=spec,
            embedding_dim=embedding_dim,
            seed=seed,
            save_raw_route=save_raw_route,
        )
        rows.append(row)
        elapsed = time.time() - before
        print(
            f"[{index}/{total}] {path.name} -> {output.name} "
            f"({row['response_tokens']} response tokens, {elapsed:.2f}s)",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA,
        "state": "complete",
        "samples": total,
        "embedding_dim": embedding_dim,
        "projection_seed": seed,
        "prompt_bins": spec.prompt_bins,
        "history_lag_edges": list(spec.history_lag_edges),
        "save_raw_route": save_raw_route,
        "elapsed_seconds": time.time() - started,
        "records": rows,
    }
    manifest_path = output_dir / "manifest.json"
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
