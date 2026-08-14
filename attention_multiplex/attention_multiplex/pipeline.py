"""Dataset pipeline for attention-only multiplex representations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
from tqdm.auto import tqdm

from .representation import (
    REPRESENTATION_SCHEMA,
    MultiplexConfig,
    represent_attention_multiplex,
)


MANIFEST_SCHEMA = "attention-dynamic-multiplex-dataset-v1"


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _selected_ids(dataset, limit=None):
    sample_ids = list(dataset.sample_ids)
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("no attention samples selected")
    return sample_ids


def build_dataset_representations(
    dataset,
    output_dir,
    *,
    config: MultiplexConfig | None = None,
    limit=None,
):
    """Encode a split without opening or serializing hallucination labels."""

    config = MultiplexConfig() if config is None else config
    config.validate()
    output_dir = Path(output_dir)
    node_dir = output_dir / "samples"
    node_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = _selected_ids(dataset, limit)
    index_rows = []
    geometry = None

    for sample_id in tqdm(sample_ids, desc="attention multiplex", unit="sample"):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            current_geometry = (int(attention.num_layers), int(attention.num_heads))
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError("layer/head geometry changes inside the split")
            representation = represent_attention_multiplex(sample, config=config)
            output_path = node_dir / f"multiplex_{sample_id}.npz"
            token_ids = attention.token_ids.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            _atomic_npz(
                output_path,
                schema=np.asarray(REPRESENTATION_SCHEMA),
                labels_included=np.asarray(False),
                sample_id=np.asarray(str(sample_id)),
                source_id=np.asarray(str(sample.source_id)),
                response_idx=np.asarray(representation.response_idx, dtype=np.int32),
                token_ids=token_ids,
                mass_query_by_layer=representation.mass.query_by_layer,
                mass_source_by_head=representation.mass.source_by_head,
                mass_singular_values=representation.mass.singular_values,
                mass_captured_energy=np.asarray(
                    representation.mass.captured_energy, dtype=np.float32
                ),
                shape_query_by_layer=representation.shape.query_by_layer,
                shape_source_by_head=representation.shape.source_by_head,
                shape_singular_values=representation.shape.singular_values,
                shape_captured_energy=np.asarray(
                    representation.shape.captured_energy, dtype=np.float32
                ),
                self_attention=representation.self_attention,
                retained_row_mass=representation.retained_row_mass,
                unresolved_row_mass=representation.unresolved_row_mass,
                attention_floor=np.asarray(
                    attention.attention_floor, dtype=np.float32
                ),
            )
            index_rows.append(
                {
                    "sample_id": str(sample_id),
                    "source_id": str(sample.source_id),
                    "path": output_path.relative_to(output_dir).as_posix(),
                    "response_tokens": int(attention.num_response_tokens),
                    "tokens": int(attention.num_tokens),
                    "retained_off_diagonal_edges": int(
                        representation.retained_off_diagonal_edges
                    ),
                    "mass_captured_energy": float(
                        representation.mass.captured_energy
                    ),
                    "shape_captured_energy": float(
                        representation.shape.captured_energy
                    ),
                    "task_type": _json_value(sample.task_type),
                    "data_source": _json_value(sample.data_source),
                }
            )
        finally:
            sample.release_attention()

    index_path = output_dir / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "representation_schema": REPRESENTATION_SCHEMA,
        "state": "complete",
        "labels_included": False,
        "samples": len(index_rows),
        "layers": geometry[0],
        "heads": geometry[1],
        "rank": int(config.rank),
        "include_diagonal": bool(config.include_diagonal),
        "attention_input": "canonical ResearchDataset/ResearchSample only",
        "query_role": "layer x response_token",
        "source_role": "head x prompt_plus_response_token",
        "pp_policy": "not_available_not_fabricated",
        "censoring_policy": (
            "legal missing edges equal attention_floor in the reconstructed view; "
            "spectral input stores only excess over that floor"
        ),
        "cross_sample_alignment": False,
        "index": index_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "index": str(index_path),
        "sample_directory": str(node_dir),
        "samples": len(index_rows),
        "layers": geometry[0],
        "heads": geometry[1],
        "rank": int(config.rank),
    }
