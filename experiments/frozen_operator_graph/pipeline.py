"""End-to-end label-free construction of exact frozen operator graphs."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import torch

from .artifacts import (
    canonical_json_sha256,
    save_graph_artifact,
    sha256,
    write_split_manifest,
)
from .basis import extract_operator_basis
from .binding import validate_exact_attention_against_cache
from .capture import ExactLlamaReplay
from .config import GraphConstructionConfig
from .encoding import build_node_encoding
from .graph import build_graph_tensors
from .schema import GRAPH_SCHEMA, GRAPH_VERSION, OperatorGraphArtifact



def _open_dataset(
    split_root: str | Path,
    *,
    verify_hashes: bool,
):
    from research_dataset import open_research_dataset

    return open_research_dataset(
        split_root,
        device="cpu",
        verify_hashes=verify_hashes,
        retain_embedded_labels=False,
    )


@dataclass(frozen=True)
class PipelineReport:
    output_root: Path
    manifest: dict[str, Any]
    rows: Sequence[dict[str, Any]]


def _package_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError("model_dtype must be float32, float16, or bfloat16")
    return mapping[name]


def _feature_contract(artifact: OperatorGraphArtifact) -> dict[str, Any]:
    return {
        "edge_feature_names": list(artifact.edge_feature_names),
        "remainder_feature_names": list(artifact.remainder_feature_names),
        "route_feature_names": list(artifact.route_feature_names),
        "layer_feature_names": list(artifact.layer_feature_names),
        "temporal_feature_names": list(artifact.temporal_feature_names),
        "node_feature_names": list(artifact.node_feature_names),
        "edge_attention_code_width": int(artifact.edge_attention_code.shape[1]),
        "route_shape_without_token": list(artifact.route_features.shape[1:]),
        "layer_shape_without_token": list(artifact.layer_features.shape[1:]),
        "remainder_shape_without_layer_token": list(
            artifact.remainder_features.shape[2:]
        ),
        "final_hidden_width": int(artifact.final_hidden.shape[1]),
        "node_width": int(artifact.node_embedding.shape[1]),
    }


def _artifact_filename(sample_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]
    return f"graph_{int(ordinal):08d}_{digest}.pt"


def _selected_sample_ids(
    dataset: Any,
    *,
    sample_ids: Sequence[str] | None,
    limit: int | None,
) -> list[str]:
    available = [str(value) for value in dataset.sample_ids]
    if sample_ids:
        requested = [str(value) for value in sample_ids]
        missing = sorted(set(requested).difference(available))
        if missing:
            raise KeyError(f"requested sample IDs are absent: {missing[:10]}")
        selected = requested
    else:
        selected = available
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no samples selected for graph construction")
    return selected


def _prepare_output(root: Path, *, overwrite: bool) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    if root.exists():
        if not overwrite:
            if any(root.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {root}; pass --overwrite explicitly"
                )
        else:
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    samples = root / "samples"
    samples.mkdir()
    return root, samples


def construct_split(
    *,
    split_root: str | Path,
    source_json: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    device: str,
    model_dtype: str,
    config: GraphConstructionConfig | None = None,
    sample_ids: Sequence[str] | None = None,
    limit: int | None = None,
    verify_hashes: bool = False,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    revision: str | None = None,
    overwrite: bool = False,
) -> PipelineReport:
    """Construct a split without opening the evaluation label interface.

    Every sample is teacher-forced through the frozen checkpoint.  Construction
    stops on any cache/checkpoint mismatch, unsupported model path, missing
    attention output, message-conservation failure, or feature-contract drift.
    There is intentionally no approximate or reduced-data fallback.
    """

    config = GraphConstructionConfig() if config is None else config
    config.validate()
    split_path = Path(split_root).expanduser().resolve()
    manifest_path = split_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset split has no manifest: {split_path}")
    dataset_manifest_sha = sha256(manifest_path)
    source_json_path = Path(source_json).expanduser().resolve()
    if not source_json_path.is_file():
        raise ValueError(f"raw source JSONL is missing: {source_json_path}")
    source_dataset = {
        "path": str(source_json_path),
        "sha256": sha256(source_json_path),
        "role": "raw_source_provenance_only",
        "content_read_for_graph_features": False,
        "labels_read_during_construction": False,
    }

    # Formal caches may physically contain embedded labels; the canonical data
    # boundary seals them because retain_embedded_labels is explicitly false.
    dataset = _open_dataset(
        split_path,
        verify_hashes=verify_hashes,
    )
    selected = _selected_sample_ids(dataset, sample_ids=sample_ids, limit=limit)
    output_path, sample_output = _prepare_output(
        Path(output_root), overwrite=overwrite
    )

    replay = ExactLlamaReplay.from_pretrained(
        model_path,
        device=device,
        torch_dtype=_model_dtype(model_dtype),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    basis = extract_operator_basis(
        replay.model,
        checkpoint=replay.checkpoint,
        compute_device=device,
        compute_dtype=torch.float32,
    )
    checkpoint_identity = replay.checkpoint

    package_sha = _package_digest()
    rows: list[dict[str, Any]] = []
    expected_contract: dict[str, Any] | None = None
    try:
        for ordinal, sample_id in enumerate(selected):
            sample = dataset[sample_id]
            attention_sample = sample.attention()

            def bind(exact_attention: list[torch.Tensor]) -> dict[str, object]:
                return validate_exact_attention_against_cache(
                    attention_sample,
                    exact_attention,
                    absolute_tolerance=config.cache_binding_atol,
                ).as_dict()

            capture = None
            graph = None
            encoding = None
            artifact = None
            try:
                capture = replay.capture(
                    attention_sample.token_ids,
                    int(attention_sample.response_idx),
                    conservation_atol=config.conservation_atol,
                    conservation_rtol=config.conservation_rtol,
                    attention_validator=bind,
                )
                graph = build_graph_tensors(capture, basis, config)
                encoding = build_node_encoding(
                    graph,
                    eps=config.feature_epsilon,
                )
                provenance = {
                    "schema": GRAPH_SCHEMA,
                    "version": GRAPH_VERSION,
                    "method": "frozen_hypernetwork_operator_graph",
                    "checkpoint": replay.checkpoint,
                    "dataset_manifest_sha256": dataset_manifest_sha,
                    "source_dataset": dict(source_dataset),
                    "construction_config": config.as_dict(),
                    "package_sha256": package_sha,
                    "sample_ordinal": ordinal,
                    "labels_consumed_by_construction": False,
                    "exact_inputs": [
                        "teacher_forced_token_ids",
                        "full_eager_attention_probabilities",
                        "value_projection_states",
                        "o_projection_input",
                        "o_projection_weights_and_bias",
                        "residual_stream_states",
                        "pre_attention_normalized_states",
                        "pre_mlp_normalized_states",
                        "mlp_updates",
                        "final_normalized_hidden_states",
                    ],
                    "fallbacks_used": [],
                }
                audit = {
                    **dict(graph.audit),
                    "labels_consumed_by_construction": False,
                    "feature_contract_verified": True,
                    "fallbacks_used": [],
                }
                artifact = OperatorGraphArtifact(
                    sample_id=str(sample.sample_id),
                    source_id=str(sample.source_id),
                    metadata=dict(sample.metadata),
                    token_ids=capture.token_ids.long(),
                    response_start=int(capture.response_start),
                    edge_index=graph.edge_index,
                    edge_layer=graph.edge_layer,
                    edge_role=graph.edge_role,
                    edge_attention_code=graph.edge_attention_code,
                    edge_features=graph.edge_features,
                    edge_feature_names=tuple(graph.edge_feature_names),
                    remainder_features=graph.remainder_features,
                    remainder_feature_names=tuple(graph.remainder_feature_names),
                    route_features=graph.route_features,
                    route_feature_names=tuple(graph.route_feature_names),
                    layer_features=graph.layer_features,
                    layer_feature_names=tuple(graph.layer_feature_names),
                    temporal_features=encoding.temporal_features,
                    temporal_feature_names=tuple(encoding.temporal_feature_names),
                    final_hidden=graph.final_hidden,
                    node_embedding=encoding.node_embedding,
                    node_feature_names=tuple(encoding.node_feature_names),
                    audit=audit,
                    provenance=provenance,
                ).validate()
                contract = _feature_contract(artifact)
                if expected_contract is None:
                    expected_contract = contract
                elif canonical_json_sha256(contract) != canonical_json_sha256(
                    expected_contract
                ):
                    raise RuntimeError(
                        "feature contract changed across samples in one split"
                    )
                destination = sample_output / _artifact_filename(
                    str(sample_id), ordinal
                )
                row = save_graph_artifact(
                    destination,
                    artifact,
                    output_dtype=config.output_dtype,
                )
                row["path"] = destination.relative_to(output_path).as_posix()
                row["task_type"] = sample.task_type
                row["data_source"] = sample.data_source
                rows.append(row)
            finally:
                sample.release_attention()
                del capture, graph, encoding, artifact
                gc.collect()
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
    except Exception:
        # A partial split is scientifically unsafe because it can silently
        # change the sample population.  Keep no manifest that could be mistaken
        # for a complete artifact; sample files remain for debugging only.
        failure = {
            "complete": False,
            "processed": len(rows),
            "selected": len(selected),
            "labels_consumed_by_construction": False,
        }
        (output_path / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        del basis
        del replay
        gc.collect()
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    assert expected_contract is not None
    manifest = write_split_manifest(
        output_path,
        rows,
        checkpoint=checkpoint_identity,
        dataset_manifest_sha256=dataset_manifest_sha,
        configuration=config.as_dict(),
        feature_contract=expected_contract,
        source_dataset=source_dataset,
    )
    return PipelineReport(
        output_root=output_path,
        manifest=manifest,
        rows=tuple(rows),
    )


__all__ = ["PipelineReport", "construct_split"]
