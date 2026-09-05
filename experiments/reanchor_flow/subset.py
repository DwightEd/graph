"""Resumable orchestration for the label-free native mechanism subset."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.ragtruth_alignment import TASK_TYPES, load_source_info
from research_dataset import open_research_dataset

from .flow import FlowSignal
from .native import audit_native_target
from .native_world import load_native_world, save_native_world
from .subset_artifacts import (
    MANIFEST_SCHEMA,
    capture_config_sha256,
    save_compact_native_audit,
    validate_compact_native_audit,
)
from .subset_data import (
    SampleRecord,
    file_sha256,
    inspect_records,
    load_world_from_dataset,
    safe_sample_key,
    select_records,
)
from .worlds import TargetContrast

MANIFEST_NAME = "run_manifest.json"


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_manifest(
    path: Path,
    config: dict,
    selection: tuple[SampleRecord, ...],
) -> dict:
    rows = [record.__dict__ for record in selection]
    config_sha256 = capture_config_sha256(config)
    if not path.is_file():
        orphan = next(
            (
                artifact
                for directory in ("worlds", "audits")
                for artifact in (path.parent / directory).rglob("*.npz")
            ),
            None,
        )
        if orphan is not None:
            raise ValueError(
                "output has mechanism artifacts but no manifest; choose a new --output"
            )
        return {
            "subset_manifest_schema": MANIFEST_SCHEMA,
            "config": config,
            "config_sha256": config_sha256,
            "selection": rows,
            "labels_used_for_capture": False,
            "analysis_complete": False,
            "samples": {},
            "audits": {},
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("subset_manifest_schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported subset manifest schema")
    if manifest.get("config") != config or manifest.get("selection") != rows:
        raise ValueError(
            "output contains another subset configuration; choose a new --output"
        )
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("subset manifest configuration fingerprint is invalid")
    if manifest.get("labels_used_for_capture") is not False:
        raise ValueError("capture manifest violates the label firewall")
    if not isinstance(manifest.get("samples"), dict) or not isinstance(
        manifest.get("audits"), dict
    ):
        raise ValueError("subset manifest inventory is invalid")
    manifest["analysis_complete"] = False
    manifest.pop("counts", None)
    return manifest


def _target_key(target: TargetContrast, signal: FlowSignal) -> str:
    return (
        f"q{target.query_position}_a{target.positive_token_id}"
        f"_b{target.negative_token_id}_{signal.value}"
    )


def _model_matches(dataset, model_path: str | Path) -> bool:
    recorded = str(getattr(dataset, "spec", {}).get("model_path", ""))
    if not recorded:
        return True
    recorded_path = Path(recorded)
    requested_path = Path(model_path)
    if recorded_path.is_absolute():
        return recorded_path.resolve() == requested_path.resolve()
    return recorded_path.name == requested_path.name


def _configuration(
    tokenizer,
    dataset_root: Path,
    source_path: Path,
    *,
    split: str,
    model_path: str | Path,
    model_dtype: str,
    tasks: tuple[str, ...],
    samples_per_task: int,
    sample_ids: tuple[str, ...],
    selection_seed: int,
    targets_per_sample: int,
    target_policy: str,
    max_response_tokens: int | None,
    signal: FlowSignal,
    carrier_scope: str,
    coverage: float,
    query_chunk: int,
    root_screen_limit: int,
    carrier_limit: int,
    saved_edges: int,
) -> dict:
    return {
        "model": str(Path(model_path).resolve()),
        "model_dtype": model_dtype,
        "tokenizer": Path(tokenizer.name_or_path).name,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "source_info": str(source_path.resolve()),
        "source_info_sha256": file_sha256(source_path),
        "split": split,
        "tasks": list(tasks),
        "samples_per_task": samples_per_task,
        "explicit_sample_ids": list(sample_ids),
        "selection_seed": selection_seed,
        "targets_per_sample": targets_per_sample,
        "target_policy": target_policy,
        "max_response_tokens": max_response_tokens,
        "flow_signal": signal.value,
        "carrier_scope": carrier_scope,
        "edge_coverage": coverage,
        "query_chunk": query_chunk,
        "root_screen_limit": root_screen_limit,
        "carrier_limit": carrier_limit,
        "saved_edges": saved_edges,
        "world": "native_source_value_cut_v1",
        "contrast": "observed_token_vs_native_runner_v1",
        "functional_edge": "native_gradient_dot_true_message_v1",
        "capture_metadata_access": (
            "formal_mmap_weights_only_scalar_whitelist_when_available"
        ),
        "capture_label_access": (
            "sidecar_unopened; formal_y_token_storage_not_dereferenced_or_validated"
        ),
        "selected_payload_hash_verification": True,
    }


def run_subset_split(
    model,
    tokenizer,
    dataset_root: str | Path,
    source_info: str | Path,
    output_root: str | Path,
    *,
    split: str,
    model_path: str | Path,
    model_dtype: str,
    tasks: tuple[str, ...] = TASK_TYPES,
    samples_per_task: int = 1,
    sample_ids: tuple[str, ...] = (),
    selection_seed: int = 2026,
    targets_per_sample: int = 1,
    target_policy: str = "uncertain",
    max_response_tokens: int | None = 128,
    signal: FlowSignal | str = FlowSignal.MESSAGE,
    carrier_scope: str = "response",
    coverage: float = 0.9,
    query_chunk: int = 8,
    root_screen_limit: int = 4,
    carrier_limit: int = 2,
    saved_edges: int = 2048,
) -> dict[str, int]:
    """Run resumable compact native audits on one real-data split."""

    signal = FlowSignal(signal)
    dataset_root = Path(dataset_root)
    source_path = Path(source_info)
    output = Path(output_root)
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    dataset_split = str(dataset.manifest.get("split", "")).casefold()
    if dataset_split != str(split).casefold():
        raise ValueError(
            f"dataset split {dataset_split!r} differs from requested split {split!r}"
        )
    if not _model_matches(dataset, model_path):
        raise ValueError("cached observer and current model differ")
    sources = load_source_info(source_path)
    selected = select_records(
        inspect_records(
            dataset,
            sample_ids=sample_ids or None,
            source_info=sources,
        ),
        tasks=tasks,
        samples_per_task=samples_per_task,
        seed=selection_seed,
        sample_ids=sample_ids,
    )
    dataset.verify_hashes = True
    if len({record.sample_id for record in selected}) != len(selected):
        raise ValueError("subset selection contains duplicate sample IDs")
    expected_audits: set[str] = set()
    for record in selected:
        if record.source_id not in sources:
            raise ValueError(
                f"source_info lacks {record.source_id} for {record.sample_id}"
            )
    config = _configuration(
        tokenizer,
        dataset_root,
        source_path,
        split=split,
        model_path=model_path,
        model_dtype=model_dtype,
        tasks=tasks,
        samples_per_task=samples_per_task,
        sample_ids=sample_ids,
        selection_seed=selection_seed,
        targets_per_sample=targets_per_sample,
        target_policy=target_policy,
        max_response_tokens=max_response_tokens,
        signal=signal,
        carrier_scope=carrier_scope,
        coverage=coverage,
        query_chunk=query_chunk,
        root_screen_limit=root_screen_limit,
        carrier_limit=carrier_limit,
        saved_edges=saved_edges,
    )
    manifest_path = output / MANIFEST_NAME
    manifest = open_manifest(manifest_path, config, selected)
    _save_json(manifest_path, manifest)

    counts = {"samples": 0, "targets": 0, "resumed": 0, "confirmed": 0}
    for record in selected:
        sample_key = safe_sample_key(record.sample_id)
        world_path = output / "worlds" / record.task_type / f"{sample_key}.npz"
        frozen_sample = manifest["samples"].get(record.sample_id)
        if world_path.is_file():
            world = load_native_world(world_path)
            if world.sample_id != sample_key:
                raise ValueError("saved native world has the wrong sample identity")
            if Path(world.tokenizer_id).name != Path(tokenizer.name_or_path).name:
                raise ValueError("saved native world uses another tokenizer")
        else:
            if frozen_sample is not None:
                raise ValueError(f"frozen native world is missing: {world_path}")
            world = load_world_from_dataset(
                dataset,
                record,
                sources[record.source_id],
                tokenizer,
                model,
                max_response_tokens=max_response_tokens,
                targets_per_sample=targets_per_sample,
                target_policy=target_policy,
                query_chunk=query_chunk,
            )
            save_native_world(world_path, world)
        world_sha256 = file_sha256(world_path)

        sample_entry = {
            "source_id": record.source_id,
            "task_type": record.task_type,
            "world": world_path.relative_to(output).as_posix(),
            "world_sha256": world_sha256,
            "targets": [
                {
                    "query_position": target.query_position,
                    "positive_token_id": target.positive_token_id,
                    "negative_token_id": target.negative_token_id,
                    "contrast_origin": target.origin,
                }
                for target in world.targets
            ],
        }
        if frozen_sample is not None and frozen_sample != sample_entry:
            raise ValueError(
                f"saved native world disagrees with frozen sample {record.sample_id}"
            )
        manifest["samples"][record.sample_id] = sample_entry
        _save_json(manifest_path, manifest)
        _run_world_targets(
            model,
            world,
            record,
            output,
            manifest,
            manifest_path,
            counts,
            expected_audits,
            split=split,
            model_path=model_path,
            model_dtype=model_dtype,
            signal=signal,
            target_policy=target_policy,
            carrier_scope=carrier_scope,
            coverage=coverage,
            query_chunk=query_chunk,
            root_screen_limit=root_screen_limit,
            carrier_limit=carrier_limit,
            saved_edges=saved_edges,
            world_sha256=world_sha256,
            capture_config=config,
        )
        counts["samples"] += 1

    expected_samples = {record.sample_id for record in selected}
    if set(manifest["samples"]) != expected_samples:
        raise ValueError(
            "subset manifest contains samples outside the frozen selection"
        )
    if set(manifest["audits"]) != expected_audits:
        raise ValueError("subset manifest audit inventory is inconsistent")
    manifest["analysis_complete"] = True
    manifest["counts"] = counts
    _save_json(manifest_path, manifest)
    return counts


def _run_world_targets(
    model,
    world,
    record: SampleRecord,
    output: Path,
    manifest: dict,
    manifest_path: Path,
    counts: dict[str, int],
    expected_audits: set[str],
    **options,
) -> None:
    signal = options["signal"]
    sample_key = safe_sample_key(record.sample_id)
    for target_rank, target in enumerate(world.targets):
        target_key = _target_key(target, signal)
        key = f"{record.sample_id}:{target_key}"
        expected_audits.add(key)
        destination = (
            output / "audits" / record.task_type / sample_key / f"{target_key}.npz"
        )
        relative_destination = destination.relative_to(output).as_posix()
        manifest_identity = {
            "result": relative_destination,
            "complete": True,
            "dataset_sample_id": record.sample_id,
            "sample_id": world.sample_id,
            "source_id": record.source_id,
            "task_type": record.task_type,
            "generator_model": record.generator_model,
            "split": options["split"],
            "query_position": target.query_position,
            "positive_token_id": target.positive_token_id,
            "negative_token_id": target.negative_token_id,
            "contrast_origin": target.origin,
            "flow_signal": signal.value,
            "target_rank": target_rank,
            "world_sha256": options["world_sha256"],
            "config_sha256": capture_config_sha256(options["capture_config"]),
        }
        existing_entry = manifest["audits"].get(key)
        if existing_entry is not None:
            for name, expected in manifest_identity.items():
                if existing_entry.get(name) != expected:
                    raise ValueError(
                        f"subset manifest audit {key} has inconsistent {name}"
                    )
            if not isinstance(existing_entry.get("sha256"), str):
                raise ValueError(f"subset manifest audit {key} lacks an artifact hash")

        validation = {
            "dataset_sample_id": record.sample_id,
            "sample_id": world.sample_id,
            "source_id": record.source_id,
            "split": options["split"],
            "task_type": record.task_type,
            "generator_model": record.generator_model,
            "tokenizer_id": world.tokenizer_id,
            "world_sha256": options["world_sha256"],
            "target": target,
            "target_rank": target_rank,
            "signal": signal,
            "model_id": str(Path(options["model_path"]).resolve()),
            "model_dtype": options["model_dtype"],
            "capture_config": options["capture_config"],
        }
        if destination.is_file():
            validate_compact_native_audit(destination, **validation)
            artifact_sha256 = file_sha256(destination)
            if (
                existing_entry is not None
                and existing_entry["sha256"] != artifact_sha256
            ):
                raise ValueError(f"subset artifact hash mismatch for {key}")
            counts["targets"] += 1
            counts["resumed"] += 1
            with np.load(destination, allow_pickle=False) as stored:
                counts["confirmed"] += int(stored["corridor_confirmed"])
        else:
            result = audit_native_target(
                model,
                world,
                target,
                signal,
                carrier_scope=options["carrier_scope"],
                coverage=options["coverage"],
                query_chunk=options["query_chunk"],
                root_screen_limit=options["root_screen_limit"],
                carrier_limit=options["carrier_limit"],
            )
            save_compact_native_audit(
                destination,
                world,
                result,
                dataset_sample_id=record.sample_id,
                source_id=record.source_id,
                split=options["split"],
                task_type=record.task_type,
                generator_model=record.generator_model,
                model_id=str(Path(options["model_path"]).resolve()),
                model_dtype=options["model_dtype"],
                target_policy=options["target_policy"],
                target_rank=target_rank,
                coverage=options["coverage"],
                carrier_scope=options["carrier_scope"],
                query_chunk=options["query_chunk"],
                root_screen_limit=options["root_screen_limit"],
                carrier_limit=options["carrier_limit"],
                saved_edges=options["saved_edges"],
                world_sha256=options["world_sha256"],
                capture_config=options["capture_config"],
            )
            validate_compact_native_audit(destination, **validation)
            artifact_sha256 = file_sha256(destination)
            counts["targets"] += 1
            counts["confirmed"] += int(result.corridor_confirmed)
            print(
                f"{options['split']}/{record.task_type}/{record.sample_id} "
                f"q={target.query_position} signal={signal.value} "
                f"root={result.selected_root_unit_id} "
                f"root_ok={result.selected_root_confirmed} "
                f"corridor_ok={result.corridor_confirmed} "
                f"restore={result.effect.restoration_error:.4g}"
            )
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest["audits"][key] = {
            **manifest_identity,
            "sha256": artifact_sha256,
        }
        _save_json(manifest_path, manifest)
