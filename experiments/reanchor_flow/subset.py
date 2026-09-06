"""Resumable orchestration for label-free, head-resolved mechanism audits."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.common.ragtruth_alignment import load_source_info
from research_dataset import open_research_dataset

from .artifact_payload import save_native_audit
from .artifact_schema import METHOD_VERSION, NativeAuditMetadata
from .artifact_validation import validate_native_audit
from .artifacts import save_json
from .flow import FlowSignal
from .native import audit_native_target
from .native_world import load_native_world, save_native_world
from .route_plan import RouteBudget
from .sample_scan import SampleScan
from .subset_data import (
    SampleRecord,
    inspect_records,
    load_world_from_dataset,
    safe_sample_key,
    sample_tokens,
    select_records,
)
from .worlds import TargetContrast

MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA = 3


@dataclass(frozen=True)
class SubsetRunConfig:
    """Scientific and input identity shared by every target in one run."""

    model_id: str
    model_dtype: str
    tokenizer_id: str
    dataset_root: str
    source_info: str
    split: str
    tasks: tuple[str, ...]
    samples_per_task: int
    explicit_sample_ids: tuple[str, ...]
    selection_seed: int
    targets_per_sample: int
    target_policy: str
    max_response_tokens: int | None
    signal: FlowSignal
    carrier_scope: str
    coverage: float
    query_chunk: int
    route_budget: RouteBudget
    local_window: int
    saved_edges: int = 2048
    scan_only: bool = False

    def manifest_value(self) -> dict[str, object]:
        """Return the small JSON contract used to resume this exact run."""

        return {
            "method": METHOD_VERSION,
            "model": self.model_id,
            "model_dtype": self.model_dtype,
            "tokenizer": self.tokenizer_id,
            "dataset_root": self.dataset_root,
            "source_info": self.source_info,
            "split": self.split,
            "tasks": list(self.tasks),
            "samples_per_task": self.samples_per_task,
            "explicit_sample_ids": list(self.explicit_sample_ids),
            "selection_seed": self.selection_seed,
            "targets_per_sample": self.targets_per_sample,
            "target_policy": self.target_policy,
            "max_response_tokens": self.max_response_tokens,
            "flow_signal": self.signal.value,
            "carrier_scope": self.carrier_scope,
            "edge_coverage": self.coverage,
            "query_chunk": self.query_chunk,
            "route_budget": asdict(self.route_budget),
            "local_window": self.local_window,
            "saved_edges": self.saved_edges,
        }


def open_manifest(
    path: Path,
    config: dict[str, object],
    selection: tuple[SampleRecord, ...],
) -> dict:
    """Open or initialize the minimal resume ledger for a frozen selection."""

    rows = [asdict(record) for record in selection]
    if not path.is_file():
        return {
            "subset_manifest_schema": MANIFEST_SCHEMA,
            "config": config,
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
    if manifest.get("labels_used_for_capture") is not False:
        raise ValueError("capture manifest violates the label firewall")
    manifest["analysis_complete"] = False
    manifest.pop("counts", None)
    return manifest


def _target_key(target: TargetContrast, signal: FlowSignal) -> str:
    return (
        f"q{target.query_position}_a{target.positive_token_id}"
        f"_b{target.negative_token_id}_{signal.value}"
    )


def _target_selection_manifest(world, target_rank: int) -> dict | None:
    if not world.target_selection:
        return None
    selection = world.target_selection[target_rank]
    return {**asdict(selection), "is_center": selection.is_center}


def _model_matches(dataset, model_path: str | Path) -> bool:
    recorded = str(getattr(dataset, "spec", {}).get("model_path", ""))
    if not recorded:
        return True
    recorded_path = Path(recorded)
    requested_path = Path(model_path)
    if recorded_path.is_absolute():
        return recorded_path.resolve() == requested_path.resolve()
    return recorded_path.name == requested_path.name


def run_subset_split(
    model,
    tokenizer,
    output_root: str | Path,
    config: SubsetRunConfig,
) -> dict[str, int]:
    """Run resumable native mechanism audits on one real-data split."""

    dataset_root = Path(config.dataset_root)
    source_path = Path(config.source_info)
    output = Path(output_root)
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    dataset_split = str(dataset.manifest.get("split", "")).casefold()
    if dataset_split != config.split.casefold():
        raise ValueError(
            f"dataset split {dataset_split!r} differs from requested split "
            f"{config.split!r}"
        )
    if not _model_matches(dataset, config.model_id):
        raise ValueError("cached observer and current model differ")
    if Path(tokenizer.name_or_path).name != config.tokenizer_id:
        raise ValueError("configured and loaded tokenizers differ")
    sources = load_source_info(source_path)
    selected = select_records(
        inspect_records(
            dataset,
            sample_ids=config.explicit_sample_ids or None,
            source_info=sources,
        ),
        tasks=config.tasks,
        samples_per_task=config.samples_per_task,
        seed=config.selection_seed,
        sample_ids=config.explicit_sample_ids,
    )
    dataset.verify_hashes = True
    if len({record.sample_id for record in selected}) != len(selected):
        raise ValueError("subset selection contains duplicate sample IDs")
    for record in selected:
        if record.source_id not in sources:
            raise ValueError(
                f"source_info lacks {record.source_id} for {record.sample_id}"
            )

    manifest_path = output / MANIFEST_NAME
    manifest = open_manifest(manifest_path, config.manifest_value(), selected)
    manifest["analysis_scope"] = (
        "structure_only"
        if config.scan_only
        else "structure_and_selected_target_function"
    )
    save_json(manifest_path, manifest)

    counts = {"samples": 0, "targets": 0, "resumed": 0, "confirmed": 0}
    samples = tqdm(
        selected,
        desc=f"{config.split} samples",
        unit="sample",
        dynamic_ncols=True,
    )
    for record in samples:
        samples.set_postfix_str(f"{record.task_type}/{record.sample_id}", refresh=False)
        sample_key = safe_sample_key(record.sample_id)
        world_path = output / "worlds" / record.task_type / f"{sample_key}.npz"
        scan_path = output / "scans" / record.task_type / f"{sample_key}.npz"
        frozen_sample = manifest["samples"].get(record.sample_id)
        if world_path.is_file():
            world = load_native_world(world_path)
            if world.sample_id != sample_key:
                raise ValueError("saved native world has the wrong sample identity")
            if Path(world.tokenizer_id).name != config.tokenizer_id:
                raise ValueError("saved native world uses another tokenizer")
            if not scan_path.is_file():
                samples.set_postfix_str(
                    f"{record.task_type}/{record.sample_id} rebuild scan", refresh=True
                )
                full_tokens, response_start = sample_tokens(dataset, record.sample_id)
                scan = SampleScan.capture(
                    model,
                    world,
                    query_chunk=config.query_chunk,
                    local_window=config.local_window,
                )
                scan.save(
                    scan_path,
                    dataset_sample_id=record.sample_id,
                    source_id=record.source_id,
                    task_type=record.task_type,
                    token_ids=world.token_ids,
                    response_start=world.response_start,
                    full_response_tokens=len(full_tokens) - response_start,
                    local_window=config.local_window,
                )
                del full_tokens, scan
        else:
            if frozen_sample is not None:
                raise ValueError(f"frozen native world is missing: {world_path}")
            world = load_world_from_dataset(
                dataset,
                record,
                sources[record.source_id],
                tokenizer,
                model,
                max_response_tokens=config.max_response_tokens,
                targets_per_sample=config.targets_per_sample,
                target_policy=config.target_policy,
                query_chunk=config.query_chunk,
                local_window=config.local_window,
                scan_path=scan_path,
            )
            save_native_world(world_path, world)

        sample_entry = {
            "source_id": record.source_id,
            "task_type": record.task_type,
            "world": world_path.relative_to(output).as_posix(),
            "scan": scan_path.relative_to(output).as_posix(),
            "targets": [
                {
                    "query_position": target.query_position,
                    "positive_token_id": target.positive_token_id,
                    "negative_token_id": target.negative_token_id,
                    "contrast_origin": target.origin,
                    "reanchor_selection": _target_selection_manifest(world, rank),
                }
                for rank, target in enumerate(world.targets)
            ],
        }
        if frozen_sample is not None:
            # Runs created before sample scans retain their frozen target plan.
            frozen_sample.setdefault("scan", sample_entry["scan"])
            if frozen_sample != sample_entry:
                raise ValueError(
                    f"saved native world disagrees with frozen sample {record.sample_id}"
                )
        manifest["samples"][record.sample_id] = sample_entry
        save_json(manifest_path, manifest)
        if not config.scan_only:
            _run_world_targets(
                model,
                world,
                record,
                output,
                manifest,
                manifest_path,
                counts,
                config,
            )
        counts["samples"] += 1
        del world
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["analysis_complete"] = True
    manifest["counts"] = counts
    save_json(manifest_path, manifest)
    return counts


def _audit_identity(
    world,
    record: SampleRecord,
    target: TargetContrast,
    target_rank: int,
    destination: Path,
    output: Path,
    config: SubsetRunConfig,
) -> dict[str, object]:
    return {
        "result": destination.relative_to(output).as_posix(),
        "dataset_sample_id": record.sample_id,
        "sample_id": world.sample_id,
        "source_id": record.source_id,
        "task_type": record.task_type,
        "split": config.split,
        "query_position": target.query_position,
        "positive_token_id": target.positive_token_id,
        "negative_token_id": target.negative_token_id,
        "contrast_origin": target.origin,
        "reanchor_selection": _target_selection_manifest(world, target_rank),
        "flow_signal": config.signal.value,
        "target_rank": target_rank,
    }


def _run_world_targets(
    model,
    world,
    record: SampleRecord,
    output: Path,
    manifest: dict,
    manifest_path: Path,
    counts: dict[str, int],
    config: SubsetRunConfig,
) -> None:
    sample_key = safe_sample_key(record.sample_id)
    targets = tqdm(
        enumerate(world.targets),
        total=len(world.targets),
        desc=f"{config.split}/{record.task_type}/{record.sample_id} targets",
        unit="target",
        leave=False,
        dynamic_ncols=True,
    )
    for target_rank, target in targets:
        target_key = _target_key(target, config.signal)
        key = f"{record.sample_id}:{target_key}"
        destination = (
            output / "audits" / record.task_type / sample_key / f"{target_key}.npz"
        )
        identity = _audit_identity(
            world,
            record,
            target,
            target_rank,
            destination,
            output,
            config,
        )
        existing_entry = manifest["audits"].get(key)
        if existing_entry is not None and existing_entry != identity:
            raise ValueError(f"subset manifest audit {key} has another identity")

        metadata = NativeAuditMetadata(
            dataset_sample_id=record.sample_id,
            source_id=record.source_id,
            split=config.split,
            task_type=record.task_type,
            generator_model=record.generator_model,
            model_id=config.model_id,
            model_dtype=config.model_dtype,
            target_policy=config.target_policy,
            target_rank=target_rank,
            coverage=config.coverage,
            carrier_scope=config.carrier_scope,
            query_chunk=config.query_chunk,
            route_budget=config.route_budget,
            local_window=config.local_window,
            saved_edges=config.saved_edges,
        )
        if destination.is_file():
            validate_native_audit(
                destination,
                world,
                target,
                config.signal,
                metadata,
            )
            counts["targets"] += 1
            counts["resumed"] += 1
            targets.set_postfix_str("resumed", refresh=False)
            with np.load(destination, allow_pickle=False) as stored:
                counts["confirmed"] += int(stored["corridor_confirmed"])
        else:
            result = audit_native_target(
                model,
                world,
                target,
                config.signal,
                carrier_scope=config.carrier_scope,
                coverage=config.coverage,
                query_chunk=config.query_chunk,
                route_budget=config.route_budget,
                local_window=config.local_window,
                on_phase=lambda phase: targets.set_postfix_str(phase, refresh=True),
            )
            save_native_audit(
                destination,
                world,
                result,
                metadata,
            )
            validate_native_audit(
                destination,
                world,
                target,
                config.signal,
                metadata,
            )
            counts["targets"] += 1
            counts["confirmed"] += int(result.corridor_confirmed)
            targets.set_postfix_str("computed", refresh=False)
            prefix = (
                f"{config.split}/{record.task_type}/{record.sample_id} "
                f"q={target.query_position} signal={config.signal.value} "
                f"root={result.selected_root_unit_id}"
            )
            if config.route_budget.confirm:
                detail = (
                    f"root_ok={result.selected_root_confirmed} "
                    f"corridor_ok={result.corridor_confirmed} "
                    f"restore={result.effect.restoration_error:.4g}"
                )
            else:
                detail = (
                    f"corridor_edges={result.corridor.count} "
                    f"hubs={len(result.plan.hubs)} exact=not-run"
                )
            tqdm.write(f"{prefix} {detail}")
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest["audits"][key] = identity
        save_json(manifest_path, manifest)
