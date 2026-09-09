"""Resumable orchestration for label-free, head-resolved mechanism audits."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .artifact_payload import save_native_audit
from .artifact_schema import METHOD_VERSION, NativeAuditMetadata
from .artifact_validation import validate_native_audit
from .artifacts import safe_sample_key, save_json
from .dataset import AuditCorpus, AuditSample, CorpusIdentity, SampleRecord, select_records
from .flow import FlowSignal
from .native import audit_native_target
from .native_world import NativeWorld, load_native_world, save_native_world
from .route_plan import RouteBudget
from .sample_scan import SampleScan
from .target_selection import freeze_target_plan
from .worlds import TargetContrast

MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA = 3


@dataclass(frozen=True)
class CohortPlan:
    """Label-free sample-selection policy."""

    tasks: tuple[str, ...]
    samples_per_task: int
    sample_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class TargetPlan:
    """Teacher-forced response rows to freeze before functional analysis."""

    count: int
    policy: str
    max_response_tokens: int | None


@dataclass(frozen=True)
class MechanismPlan:
    """Route-capture, intervention, and persistence settings."""

    signal: FlowSignal
    carrier_scope: str
    coverage: float
    query_chunk: int
    route_budget: RouteBudget
    local_window: int
    saved_edges: int = 2048


@dataclass(frozen=True)
class SubsetRunConfig:
    """One readable run contract instead of a flat parameter bundle."""

    model_id: str
    model_dtype: str
    tokenizer_id: str
    cohort: CohortPlan
    targets: TargetPlan
    mechanism: MechanismPlan
    scan_only: bool = False

    def manifest_value(self, corpus: CorpusIdentity) -> dict[str, object]:
        """Serialize the existing resume contract without changing audit math."""

        return {
            "method": METHOD_VERSION,
            "model": self.model_id,
            "model_dtype": self.model_dtype,
            "tokenizer": self.tokenizer_id,
            **corpus.manifest_fields(),
            "tasks": list(self.cohort.tasks),
            "samples_per_task": self.cohort.samples_per_task,
            "explicit_sample_ids": list(self.cohort.sample_ids),
            "selection_seed": self.cohort.seed,
            "targets_per_sample": self.targets.count,
            "target_policy": self.targets.policy,
            "max_response_tokens": self.targets.max_response_tokens,
            "flow_signal": self.mechanism.signal.value,
            "carrier_scope": self.mechanism.carrier_scope,
            "edge_coverage": self.mechanism.coverage,
            "query_chunk": self.mechanism.query_chunk,
            "route_budget": asdict(self.mechanism.route_budget),
            "local_window": self.mechanism.local_window,
            "saved_edges": self.mechanism.saved_edges,
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


def _target_selection_manifest(world: NativeWorld, target_rank: int) -> dict | None:
    if not world.target_selection:
        return None
    selection = world.target_selection[target_rank]
    return {**asdict(selection), "is_center": selection.is_center}


class SubsetAuditRunner:
    """Run ``select -> freeze targets -> audit -> persist`` for one corpus split."""

    def __init__(
        self,
        model,
        tokenizer,
        corpus: AuditCorpus,
        output_root: str | Path,
        config: SubsetRunConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.output = Path(output_root)
        self.config = config

    def run(self) -> dict[str, int]:
        self.corpus.identity.validate_runtime(
            self.config.model_id,
            self.config.tokenizer_id,
        )
        cohort = self.config.cohort
        selected = select_records(
            self.corpus.records,
            tasks=cohort.tasks,
            samples_per_task=cohort.samples_per_task,
            seed=cohort.seed,
            sample_ids=cohort.sample_ids,
        )
        manifest_path = self.output / MANIFEST_NAME
        manifest = open_manifest(
            manifest_path,
            self.config.manifest_value(self.corpus.identity),
            selected,
        )
        manifest["analysis_scope"] = (
            "structure_only"
            if self.config.scan_only
            else "structure_and_selected_target_function"
        )
        save_json(manifest_path, manifest)

        counts = {"samples": 0, "targets": 0, "resumed": 0, "confirmed": 0}
        samples = tqdm(
            selected,
            desc=f"{self.corpus.identity.split} samples",
            unit="sample",
            dynamic_ncols=True,
        )
        for record in samples:
            samples.set_postfix_str(
                f"{record.task_type}/{record.sample_id}",
                refresh=False,
            )
            world = self._load_or_build_world(record, manifest, samples)
            self._record_sample(record, world, manifest, manifest_path)
            if not self.config.scan_only:
                self._run_targets(
                    record,
                    world,
                    manifest,
                    manifest_path,
                    counts,
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

    def _paths(self, record: SampleRecord) -> tuple[Path, Path]:
        sample_key = safe_sample_key(record.sample_id)
        return (
            self.output / "worlds" / record.task_type / f"{sample_key}.npz",
            self.output / "scans" / record.task_type / f"{sample_key}.npz",
        )

    def _load_or_build_world(
        self,
        record: SampleRecord,
        manifest: dict,
        progress,
    ) -> NativeWorld:
        world_path, scan_path = self._paths(record)
        frozen_sample = manifest["samples"].get(record.sample_id)
        if world_path.is_file():
            world = load_native_world(world_path)
            if world.sample_id != safe_sample_key(record.sample_id):
                raise ValueError("saved native world has the wrong sample identity")
            if Path(world.tokenizer_id).name != self.config.tokenizer_id:
                raise ValueError("saved native world uses another tokenizer")
            if not scan_path.is_file():
                progress.set_postfix_str(
                    f"{record.task_type}/{record.sample_id} rebuild scan",
                    refresh=True,
                )
                sample = self.corpus.load_sample(record)
                scan = SampleScan.capture(
                    self.model,
                    world,
                    query_chunk=self.config.mechanism.query_chunk,
                    local_window=self.config.mechanism.local_window,
                )
                self._save_scan(scan, scan_path, sample, world)
            return world

        if frozen_sample is not None:
            raise ValueError(f"frozen native world is missing: {world_path}")
        sample = self.corpus.load_sample(record)
        world = self._build_world(sample, scan_path)
        save_native_world(world_path, world)
        return world

    def _build_world(self, sample: AuditSample, scan_path: Path) -> NativeWorld:
        full_response_tokens = sample.full_response_tokens
        prepared = sample.truncate_response(self.config.targets.max_response_tokens)
        targets, target_selection = freeze_target_plan(
            self.model,
            prepared.token_ids,
            prepared.response_start,
            count=self.config.targets.count,
            policy=self.config.targets.policy,
            query_chunk=self.config.mechanism.query_chunk,
            units=prepared.units,
            evidence_unit_id=prepared.evidence_unit_id,
            local_window=self.config.mechanism.local_window,
            on_scan=lambda scan: scan.save(
                scan_path,
                dataset_sample_id=prepared.record.sample_id,
                source_id=prepared.record.source_id,
                task_type=prepared.record.task_type,
                token_ids=prepared.token_ids,
                response_start=prepared.response_start,
                full_response_tokens=full_response_tokens,
                local_window=self.config.mechanism.local_window,
            ),
        )
        return NativeWorld(
            safe_sample_key(prepared.record.sample_id),
            self.config.tokenizer_id,
            prepared.token_ids,
            prepared.response_start,
            prepared.units,
            prepared.evidence_unit_id,
            targets,
            target_selection,
        ).check()

    def _save_scan(
        self,
        scan: SampleScan,
        scan_path: Path,
        sample: AuditSample,
        world: NativeWorld,
    ) -> None:
        scan.save(
            scan_path,
            dataset_sample_id=sample.record.sample_id,
            source_id=sample.record.source_id,
            task_type=sample.record.task_type,
            token_ids=world.token_ids,
            response_start=world.response_start,
            full_response_tokens=sample.full_response_tokens,
            local_window=self.config.mechanism.local_window,
        )

    def _record_sample(
        self,
        record: SampleRecord,
        world: NativeWorld,
        manifest: dict,
        manifest_path: Path,
    ) -> None:
        world_path, scan_path = self._paths(record)
        sample_entry = {
            "source_id": record.source_id,
            "task_type": record.task_type,
            "world": world_path.relative_to(self.output).as_posix(),
            "scan": scan_path.relative_to(self.output).as_posix(),
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
        frozen_sample = manifest["samples"].get(record.sample_id)
        if frozen_sample is not None:
            frozen_sample.setdefault("scan", sample_entry["scan"])
            if frozen_sample != sample_entry:
                raise ValueError(
                    f"saved native world disagrees with frozen sample {record.sample_id}"
                )
        manifest["samples"][record.sample_id] = sample_entry
        save_json(manifest_path, manifest)

    def _audit_identity(
        self,
        world: NativeWorld,
        record: SampleRecord,
        target: TargetContrast,
        target_rank: int,
        destination: Path,
    ) -> dict[str, object]:
        return {
            "result": destination.relative_to(self.output).as_posix(),
            "dataset_sample_id": record.sample_id,
            "sample_id": world.sample_id,
            "source_id": record.source_id,
            "task_type": record.task_type,
            "split": self.corpus.identity.split,
            "query_position": target.query_position,
            "positive_token_id": target.positive_token_id,
            "negative_token_id": target.negative_token_id,
            "contrast_origin": target.origin,
            "reanchor_selection": _target_selection_manifest(world, target_rank),
            "flow_signal": self.config.mechanism.signal.value,
            "target_rank": target_rank,
        }

    def _metadata(
        self,
        record: SampleRecord,
        target_rank: int,
    ) -> NativeAuditMetadata:
        mechanism = self.config.mechanism
        return NativeAuditMetadata(
            dataset_sample_id=record.sample_id,
            source_id=record.source_id,
            split=self.corpus.identity.split,
            task_type=record.task_type,
            generator_model=record.generator_model,
            model_id=self.config.model_id,
            model_dtype=self.config.model_dtype,
            target_policy=self.config.targets.policy,
            target_rank=target_rank,
            coverage=mechanism.coverage,
            carrier_scope=mechanism.carrier_scope,
            query_chunk=mechanism.query_chunk,
            route_budget=mechanism.route_budget,
            local_window=mechanism.local_window,
            saved_edges=mechanism.saved_edges,
        )

    def _run_targets(
        self,
        record: SampleRecord,
        world: NativeWorld,
        manifest: dict,
        manifest_path: Path,
        counts: dict[str, int],
    ) -> None:
        mechanism = self.config.mechanism
        if not world.targets:
            return
        sample_key = safe_sample_key(record.sample_id)
        targets = tqdm(
            enumerate(world.targets),
            total=len(world.targets),
            desc=(
                f"{self.corpus.identity.split}/{record.task_type}/"
                f"{record.sample_id} targets"
            ),
            unit="target",
            leave=False,
            dynamic_ncols=True,
        )
        for target_rank, target in targets:
            target_key = _target_key(target, mechanism.signal)
            key = f"{record.sample_id}:{target_key}"
            destination = (
                self.output
                / "audits"
                / record.task_type
                / sample_key
                / f"{target_key}.npz"
            )
            identity = self._audit_identity(
                world,
                record,
                target,
                target_rank,
                destination,
            )
            existing_entry = manifest["audits"].get(key)
            if existing_entry is not None and existing_entry != identity:
                raise ValueError(f"subset manifest audit {key} has another identity")
            metadata = self._metadata(record, target_rank)
            if destination.is_file():
                validate_native_audit(
                    destination,
                    world,
                    target,
                    mechanism.signal,
                    metadata,
                )
                counts["targets"] += 1
                counts["resumed"] += 1
                targets.set_postfix_str("resumed", refresh=False)
                with np.load(destination, allow_pickle=False) as stored:
                    counts["confirmed"] += int(stored["corridor_confirmed"])
            else:
                result = audit_native_target(
                    self.model,
                    world,
                    target,
                    mechanism.signal,
                    carrier_scope=mechanism.carrier_scope,
                    coverage=mechanism.coverage,
                    query_chunk=mechanism.query_chunk,
                    route_budget=mechanism.route_budget,
                    local_window=mechanism.local_window,
                    on_phase=lambda phase: targets.set_postfix_str(
                        phase,
                        refresh=True,
                    ),
                )
                save_native_audit(destination, world, result, metadata)
                validate_native_audit(
                    destination,
                    world,
                    target,
                    mechanism.signal,
                    metadata,
                )
                counts["targets"] += 1
                counts["confirmed"] += int(result.corridor_confirmed)
                targets.set_postfix_str("computed", refresh=False)
                prefix = (
                    f"{self.corpus.identity.split}/{record.task_type}/"
                    f"{record.sample_id} q={target.query_position} "
                    f"signal={mechanism.signal.value} "
                    f"root={result.selected_root_unit_id}"
                )
                if mechanism.route_budget.confirm:
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
