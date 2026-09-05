"""Post-hoc label join for a frozen native mechanism subset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset

from .subset import MANIFEST_NAME
from .subset_artifacts import (
    MANIFEST_SCHEMA,
    capture_config_sha256,
    validate_compact_native_audit,
)
from .subset_data import file_sha256, safe_sample_key
from .worlds import TargetContrast

REPORT_NAME = "mechanism_evaluation.json"


def _mean(rows: list[dict], name: str) -> float | None:
    values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
    if not values:
        return None
    result = float(np.mean(values))
    return result if math.isfinite(result) else None


def _rate(rows: list[dict], name: str) -> float | None:
    return float(np.mean([bool(row[name]) for row in rows])) if rows else None


def summarize(rows: list[dict]) -> dict:
    return {
        "targets": len(rows),
        "samples": len({row["sample_id"] for row in rows}),
        "root_confirmed_rate": _rate(rows, "root_confirmed"),
        "corridor_confirmed_rate": _rate(rows, "corridor_confirmed"),
        "carrier_confirmed_rate": _rate(rows, "carrier_confirmed"),
        "restoration_valid_rate": _rate(rows, "restoration_valid"),
        "mean_root_value_effect": _mean(rows, "root_value_effect"),
        "mean_corridor_necessity": _mean(rows, "corridor_necessity"),
        "mean_corridor_rescue": _mean(rows, "corridor_rescue"),
        "mean_corridor_mediated_rescue": _mean(rows, "corridor_mediated_rescue"),
    }


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_dataset_identity(dataset_root: Path, manifest: dict) -> dict:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("capture manifest has no dataset configuration")
    captured_root = config.get("dataset_root")
    if not isinstance(captured_root, str) or not captured_root:
        raise ValueError("capture manifest has no dataset root")
    if dataset_root.resolve() != Path(captured_root).resolve():
        raise ValueError("evaluation dataset root differs from capture")
    expected_hash = config.get("dataset_manifest_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("capture manifest has no dataset-manifest hash")
    if file_sha256(dataset_root / "manifest.json") != expected_hash:
        raise ValueError("evaluation dataset manifest differs from capture")
    return config


def _target_key(target: TargetContrast, signal: str) -> str:
    return (
        f"q{target.query_position}_a{target.positive_token_id}"
        f"_b{target.negative_token_id}_{signal}"
    )


def _contained_file(root: Path, relative_value: object, *, kind: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"subset manifest {kind} path is invalid")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ValueError(f"subset manifest {kind} path must be relative")
    resolved_root = root.resolve()
    destination = (root / relative).resolve()
    try:
        destination.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"subset manifest {kind} path escapes output root") from error
    if not destination.is_file():
        raise ValueError(f"subset manifest {kind} file is missing: {relative_value}")
    return destination


def _target_from_manifest(value: object) -> TargetContrast:
    if not isinstance(value, dict):
        raise ValueError("subset manifest target is invalid")
    try:
        target = TargetContrast(
            int(value["query_position"]),
            int(value["positive_token_id"]),
            int(value["negative_token_id"]),
            str(value["contrast_origin"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("subset manifest target is invalid") from error
    if (
        target.query_position < 0
        or target.positive_token_id < 0
        or target.negative_token_id < 0
        or target.positive_token_id == target.negative_token_id
        or not target.origin
    ):
        raise ValueError("subset manifest target is invalid")
    return target


def _artifact_scalar(stored, name: str):
    value = stored[name]
    if value.shape != ():
        raise ValueError(f"subset artifact {name} must be scalar")
    return value.item()


def _preflight_audits(output: Path, manifest: dict, config: dict) -> list[dict]:
    """Validate and load every label-free result before labels are accessible."""

    selection = manifest.get("selection")
    samples = manifest.get("samples")
    audits = manifest.get("audits")
    if not isinstance(selection, list) or not isinstance(samples, dict):
        raise ValueError("subset manifest sample inventory is invalid")
    if not isinstance(audits, dict):
        raise ValueError("subset manifest audit inventory is invalid")
    if manifest.get("config_sha256") != capture_config_sha256(config):
        raise ValueError("subset manifest configuration fingerprint is invalid")

    selected: dict[str, dict] = {}
    for value in selection:
        if not isinstance(value, dict):
            raise ValueError("subset manifest selection is invalid")
        try:
            sample_id = str(value["sample_id"])
            source_id = str(value["source_id"])
            task_type = str(value["task_type"])
            generator_model = str(value["generator_model"])
        except KeyError as error:
            raise ValueError("subset manifest selection is invalid") from error
        if not sample_id or sample_id in selected or not source_id or not task_type:
            raise ValueError("subset manifest selection is invalid")
        selected[sample_id] = {
            "source_id": source_id,
            "task_type": task_type,
            "generator_model": generator_model,
        }
    if set(samples) != set(selected):
        raise ValueError("subset manifest samples differ from frozen selection")

    try:
        signal = str(config["flow_signal"])
        split = str(config["split"])
        tokenizer_id = str(config["tokenizer"])
        model_id = str(config["model"])
        model_dtype = str(config["model_dtype"])
    except KeyError as error:
        raise ValueError("capture manifest configuration is incomplete") from error

    expected_keys: set[str] = set()
    rows: list[dict] = []
    for sample_id, identity in selected.items():
        sample = samples[sample_id]
        if not isinstance(sample, dict):
            raise ValueError(f"subset manifest sample {sample_id} is invalid")
        for name in ("source_id", "task_type"):
            expected = identity[name]
            if sample.get(name) != expected:
                raise ValueError(
                    f"subset manifest sample {sample_id} has inconsistent {name}"
                )
        targets = sample.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"subset manifest sample {sample_id} has no targets")

        sample_key = safe_sample_key(sample_id)
        expected_world = (
            Path("worlds") / identity["task_type"] / f"{sample_key}.npz"
        ).as_posix()
        if sample.get("world") != expected_world:
            raise ValueError(
                f"subset manifest sample {sample_id} has inconsistent world"
            )
        world_path = _contained_file(output, sample["world"], kind="world")
        world_sha256 = sample.get("world_sha256")
        if not isinstance(world_sha256, str) or not world_sha256:
            raise ValueError(f"subset manifest sample {sample_id} has no world hash")
        if file_sha256(world_path) != world_sha256:
            raise ValueError(f"subset native-world hash mismatch for {sample_id}")
        for target_rank, target_value in enumerate(targets):
            target = _target_from_manifest(target_value)
            target_key = _target_key(target, signal)
            key = f"{sample_id}:{target_key}"
            if key in expected_keys:
                raise ValueError("subset manifest contains duplicate audit targets")
            expected_keys.add(key)
            entry = audits.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"subset manifest lacks expected audit {key}")
            relative = (
                Path("audits")
                / identity["task_type"]
                / sample_key
                / f"{target_key}.npz"
            ).as_posix()
            expected_entry = {
                "result": relative,
                "complete": True,
                "dataset_sample_id": sample_id,
                "sample_id": sample_key,
                "source_id": identity["source_id"],
                "task_type": identity["task_type"],
                "generator_model": identity["generator_model"],
                "split": split,
                "query_position": target.query_position,
                "positive_token_id": target.positive_token_id,
                "negative_token_id": target.negative_token_id,
                "contrast_origin": target.origin,
                "flow_signal": signal,
                "target_rank": target_rank,
                "world_sha256": world_sha256,
                "config_sha256": manifest["config_sha256"],
            }
            for name, expected in expected_entry.items():
                if entry.get(name) != expected:
                    raise ValueError(
                        f"subset manifest audit {key} has inconsistent {name}"
                    )
            destination = _contained_file(output, entry["result"], kind="audit")
            expected_sha256 = entry.get("sha256")
            if not isinstance(expected_sha256, str) or not expected_sha256:
                raise ValueError(f"subset manifest audit {key} has no artifact hash")
            if file_sha256(destination) != expected_sha256:
                raise ValueError(f"subset artifact hash mismatch for {key}")
            validate_compact_native_audit(
                destination,
                dataset_sample_id=sample_id,
                sample_id=sample_key,
                source_id=identity["source_id"],
                split=split,
                task_type=identity["task_type"],
                generator_model=identity["generator_model"],
                tokenizer_id=tokenizer_id,
                world_sha256=world_sha256,
                target=target,
                target_rank=target_rank,
                signal=signal,
                model_id=model_id,
                model_dtype=model_dtype,
                capture_config=config,
            )
            with np.load(destination, allow_pickle=False) as stored:
                row = {
                    "sample_id": sample_id,
                    "task_type": identity["task_type"],
                    "query_position": int(_artifact_scalar(stored, "query_position")),
                    "response_start": int(_artifact_scalar(stored, "response_start")),
                    "prediction_position": int(
                        _artifact_scalar(stored, "prediction_position")
                    ),
                    "root_confirmed": bool(
                        _artifact_scalar(stored, "selected_root_confirmed")
                    ),
                    "corridor_confirmed": bool(
                        _artifact_scalar(stored, "corridor_confirmed")
                    ),
                    "carrier_confirmed": bool(
                        _artifact_scalar(stored, "carrier_value_mediated")
                    ),
                    "restoration_valid": bool(
                        _artifact_scalar(stored, "corridor_restoration_valid")
                    ),
                    "root_value_effect": float(
                        _artifact_scalar(stored, "root_value_effect")
                    ),
                    "corridor_necessity": float(
                        _artifact_scalar(stored, "corridor_necessity")
                    ),
                    "corridor_rescue": float(
                        _artifact_scalar(stored, "corridor_conditional_rescue")
                    ),
                    "corridor_mediated_rescue": float(
                        _artifact_scalar(stored, "corridor_mediated_rescue")
                    ),
                }
            metric_names = (
                "root_value_effect",
                "corridor_necessity",
                "corridor_rescue",
                "corridor_mediated_rescue",
            )
            if not all(math.isfinite(row[name]) for name in metric_names):
                raise ValueError(f"subset artifact has non-finite metrics: {key}")
            if not 0 < row["response_start"] <= row["prediction_position"]:
                raise ValueError(
                    f"subset artifact has invalid response positions: {key}"
                )
            rows.append(row)
    if set(audits) != expected_keys:
        raise ValueError("subset manifest audit inventory is inconsistent")
    return rows


def evaluate_subset_split(
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict:
    """Join labels only after capture completion and summarize mechanisms."""

    dataset_root = Path(dataset_root)
    output = Path(output_root)
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_manifest_schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported subset manifest schema")
    if not manifest.get("analysis_complete"):
        raise ValueError("subset capture is incomplete")
    if manifest.get("labels_used_for_capture") is not False:
        raise ValueError("capture manifest violates the label firewall")
    config = _validate_dataset_identity(dataset_root, manifest)
    rows = _preflight_audits(output, manifest, config)

    sample_ids = list(manifest["samples"])
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=True,
    )
    labels = dataset.prepare_evaluation_labels(sample_ids)
    label_by_sample = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            label_by_sample[sample_id] = (
                labels.response_labels(sample).detach().cpu().numpy()
            )
        finally:
            sample.release_attention()

    for row in rows:
        response_start = row.pop("response_start")
        prediction = row.pop("prediction_position")
        relative = prediction - response_start
        sample_labels = label_by_sample[row["sample_id"]]
        if not 0 <= relative < len(sample_labels):
            raise ValueError(
                f"audit target lies outside labels: {row['sample_id']} "
                f"q={row['query_position']}"
            )
        row["hallucination_label"] = int(sample_labels[relative])

    groups = {"ALL": {}}
    for task in ("QA", "Summary", "Data2txt"):
        groups[task] = {}
    for task in groups:
        task_rows = (
            rows if task == "ALL" else [row for row in rows if row["task_type"] == task]
        )
        groups[task]["all"] = summarize(task_rows)
        groups[task]["clean"] = summarize(
            [row for row in task_rows if row["hallucination_label"] == 0]
        )
        groups[task]["hallucinated"] = summarize(
            [row for row in task_rows if row["hallucination_label"] == 1]
        )

    report = {
        "subset_evaluation_schema": 1,
        "capture_manifest": str(manifest_path.resolve()),
        "labels_accessed_after_capture": True,
        "selection_is_not_population_evaluation": True,
        "claim_scope": (
            "hallucinated-vs-clean observed-target dependence under a source "
            "Value-message cut; "
            "not factual correctness"
        ),
        "groups": groups,
        "targets": rows,
    }
    _save_json(output / REPORT_NAME, report)
    return report
