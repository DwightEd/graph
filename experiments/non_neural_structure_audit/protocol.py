"""Frozen source splits and artifact bindings for label-aware evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from experiment_protocol import (
    FrozenFile,
    canonical_source_group,
    dataset_manifest_sha256,
    file_sha256,
    validate_source_audit,
)

from .artifacts import read_json, write_json
from .config import EvaluationConfig
from .features import RELATION_NAMES

SPLIT_SCHEMA = "non-neural-structure-split-plan-v1"
CONFIRMATION_SCHEMA = "non-neural-structure-confirmation-plan-v1"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
)


def _method_files() -> list[Path]:
    package = Path(__file__).parent
    repository = package.parents[1]
    files = list(package.glob("*.py"))
    files.extend((repository / "experiments" / "attention_phenomenology").glob("*.py"))
    files.extend(
        repository / "experiments" / name
        for name in ("causal_attention_edges.py", "disk_row_store.py")
    )
    files.extend(
        repository / name
        for name in (
            "attention_lifecycle.py",
            "cache.py",
            "experiment_protocol.py",
            "formal_cache.py",
            "research_dataset.py",
        )
    )
    return sorted(path.resolve() for path in files)


def method_sha256() -> str:
    repository = Path(__file__).parent.parents[1]
    digest = hashlib.sha256()
    digest.update(f"transformers={version('transformers')}".encode())
    for path in _method_files():
        digest.update(path.relative_to(repository).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tokenizer_sha256(path) -> str:
    root = Path(path).resolve()
    files = [root / name for name in TOKENIZER_FILES if (root / name).is_file()]
    if not files:
        raise ValueError("tokenizer directory has no tokenizer files")
    digest = hashlib.sha256()
    digest.update(f"transformers={version('transformers')}".encode())
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def prepare_split_plan(
    *, score_dir, output, discovery_fraction: float = 0.5, seed: int = 20260824
) -> dict:
    score_dir = Path(score_dir).resolve()
    manifest_path = score_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest["schema"] != "non-neural-structure-manifest-v2"
        or manifest["labels_read"] is not False
    ):
        raise ValueError("split plan requires label-free structure scores")
    if manifest["method_sha256"] != method_sha256():
        raise ValueError("scoring code differs from the current audit method")
    if not 0.0 < float(discovery_fraction) < 1.0:
        raise ValueError("discovery_fraction must be in (0, 1)")
    groups: dict[str, list[str]] = {}
    for row in manifest["samples"]:
        groups.setdefault(str(row["source_id"]), []).append(str(row["sample_id"]))
    if len(groups) < 2:
        raise ValueError("split plan requires at least two source groups")
    ordered = sorted(
        groups,
        key=lambda source: hashlib.sha256(
            f"structure-audit-split-v1\0{seed}\0{source}".encode()
        ).digest(),
    )
    discovery_count = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * float(discovery_fraction))),
    )
    discovery_sources = set(ordered[:discovery_count])
    confirmation_sources = set(ordered[discovery_count:])
    plan = {
        "schema": SPLIT_SCHEMA,
        "labels_read": False,
        "score_manifest_path": str(manifest_path),
        "score_manifest_sha256": file_sha256(manifest_path),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "reference_sha256": manifest["reference_sha256"],
        "score_method_sha256": manifest["method_sha256"],
        "seed": int(seed),
        "discovery_fraction": float(discovery_fraction),
        "discovery_source_ids": sorted(discovery_sources),
        "confirmation_source_ids": sorted(confirmation_sources),
        "discovery_sample_ids": [
            str(row["sample_id"])
            for row in manifest["samples"]
            if str(row["source_id"]) in discovery_sources
        ],
        "confirmation_sample_ids": [
            str(row["sample_id"])
            for row in manifest["samples"]
            if str(row["source_id"]) in confirmation_sources
        ],
    }
    write_json(output, plan)
    return plan


def load_split_plan(path, *, score_dir) -> dict:
    plan_path = Path(path).resolve()
    plan = read_json(plan_path)
    if plan["schema"] != SPLIT_SCHEMA or plan["labels_read"] is not False:
        raise ValueError("invalid label-free split plan")
    manifest_path = Path(score_dir).resolve() / "manifest.json"
    FrozenFile(Path(plan["score_manifest_path"]), plan["score_manifest_sha256"]).verify(
        manifest_path
    )
    discovery = set(plan["discovery_source_ids"])
    confirmation = set(plan["confirmation_source_ids"])
    if discovery & confirmation:
        raise ValueError("discovery and confirmation source groups overlap")
    manifest = read_json(manifest_path)
    current_method = method_sha256()
    if (
        plan["score_method_sha256"] != manifest["method_sha256"]
        or manifest["method_sha256"] != current_method
    ):
        raise ValueError("split plan, scores, and current audit method differ")
    rows = manifest["samples"]
    all_sources = {str(row["source_id"]) for row in rows}
    discovery_samples = list(map(str, plan["discovery_sample_ids"]))
    confirmation_samples = list(map(str, plan["confirmation_sample_ids"]))
    expected_discovery = {
        str(row["sample_id"]) for row in rows if str(row["source_id"]) in discovery
    }
    expected_confirmation = {
        str(row["sample_id"]) for row in rows if str(row["source_id"]) in confirmation
    }
    if discovery | confirmation != all_sources:
        raise ValueError("split source groups do not cover the frozen scores")
    if (
        len(discovery_samples) != len(set(discovery_samples))
        or len(confirmation_samples) != len(set(confirmation_samples))
        or set(discovery_samples) != expected_discovery
        or set(confirmation_samples) != expected_confirmation
    ):
        raise ValueError("split sample groups do not match their frozen sources")
    return plan


def freeze_confirmation(
    *,
    split_plan,
    discovery_evaluation,
    output,
    tokenizer_path,
    config: EvaluationConfig,
) -> dict:
    if config.scope != "confirmation":
        raise ValueError("confirmation freeze requires confirmation scope")
    split_path = Path(split_plan).resolve()
    split = load_split_plan(
        split_path, score_dir=Path(read_json(split_path)["score_manifest_path"]).parent
    )
    discovery_path = Path(discovery_evaluation).resolve()
    discovery = read_json(discovery_path)
    if discovery["scope"] != "discovery":
        raise ValueError("confirmation must be frozen from a discovery evaluation")
    if set(discovery["selected_sample_ids"]) != set(split["discovery_sample_ids"]):
        raise ValueError("discovery evaluation does not match the frozen split")
    if discovery["score_manifest_sha256"] != split["score_manifest_sha256"]:
        raise ValueError("discovery evaluation uses different frozen scores")
    if discovery["method_sha256"] != method_sha256():
        raise ValueError("discovery evaluation uses different audit code")
    a0 = next(
        (row for row in discovery["decisions"] if row["audit"] == "A0"),
        None,
    )
    if a0 is None or a0["status"] != "PASS":
        raise ValueError("discovery A0 is incomplete; confirmation cannot be frozen")
    tokenizer_path = Path(tokenizer_path).resolve()
    tokenizer_digest = tokenizer_sha256(tokenizer_path)
    if discovery["tokenizer_sha256"] != tokenizer_digest:
        raise ValueError("confirmation tokenizer differs from discovery")
    plan = {
        "schema": CONFIRMATION_SCHEMA,
        "labels_read": False,
        "split_plan_path": str(split_path),
        "split_plan_sha256": file_sha256(split_path),
        "discovery_evaluation_path": str(discovery_path),
        "discovery_evaluation_sha256": file_sha256(discovery_path),
        "score_manifest_sha256": split["score_manifest_sha256"],
        "method_sha256": method_sha256(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_digest,
        "decision_rule": "non-neural-structure-gates-v1",
        "evaluation_config": asdict(config),
        "confirmation_source_ids": split["confirmation_source_ids"],
        "confirmation_sample_ids": split["confirmation_sample_ids"],
    }
    write_json(output, plan)
    return plan


def load_confirmation_plan(
    path, *, score_dir, tokenizer_path, config: EvaluationConfig
) -> dict:
    plan = read_json(path)
    if plan["schema"] != CONFIRMATION_SCHEMA or plan["labels_read"] is not False:
        raise ValueError("invalid confirmation plan")
    split_path = Path(plan["split_plan_path"])
    FrozenFile(split_path, plan["split_plan_sha256"]).verify(split_path)
    split = load_split_plan(split_path, score_dir=score_dir)
    discovery_path = Path(plan["discovery_evaluation_path"])
    FrozenFile(discovery_path, plan["discovery_evaluation_sha256"]).verify(
        discovery_path
    )
    if plan["method_sha256"] != method_sha256():
        raise ValueError("audit code differs from the frozen confirmation plan")
    if plan["evaluation_config"] != asdict(config):
        raise ValueError("evaluation config differs from the frozen confirmation plan")
    tokenizer_path = Path(tokenizer_path).resolve()
    if tokenizer_path != Path(plan["tokenizer_path"]):
        raise ValueError("tokenizer path differs from the confirmation plan")
    if tokenizer_sha256(tokenizer_path) != plan["tokenizer_sha256"]:
        raise ValueError("tokenizer files differ from the confirmation plan")
    if plan["score_manifest_sha256"] != split["score_manifest_sha256"]:
        raise ValueError("confirmation plan uses different frozen scores")
    return plan


def validate_score_binding(
    *, manifest: dict, score_dir, dataset, selected_sample_ids
) -> None:
    if manifest["schema"] != "non-neural-structure-manifest-v2":
        raise ValueError("unsupported structure score manifest")
    if manifest["method_sha256"] != method_sha256():
        raise ValueError("structure scores use different audit code")
    if tuple(manifest["relation_names"]) != RELATION_NAMES:
        raise ValueError("structure score relations differ from the current method")
    if manifest["trace_alignment"] != "post_token_query_at_same_position":
        raise ValueError("unexpected cached trace alignment")
    if manifest["evaluation_alignment"] != "query_t_to_response_token_t_plus_1":
        raise ValueError("unexpected score/label alignment")
    if manifest["dataset_manifest_sha256"] != dataset_manifest_sha256(dataset):
        raise ValueError("evaluation dataset differs from frozen scores")
    FrozenFile(Path(manifest["reference_path"]), manifest["reference_sha256"]).verify(
        manifest["reference_path"]
    )
    rows = manifest["samples"]
    validate_source_audit(
        reserved_source_ids=manifest["reference_source_ids"],
        test_source_ids=manifest["test_source_ids"],
        test_sample_ids=manifest["test_sample_ids"],
        row_sample_ids=[row["sample_id"] for row in rows],
        row_source_ids=[row["source_id"] for row in rows],
        audit_scope=manifest["audit_scope"],
    )
    selected = set(selected_sample_ids)
    if not selected or not selected.issubset(set(manifest["test_sample_ids"])):
        raise ValueError("evaluation selection is outside frozen scores")
    if manifest["audit_scope"] == "complete_split" and set(
        manifest["test_sample_ids"]
    ) != set(map(str, dataset.sample_ids)):
        raise ValueError("complete score scope does not cover the dataset")
    for row in rows:
        sample_id = str(row["sample_id"])
        sample = dataset[sample_id]
        try:
            if canonical_source_group(sample) != str(row["source_id"]):
                raise ValueError("score manifest source differs from the dataset")
        finally:
            sample.release_attention()
        if sample_id not in selected:
            continue
        score_path = Path(score_dir) / row["score_path"]
        if file_sha256(score_path) != row["score_sha256"]:
            raise ValueError("frozen sample score digest differs from its manifest")
        audit = row["null_audit"]
        if any(
            float(audit[name]) != 0.0
            for name in (
                "row_mass_max_error",
                "role_mass_max_error",
                "source_count_degree_max_error",
                "stratified_source_count_max_error",
                "causal_violations",
                "coarse_lag_violations",
                "duplicate_edges",
            )
        ):
            raise ValueError("response-endpoint null violates a required invariant")
