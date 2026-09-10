"""Label-free scoring of completed native attention-audit traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from tqdm.auto import tqdm

AUDIT_SCORE_SCHEMA = "control-graph/audit-token-score@1"
REQUIRED_GROUPS = (
    "special",
    "evidence",
    "other_prompt",
    "history_far",
    "history_local",
    "self",
)


@dataclass(frozen=True)
class AuditScoreConfig:
    audit_root: Path
    output_dir: Path
    completed_only: bool = False


class AttentionAuditScorer:
    """Turn each ordinary response token into a four-edge support proxy."""

    def __init__(self, config: AuditScoreConfig, *, progress: bool = True) -> None:
        self.config = config
        self.progress = progress

    def run(self) -> dict:
        root = self.config.audit_root.resolve()
        entries = load_audit_manifest(root)
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"audit-score output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)

        completed = skipped = available_tokens = special_tokens = unscorable_tokens = 0
        scored_tokens = 0
        sources: set[str] = set()
        score_path = output / "scores.jsonl"
        iterator = tqdm(entries, desc="score attention audit", unit="sample", disable=not self.progress)
        with score_path.open("w", encoding="utf-8") as stream:
            for entry in iterator:
                trace_path = _trace_path(root, entry["path"])
                if not trace_path.is_file():
                    if not self.config.completed_only:
                        raise FileNotFoundError(
                            f"attention trace is incomplete: {trace_path}; "
                            "pass --completed-only to evaluate available samples"
                        )
                    skipped += 1
                    continue
                counts = self._score_trace(trace_path, entry, stream)
                completed += 1
                available_tokens += counts["available"]
                special_tokens += counts["special"]
                unscorable_tokens += counts["unscorable"]
                scored_tokens += counts["scored"]
                sources.add(entry["source_id"])
                iterator.set_postfix(completed=completed, tokens=scored_tokens)

        summary = {
            "schema": "control-graph/audit-score-summary@1",
            "planned_samples": len(entries),
            "completed_samples": completed,
            "skipped_samples": skipped,
            "available_response_tokens": available_tokens,
            "special_target_tokens": special_tokens,
            "unscorable_tokens": unscorable_tokens,
            "scored_tokens": scored_tokens,
            "sources": len(sources),
            "labels_used": False,
            "output": str(score_path),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def _score_trace(self, path: Path, entry: dict, stream: TextIO) -> dict[str, int]:
        with np.load(path, allow_pickle=False) as trace:
            arrays = _validated_trace(trace, entry, path)
        response_tokens = entry["response_tokens"]
        group_index = {name: index for index, name in enumerate(arrays["group_names"])}
        message_mass = arrays["message_mass"][..., :response_tokens, :]
        message_total = arrays["message_ordinary_mass"][..., :response_tokens]
        attention_mass = arrays["mass"][..., :response_tokens, :]
        attention_total = arrays["ordinary_mass"][..., :response_tokens]
        head_margin = arrays["head_margin"][..., :response_tokens]
        observed_margin = arrays["observed_margin"][:response_tokens]
        special_targets = arrays["special_mask"][entry["response_start"] :]
        token_ids = arrays["token_ids"][entry["response_start"] :]

        message_share = _shares(message_mass, message_total)
        attention_share = _shares(attention_mass, attention_total)
        counts = {"available": response_tokens, "special": 0, "unscorable": 0, "scored": 0}
        for response_index in range(response_tokens):
            if special_targets[response_index]:
                counts["special"] += 1
                continue
            edges = _support_edges(message_share, head_margin, response_index, group_index)
            if edges is None or not np.isfinite(observed_margin[response_index]):
                counts["unscorable"] += 1
                continue
            attention_displacement = _attention_displacement(
                attention_share, response_index, group_index
            )
            if not np.isfinite(attention_displacement):
                counts["unscorable"] += 1
                continue
            scores = {
                "constraint_displacement": edges["far_history_support"]
                + edges["local_history_support"]
                - edges["evidence_support"],
                "attention_displacement": attention_displacement,
                "negative_margin": -float(observed_margin[response_index]),
                "relative_position": response_index / max(response_tokens - 1, 1),
            }
            record = {
                "schema": AUDIT_SCORE_SCHEMA,
                "event_id": f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}:{response_index}",
                "sample_id": entry["sample_id"],
                "source_id": entry["source_id"],
                "split": entry["split"],
                "task_type": entry["task_type"],
                "response_index": response_index,
                "token_id": int(token_ids[response_index]),
                "mechanism_edges": edges,
                "scores": scores,
            }
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            counts["scored"] += 1
        return counts


def _validated_manifest(manifest: dict) -> list[dict]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("audit_schema") != 3
        or manifest.get("labels_used_for_capture") is not False
        or not isinstance(manifest.get("samples"), list)
    ):
        raise ValueError("index.json is not a label-free attention-audit v3 manifest")
    required = {
        "split",
        "task_type",
        "sample_id",
        "source_id",
        "path",
        "response_tokens",
        "response_start",
    }
    entries = manifest["samples"]
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError("attention-audit manifest contains an invalid sample entry")
        if (
            any(not isinstance(entry[name], str) or not entry[name] for name in required - {"response_tokens", "response_start"})
            or type(entry["response_tokens"]) is not int
            or type(entry["response_start"]) is not int
            or entry["response_tokens"] < 1
            or entry["response_start"] < 1
        ):
            raise ValueError("attention-audit sample identity and dimensions are invalid")
    return entries


def load_audit_manifest(root: Path) -> list[dict]:
    manifest_path = root / "index.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"attention-audit index does not exist: {manifest_path}")
    return _validated_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))


def _trace_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"attention trace escapes the audit root: {relative}")
    return path


def _validated_trace(trace, entry: dict, path: Path) -> dict[str, np.ndarray]:
    required = {
        "audit_schema",
        "labels_used_for_capture",
        "sample_id",
        "source_id",
        "task_type",
        "token_ids",
        "response_start",
        "special_mask",
        "group_names",
        "message_mass",
        "message_ordinary_mass",
        "mass",
        "ordinary_mass",
        "head_margin",
        "observed_margin",
    }
    missing = required.difference(trace.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    arrays = {name: np.asarray(trace[name]) for name in required}
    if int(arrays["audit_schema"]) != 3 or bool(arrays["labels_used_for_capture"]):
        raise ValueError(f"{path} is not a label-free attention-audit v3 trace")
    for name in ("sample_id", "source_id", "task_type"):
        if str(arrays[name]) != entry[name]:
            raise ValueError(f"{path} {name} does not match index.json")
    if int(arrays["response_start"]) != entry["response_start"]:
        raise ValueError(f"{path} response_start does not match index.json")
    groups = tuple(str(value) for value in arrays["group_names"])
    if set(groups) != set(REQUIRED_GROUPS) or len(groups) != len(REQUIRED_GROUPS):
        raise ValueError(f"{path} has an unsupported attention group schema")
    response_tokens = entry["response_tokens"]
    query_count = response_tokens + 1
    message_mass = arrays["message_mass"]
    head_margin = arrays["head_margin"]
    expected_head_shape = message_mass.shape[:3]
    if (
        message_mass.ndim != 4
        or message_mass.shape[-1] != len(groups)
        or message_mass.shape[2] != query_count
        or head_margin.shape != expected_head_shape
        or arrays["message_ordinary_mass"].shape != expected_head_shape
        or arrays["mass"].shape != message_mass.shape
        or arrays["ordinary_mass"].shape != expected_head_shape
        or arrays["observed_margin"].shape != (query_count,)
    ):
        raise ValueError(f"{path} has inconsistent attention-audit array shapes")
    token_count = entry["response_start"] + response_tokens
    if arrays["token_ids"].shape != (token_count,) or arrays["special_mask"].shape != (token_count,):
        raise ValueError(f"{path} token arrays do not match the response coordinates")
    arrays["group_names"] = np.asarray(groups)
    return arrays


def _shares(mass: np.ndarray, total: np.ndarray) -> np.ndarray:
    result = np.full(mass.shape, np.nan, dtype=np.float64)
    np.divide(mass, total[..., None], out=result, where=total[..., None] > 0)
    return result


def _support_edges(
    shares: np.ndarray,
    head_margin: np.ndarray,
    response_index: int,
    groups: dict[str, int],
) -> dict[str, float] | None:
    margin = head_margin[..., response_index].astype(np.float64)
    valid = np.isfinite(margin) & np.isfinite(shares[..., response_index, :]).all(axis=-1)
    scale = np.abs(margin[valid]).sum()
    if not valid.any() or scale <= 0:
        return None

    def allocated(*names: str) -> float:
        share = sum(shares[..., response_index, groups[name]] for name in names)
        return float((share[valid] * margin[valid]).sum() / scale)

    return {
        "evidence_support": allocated("evidence"),
        "other_prompt_support": allocated("other_prompt"),
        "far_history_support": allocated("history_far"),
        "local_history_support": allocated("history_local", "self"),
    }


def _attention_displacement(
    shares: np.ndarray, response_index: int, groups: dict[str, int]
) -> float:
    evidence = shares[..., response_index, groups["evidence"]]
    history = sum(
        shares[..., response_index, groups[name]]
        for name in ("history_far", "history_local", "self")
    )
    difference = history - evidence
    return float(np.nanmean(difference)) if np.isfinite(difference).any() else float("nan")
