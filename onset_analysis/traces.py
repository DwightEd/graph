"""Read existing attention-audit v3 observations and evaluation labels."""

import json
from pathlib import Path

import numpy as np

REQUIRED_GROUPS = (
    "special",
    "evidence",
    "other_prompt",
    "history_far",
    "history_local",
    "self",
)


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
            raise ValueError(
                "attention-audit manifest contains an invalid sample entry"
            )
        if (
            any(
                not isinstance(entry[name], str) or not entry[name]
                for name in required - {"response_tokens", "response_start"}
            )
            or type(entry["response_tokens"]) is not int
            or type(entry["response_start"]) is not int
            or entry["response_tokens"] < 1
            or entry["response_start"] < 1
        ):
            raise ValueError(
                "attention-audit sample identity and dimensions are invalid"
            )
    return entries


def load_audit_manifest(root: Path) -> list[dict]:
    manifest_path = root / "index.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"attention-audit index does not exist: {manifest_path}"
        )
    return _validated_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))


def trace_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"attention trace escapes the audit root: {relative}")
    return path


def load_trace(path: Path, entry: dict) -> dict:
    fields = {
        "sample_id",
        "source_id",
        "task_type",
        "response_start",
        "token_ids",
        "token_text",
        "special_mask",
        "group_names",
        "mass",
        "ordinary_mass",
        "entropy_normalized",
        "top1",
        "observed_margin",
        "predictor_entropy",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = fields.difference(archive.files)
        if missing:
            raise ValueError(
                f"{path} is missing onset-choice arrays: {sorted(missing)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in fields}
    for name in ("sample_id", "source_id", "task_type"):
        if str(arrays[name]) != entry[name]:
            raise ValueError(f"{path} {name} does not match index.json")
    start, count = int(arrays["response_start"]), entry["response_tokens"]
    if start != entry["response_start"]:
        raise ValueError(f"{path} response_start does not match index.json")
    groups = tuple(str(value) for value in arrays["group_names"])
    mass = arrays["mass"]
    head_shape = mass.shape[:3]
    if (
        set(groups) != set(REQUIRED_GROUPS)
        or mass.ndim != 4
        or mass.shape[2] != count + 1
        or mass.shape[-1] != len(groups)
        or arrays["ordinary_mass"].shape != head_shape
        or arrays["entropy_normalized"].shape != head_shape
        or arrays["top1"].shape != head_shape
        or arrays["observed_margin"].shape != (count + 1,)
        or arrays["predictor_entropy"].shape != (count + 1,)
        or arrays["token_ids"].shape != (start + count,)
        or arrays["token_text"].shape != (start + count,)
        or arrays["special_mask"].shape != (start + count,)
    ):
        raise ValueError(f"{path} has inconsistent onset-choice array shapes")
    share = np.full(mass.shape, np.nan, dtype=np.float64)
    np.divide(
        mass,
        arrays["ordinary_mass"][..., None],
        out=share,
        where=arrays["ordinary_mass"][..., None] > 0,
    )
    return {
        "token_ids": arrays["token_ids"][start:],
        "token_text": arrays["token_text"][start:],
        "special_targets": arrays["special_mask"][start:],
        "group_index": {name: index for index, name in enumerate(groups)},
        "share": share,
        "attention_entropy": arrays["entropy_normalized"],
        "top1": arrays["top1"],
        "observed_margin": arrays["observed_margin"],
        "predictor_entropy": arrays["predictor_entropy"],
    }


def load_labels(path: Path, response_tokens: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive.files:
            raise ValueError(f"label sidecar has no labels array: {path}")
        labels = np.asarray(archive["labels"])
    if labels.shape != (response_tokens,) or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError(f"labels must cover the response and use -1/0/1: {path}")
    return labels.astype(np.int8, copy=False)
