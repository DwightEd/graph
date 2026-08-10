"""One-time conversion of existing PT artifacts to the canonical feature layout."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from cache import AttentionDataset, AttentionSample, save_attention_sample
from features import save_hidden_features, save_token_stats, teacher_forced_stats


@dataclass(frozen=True)
class ArchiveConfig:
    formal_root: str | Path
    output_root: str | Path


@dataclass(frozen=True)
class TraceArchiveConfig:
    trace_dir: str | Path
    output_dir: str | Path


def _positive_runs(y_token: torch.Tensor, response_idx: int) -> list[list[int]]:
    positions = torch.nonzero(y_token[response_idx:] > 0, as_tuple=False).flatten().tolist()
    runs: list[list[int]] = []
    for position in positions:
        if not runs or position != runs[-1][1]:
            runs.append([position, position + 1])
        else:
            runs[-1][1] += 1
    return runs


def _formal_sample(raw: dict[str, Any]) -> AttentionSample:
    return AttentionSample(
        sample_id=str(raw["response_id"]),
        source_id=str(raw["source_id"]),
        response_idx=int(raw["response_idx"]),
        token_ids=torch.as_tensor(raw["token_ids"]),
        attention_diagonal=torch.as_tensor(raw["attention_diagonal"]),
        response_row_ptr=torch.as_tensor(raw["response_row_ptr"]),
        response_column_indices=torch.as_tensor(raw["response_column_indices"]),
        response_values=torch.as_tensor(raw["response_values"]),
        attention_floor=float(raw["attention_floor"]),
    )


def convert_formal_split(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Convert one formal RAGTruth attention split to the six-field archive."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    paths = sorted(source_dir.glob("attention_*.pt"))
    if not paths:
        paths = sorted(path for path in source_dir.glob("*.pt") if path.name != "manifest.pt")
    if not paths:
        raise ValueError(f"no attention .pt files found in {source_dir}")

    (output_dir / "attention").mkdir(parents=True, exist_ok=True)
    index_rows, label_rows = [], []
    floor = layers = heads = None

    for path in tqdm(paths, desc=f"convert {source_dir.name}"):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sample = _formal_sample(raw)
        sample.validate()
        if floor is None:
            floor = sample.attention_floor
            layers, heads = sample.num_layers, sample.num_heads
        elif (sample.attention_floor, sample.num_layers, sample.num_heads) != (floor, layers, heads):
            raise ValueError("attention geometry changed within one split")

        relative = Path("attention") / f"{sample.sample_id}.npz"
        save_attention_sample(sample, output_dir / relative)
        index_rows.append({
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "path": relative.as_posix(),
        })
        if "y_token" in raw:
            label_rows.append({
                "sample_id": sample.sample_id,
                "positive_runs": _positive_runs(torch.as_tensor(raw["y_token"]), sample.response_idx),
            })

    manifest = {
        "attention_floor": floor,
        "num_layers": layers,
        "num_heads": heads,
        "count": len(index_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
    )
    if label_rows:
        (output_dir / "labels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8"
        )
    return manifest


class AttentionArchiveConverter:
    def __init__(self, config: ArchiveConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        formal_root, output_root = Path(self.config.formal_root), Path(self.config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summaries = {
            split: convert_formal_split(formal_root / split, output_root / split)
            for split in ("train", "test")
        }
        return {
            "count": sum(item["count"] for item in summaries.values()),
            "splits": {split: item["count"] for split, item in summaries.items()},
            "output_root": str(output_root),
        }


def _first(raw: dict[str, Any], *names):
    for name in names:
        if name in raw:
            return raw[name]
    return None


def convert_trace_dir(trace_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Convert legacy hidden/logit traces without copying engineering metadata."""
    trace_dir, output_dir = Path(trace_dir), Path(output_dir)
    paths = sorted(trace_dir.glob("*.pt"))
    if not paths:
        paths = sorted(trace_dir.rglob("*.pt"))
    if not paths:
        raise ValueError(f"no trace .pt files found in {trace_dir}")

    rows = []
    for path in tqdm(paths, desc="convert feature traces"):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sample_id = str(_first(raw, "example_id", "response_id", "sample_id") or path.stem)
        token_ids = _first(raw, "input_ids", "token_ids")
        if token_ids is None:
            continue
        token_ids = torch.as_tensor(token_ids).flatten()

        has_hidden = False
        hidden = _first(raw, "hidden_states")
        if hidden is not None:
            hidden = torch.as_tensor(hidden)
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            layer_ids = _first(raw, "selected_hidden_layers", "hidden_layer_ids")
            if layer_ids is None:
                layer_ids = torch.arange(hidden.shape[0])
            save_hidden_features(
                output_dir / "hidden" / f"{sample_id}.npz",
                token_ids,
                torch.as_tensor(layer_ids).flatten().tolist(),
                hidden,
            )
            has_hidden = True

        has_stats = False
        log_prob = _first(raw, "token_log_prob")
        entropy = _first(raw, "next_token_entropy", "entropy")
        if log_prob is None or entropy is None:
            logits = _first(raw, "logits")
            if logits is not None:
                log_prob, entropy = teacher_forced_stats(logits, token_ids)
        if log_prob is not None and entropy is not None:
            save_token_stats(
                output_dir / "token_stats" / f"{sample_id}.npz",
                token_ids,
                log_prob,
                entropy,
            )
            has_stats = True

        if has_hidden or has_stats:
            rows.append({"sample_id": sample_id, "hidden": has_hidden, "token_stats": has_stats})

    (output_dir / "feature_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return {"count": len(rows), "output_dir": str(output_dir)}


class TraceArchiveConverter:
    def __init__(self, config: TraceArchiveConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        return convert_trace_dir(self.config.trace_dir, self.config.output_dir)


class AttentionArchiveStore(AttentionDataset):
    """Backward-compatible name for archive-root + split loading."""

    def __init__(self, root: str | Path, split: str, device: str | torch.device = "cpu") -> None:
        super().__init__(Path(root) / split, device=device)


class AttentionArchiveVerifier:
    """Lightweight structural check: load every canonical attention sample once."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run(self) -> dict[str, Any]:
        counts = {}
        for split in ("train", "test"):
            dataset = AttentionDataset(self.root / split)
            for _ in dataset:
                pass
            counts[split] = len(dataset)
        return {"count": sum(counts.values()), "splits": counts}


class ArtifactInspector:
    """Show actual fields and tensor shapes of one formal sample per split."""

    def __init__(self, formal_root: str | Path) -> None:
        self.formal_root = Path(formal_root)

    def run(self) -> dict[str, Any]:
        report = {"root": str(self.formal_root), "splits": {}}
        for split in ("train", "test"):
            paths = sorted((self.formal_root / split).glob("attention_*.pt"))
            if not paths:
                paths = sorted((self.formal_root / split).glob("*.pt"))
            raw = torch.load(paths[0], map_location="cpu", weights_only=True)
            report["splits"][split] = {
                "count": len(paths),
                "fields": list(raw),
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in raw.items()
                    if isinstance(value, torch.Tensor)
                },
            }
        return report
