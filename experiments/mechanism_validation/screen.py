"""Streaming label-free mechanism artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from experiments.mechanism_validation.mechanisms import (
    compact_token_features,
    extract_token_mechanisms,
)


class MechanismScreen:
    """Write one compact mechanism tensor per response while streaming a dataset."""

    def __init__(self, dataset, output_dir, *, ema_decay: float = .9) -> None:
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.ema_decay = ema_decay

    def run(self) -> dict[str, int]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError("output_dir must be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        response_count = token_count = 0
        bound_invalid_rows = bound_total_rows = 0
        metadata = None
        index = []
        total = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        for research_sample in tqdm(self.dataset, total=total, desc="mechanism features", unit="sample"):
            attention = research_sample.attention()
            if metadata is not None and float(attention.attention_floor) != metadata["attention_floor"]:
                raise ValueError("attention floor differs across samples")
            raw = extract_token_mechanisms(attention)
            bound_invalid_rows += int((raw.values[..., 7] > 1).sum())
            bound_total_rows += raw.values[..., 7].numel()
            compact = compact_token_features(raw, ema_decay=self.ema_decay)
            torch.save({
                "sample_id": research_sample.sample_id,
                "source_id": research_sample.source_id,
                "prompt_length": research_sample.attention().response_idx,
                "task_type": research_sample.task_type,
                "data_source": research_sample.data_source,
                "values": compact.values.detach().cpu(),
                "valid": compact.valid.detach().cpu(),
            }, self.output_dir / f"{research_sample.sample_id}.pt")
            if metadata is None:
                metadata = {
                    "schema": "mechanism_features.v2",
                    "labels_included": False,
                    "ema_decay": self.ema_decay,
                    "attention_floor": float(attention.attention_floor),
                    "cache_bound_invalid_rows": 0, "cache_bound_total_rows": 0,
                    "feature_names": list(compact.names),
                    "family_slices": {
                        name: [selection.start, selection.stop]
                        for name, selection in compact.family_slices.items()
                    },
                }
            response_count += 1
            token_count += len(compact.values)
            index.append({"sample_id": research_sample.sample_id, "source_id": research_sample.source_id,
                          "tokens": len(compact.values)})
            research_sample.release_attention()
        if metadata is None:
            raise ValueError("dataset contains no attention samples")
        (self.output_dir / "metadata.json").write_text(
            json.dumps({**metadata, "cache_bound_invalid_rows": bound_invalid_rows,
                        "cache_bound_total_rows": bound_total_rows,
                        "cache_bound_invalid_fraction": bound_invalid_rows / bound_total_rows if bound_total_rows else 0.0}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.output_dir / "index.json").write_text(
            json.dumps({"samples": index, "tokens": token_count}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"responses": response_count, "tokens": token_count}
