"""Prompt-anchor definitions used by the evidence-lineage proxy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class Anchor:
    name: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class AnchorMap:
    token_anchor: torch.Tensor  # [prompt tokens], values in [0, max_anchors)
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    mode: str

    @property
    def count(self) -> int:
        return len(self.names)

    def permuted(self, generator: torch.Generator) -> "AnchorMap":
        if self.token_anchor.numel() < 2:
            return self
        order = torch.randperm(
            self.token_anchor.numel(),
            generator=generator,
            device=self.token_anchor.device,
        )
        return AnchorMap(
            token_anchor=self.token_anchor[order],
            names=self.names,
            kinds=self.kinds,
            mode=f"{self.mode}:token_assignment_permuted",
        )


def load_anchor_manifest(path: str | Path | None) -> dict[str, list[Anchor]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, list[Anchor]] = {}
    for sample_id, rows in payload.items():
        result[str(sample_id)] = [
            Anchor(
                name=str(row.get("name", f"anchor_{index}")),
                kind=str(row.get("kind", "evidence")),
                start=int(row["start"]),
                end=int(row["end"]),
            )
            for index, row in enumerate(rows)
        ]
    return result


def _from_segments(
    response_idx: int,
    segments: list[Anchor],
    *,
    max_anchors: int,
    device: torch.device,
) -> AnchorMap:
    selected = [
        Anchor(item.name, item.kind, max(0, item.start), min(response_idx, item.end))
        for item in segments
        if item.end > item.start and item.start < response_idx and item.end > 0
    ]
    selected = selected[:max_anchors]
    token_anchor = torch.full((response_idx,), -1, dtype=torch.long, device=device)
    names, kinds = [], []
    for index, item in enumerate(selected):
        token_anchor[item.start : item.end] = index
        names.append(item.name)
        kinds.append(item.kind)

    if bool((token_anchor < 0).any()):
        if len(names) == max_anchors:
            # Reserve the final state for uncovered prompt content instead of
            # silently attributing it to a real evidence anchor.
            fallback = max_anchors - 1
            names[fallback] = "other_prompt"
            kinds[fallback] = "other"
        else:
            fallback = len(names)
            names.append("other_prompt")
            kinds.append("other")
        token_anchor[token_anchor < 0] = fallback

    return AnchorMap(
        token_anchor=token_anchor,
        names=tuple(names),
        kinds=tuple(kinds),
        mode="manifest",
    )


def uniform_prompt_anchors(
    response_idx: int,
    *,
    max_anchors: int,
    chunk_tokens: int,
    device: torch.device,
) -> AnchorMap:
    if response_idx <= 0:
        return AnchorMap(
            token_anchor=torch.empty(0, dtype=torch.long, device=device),
            names=("empty_prompt",),
            kinds=("other",),
            mode="uniform_chunks",
        )
    count = min(max_anchors, max(1, (response_idx + chunk_tokens - 1) // chunk_tokens))
    token_anchor = torch.div(
        torch.arange(response_idx, device=device) * count,
        response_idx,
        rounding_mode="floor",
    ).clamp_max(count - 1)
    return AnchorMap(
        token_anchor=token_anchor,
        names=tuple(f"prompt_chunk_{index}" for index in range(count)),
        kinds=tuple("prompt_chunk" for _ in range(count)),
        mode="uniform_chunks",
    )


def anchors_for_sample(
    sample_id: str,
    response_idx: int,
    *,
    manifest: dict[str, list[Anchor]],
    max_anchors: int,
    chunk_tokens: int,
    device: torch.device,
) -> AnchorMap:
    segments = manifest.get(str(sample_id))
    if segments:
        return _from_segments(
            response_idx,
            segments,
            max_anchors=max_anchors,
            device=device,
        )
    return uniform_prompt_anchors(
        response_idx,
        max_anchors=max_anchors,
        chunk_tokens=chunk_tokens,
        device=device,
    )
