"""Complete per-sample routing scans, independent of selected functional targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.common.llama_message_intervention import baseline_forward

from .artifacts import save_result
from .flow import SOURCE_LOCATION_BUCKET_NAMES, SourceLocationBuckets
from .native_flow import capture_source_location_buckets
from .native_world import NativeWorld
from .reanchor_timeline import StructuralReanchorTrace, structural_reanchor_trace


@dataclass(frozen=True)
class SampleScan:
    """All predictor rows and heads, captured before selecting event targets."""

    source_location: SourceLocationBuckets
    trace: StructuralReanchorTrace

    @classmethod
    def capture(
        cls,
        model,
        world: NativeWorld,
        *,
        query_chunk: int,
        local_window: int,
    ) -> SampleScan:
        """Rebuild a missing scan without changing a saved world's target plan."""

        cache = baseline_forward(
            model,
            world.token_ids,
            world.response_start,
            checkpoint_layers=range(len(model.model.layers)),
            attention_query_chunk=query_chunk,
        )
        location = capture_source_location_buckets(
            model,
            cache,
            world.units,
            cache.query,
            response_start=world.response_start,
            evidence_unit_id=world.evidence_unit_id,
            local_window=local_window,
            query_chunk=query_chunk,
        )
        return cls(location, structural_reanchor_trace(location, cache.query))

    def save(
        self,
        path: Path,
        *,
        dataset_sample_id: str,
        source_id: str,
        task_type: str,
        token_ids: torch.Tensor,
        response_start: int,
        full_response_tokens: int,
        local_window: int,
    ) -> None:
        """Persist exact four-bucket rows without head reduction or labels."""

        location = self.source_location
        save_result(
            path,
            {
                "sample_scan_schema": 1,
                "dataset_sample_id": dataset_sample_id,
                "source_id": source_id,
                "task_type": task_type,
                "response_start": response_start,
                "sequence_length": len(token_ids),
                "full_response_tokens": full_response_tokens,
                "processed_response_tokens": len(token_ids) - response_start,
                "local_window": local_window,
                "labels_used_for_capture": False,
                "token_ids": token_ids,
                "route_row_position": self.trace.row_position,
                "reanchor_bucket_name": SOURCE_LOCATION_BUCKET_NAMES,
                "reanchor_bucket_attention": location.attention,
                "reanchor_bucket_transport": location.transport,
                "reanchor_bucket_source_position": location.source_position,
                "reanchor_bucket_source_unit_id": location.source_unit_id,
                "reanchor_score": self.trace.score,
            },
        )


def render_sample_scan(path: Path, output: Path, tokenizer) -> Path:
    """Show the complete switch raster plus exact individual-head source shares."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as stored:
        transport = stored["reanchor_bucket_transport"].astype(np.float32)
        score = stored["reanchor_score"].astype(np.float32)
        positions = stored["route_row_position"].astype(np.int64)
        response_start = int(stored["response_start"])
        token_ids = stored["token_ids"].astype(np.int64)
        sample_id = str(stored["dataset_sample_id"])
        task_type = str(stored["task_type"])
        full_tokens = int(stored["full_response_tokens"])
        bucket_names = stored["reanchor_bucket_name"].tolist()

    layers, heads, rows = score.shape
    scores_by_head = score.reshape(layers * heads, rows)
    chosen = np.argsort(-scores_by_head.max(axis=1), kind="stable")[:4]
    fractions = np.divide(
        transport,
        transport.sum(axis=-1, keepdims=True),
        out=np.zeros_like(transport),
        where=transport.sum(axis=-1, keepdims=True) > 0,
    )
    x = positions + 1 - response_start
    colors = ("#247a58", "#99b7ad", "#8067b3", "#d59b42")
    figure, axes = plt.subplots(
        len(chosen) + 1,
        1,
        figsize=(15, 3.1 + 1.6 * len(chosen)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, *([1] * len(chosen))]},
        layout="constrained",
    )
    image = axes[0].imshow(
        scores_by_head,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=max(float(score.max()), 0.05),
        extent=(x[0] - 0.5, x[-1] + 0.5, layers * heads - 0.5, -0.5),
    )
    layer_ticks = np.arange(0, layers, max(1, layers // 8))
    axes[0].set_yticks(layer_ticks * heads, [f"L{layer} H0" for layer in layer_ticks])
    axes[0].set_ylabel("Every layer / head")
    axes[0].set_title(
        "Structural reanchor score; one raster row per head, no head averaging"
    )
    figure.colorbar(image, ax=axes[0], label="Local-to-long-range switch")

    for axis, flat_head in zip(axes[1:], chosen, strict=True):
        layer, head = divmod(int(flat_head), heads)
        axis.stackplot(x, *fractions[layer, head].T, colors=colors, labels=bucket_names)
        events = score[layer, head] > 0
        axis.scatter(
            x[events], np.full(events.sum(), 1.03), marker="v", s=18, c="#333333"
        )
        axis.set_ylim(0, 1.1)
        axis.set_ylabel(f"L{layer} H{head}\ntransport share")
    axes[1].legend(loc="upper left", ncols=4, fontsize=8, framealpha=0.8)
    ticks = np.unique(np.linspace(0, rows - 1, min(rows, 10), dtype=int))
    displayed = tokenizer.convert_ids_to_tokens(
        token_ids[positions[ticks] + 1].tolist()
    )
    axes[-1].set_xticks(
        x[ticks],
        [
            f"{x[index]}: {str(token)[:16]}"
            for index, token in zip(ticks, displayed, strict=True)
        ],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    axes[-1].set_xlabel("Predicted response token index and token text")
    figure.suptitle(
        f"{task_type}/{sample_id} | {rows}/{full_tokens} response decisions | "
        "structural routing only; correctness joined in cohort report",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output
