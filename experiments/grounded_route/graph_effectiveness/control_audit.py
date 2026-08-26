"""Verify matched-control invariants in independently encoded graph bundles."""

from __future__ import annotations

import numpy as np
import torch

from .views import AlignedEmbeddingViews


def audit_controls(
    views: AlignedEmbeddingViews,
    *,
    minimum_changed_fraction: float = 0.10,
    minimum_sample_changed_fraction: float = 0.05,
    minimum_effective_samples: float = 0.80,
) -> dict[str, object]:
    """Verify each saved control against the real sidecars, sample by sample."""

    real_records = {
        record.sample_id: record for record in views.reference.bundle.records
    }
    report: dict[str, object] = {}
    for variant, view in views.views.items():
        if variant == "real":
            continue
        control_records = {
            record.sample_id: record for record in view.bundle.records
        }
        changed = []
        totals = []
        for sample_id, real_record in real_records.items():
            real = real_record.load()
            control = control_records[sample_id].load()
            current = verify_control_graph(real, control, variant)
            changed.append(current)
            totals.append(int(real.edge_weight.numel()))

        changed_array = np.asarray(changed, dtype=np.int64)
        total_array = np.asarray(totals, dtype=np.int64)
        fraction = np.divide(
            changed_array,
            total_array,
            out=np.zeros_like(changed_array, dtype=np.float64),
            where=total_array > 0,
        )
        global_fraction = float(changed_array.sum() / total_array.sum())
        metadata_tolerance = 1.0 / max(1, int(total_array.sum()))
        metadata_matches = bool(
            np.isclose(
                global_fraction,
                view.changed_fraction,
                atol=metadata_tolerance,
            )
        )
        effective_samples = float(
            np.mean(fraction >= minimum_sample_changed_fraction)
        )
        message_ablation = variant == "no_message"
        sufficient = bool(
            metadata_matches
            and (
                message_ablation
                or (
                    global_fraction >= minimum_changed_fraction
                    and effective_samples >= minimum_effective_samples
                )
            )
        )
        report[variant] = {
            "invariants_verified": True,
            "control_type": (
                "no_neighbor_message" if message_ablation else "graph_intervention"
            ),
            "changed_edges": int(changed_array.sum()),
            "total_edges": int(total_array.sum()),
            "changed_fraction": global_fraction,
            "reported_changed_fraction": view.changed_fraction,
            "reported_changed_fraction_matches": metadata_matches,
            "minimum_changed_fraction": minimum_changed_fraction,
            "samples_changed_at_least_5pct": effective_samples,
            "minimum_effective_samples": minimum_effective_samples,
            "intervention_sufficient": sufficient,
        }
    return report


def verify_control_graph(real, control, variant: str) -> int:
    """Verify one persisted matched control and return its changed-edge count."""

    _common_graph_invariants(real, control)
    if variant == "endpoint_rewire":
        return _endpoint_invariants(real, control)
    if variant == "weight_shuffle":
        return _weight_invariants(real, control)
    if variant == "no_message":
        return _no_message_invariants(real, control)
    raise ValueError(f"unsupported saved control: {variant}")


def _common_graph_invariants(real, control) -> None:
    scalar = (
        "sample_id",
        "source_id",
        "task_type",
        "response_start",
        "layer_count",
        "head_count",
        "attention_floor",
    )
    if any(getattr(real, name) != getattr(control, name) for name in scalar):
        raise ValueError("control graph identity or geometry differs from real")
    for name in ("token_ids", "diagonal", "unresolved"):
        if not torch.equal(getattr(real, name), getattr(control, name)):
            raise ValueError(f"control graph changed {name}")
    for name in ("edge_index", "edge_layer", "edge_head", "edge_weight"):
        if getattr(real, name).shape != getattr(control, name).shape:
            raise ValueError(f"control graph changed {name} shape")


def _endpoint_invariants(real, control) -> int:
    if not torch.equal(real.edge_index[1], control.edge_index[1]):
        raise ValueError("endpoint rewire changed target endpoints")
    for name in ("edge_layer", "edge_head", "edge_weight"):
        if not torch.equal(getattr(real, name), getattr(control, name)):
            raise ValueError(f"endpoint rewire changed {name}")

    real_source = real.edge_index[0]
    control_source = control.edge_index[0]
    real_role = real_source >= real.response_start
    control_role = control_source >= control.response_start
    if not torch.equal(real_role, control_role):
        raise ValueError("endpoint rewire changed source role")
    if not torch.equal(
        _lag_bucket(real.edge_index),
        _lag_bucket(control.edge_index),
    ):
        raise ValueError("endpoint rewire changed coarse lag bucket")
    if not torch.equal(_typed_source_degree(real), _typed_source_degree(control)):
        raise ValueError("endpoint rewire changed typed source degree")
    return int((real_source != control_source).sum().item())


def _weight_invariants(real, control) -> int:
    if not torch.equal(real.edge_index, control.edge_index):
        raise ValueError("weight shuffle changed endpoints")
    for name in ("edge_layer", "edge_head"):
        if not torch.equal(getattr(real, name), getattr(control, name)):
            raise ValueError(f"weight shuffle changed {name}")
    group = _weight_group(real)
    if not _same_grouped_weight_multiset(
        group,
        real.edge_weight,
        control.edge_weight,
    ):
        raise ValueError("weight shuffle changed a stratum weight multiset")
    return int((real.edge_weight != control.edge_weight).sum().item())


def _no_message_invariants(real, control) -> int:
    """Require the message ablation to receive the exact same saved graph."""

    for name in ("edge_index", "edge_layer", "edge_head", "edge_weight"):
        if not torch.equal(getattr(real, name), getattr(control, name)):
            raise ValueError(f"no-message control changed {name}")
    return 0


def _lag_bucket(edge_index: torch.Tensor) -> torch.Tensor:
    lag = (edge_index[1] - edge_index[0]).clamp_min(1).float()
    return torch.floor(torch.log2(lag)).long()


def _typed_source_key(graph) -> torch.Tensor:
    relation = graph.edge_layer * graph.head_count + graph.edge_head
    return relation * int(graph.token_ids.numel()) + graph.edge_index[0]


def _typed_source_degree(graph) -> torch.Tensor:
    size = graph.layer_count * graph.head_count * int(graph.token_ids.numel())
    return torch.bincount(_typed_source_key(graph), minlength=size)


def _weight_group(graph) -> torch.Tensor:
    role = (graph.edge_index[0] >= graph.response_start).long()
    return (
        (
            (graph.edge_index[1] * graph.layer_count + graph.edge_layer)
            * graph.head_count
            + graph.edge_head
        )
        * 2
        + role
    )


def _same_grouped_weight_multiset(
    group: torch.Tensor,
    real_weight: torch.Tensor,
    control_weight: torch.Tensor,
) -> bool:
    """Compare exact row-role multisets with two in-place packed-key sorts."""

    if not len(group):
        return True
    real_key = _grouped_weight_key(group, real_weight).numpy()
    control_key = _grouped_weight_key(group, control_weight).numpy()
    real_key.sort()
    control_key.sort()
    return bool(np.array_equal(real_key, control_key))


def _grouped_weight_key(group: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    bits = weight.float().contiguous().view(torch.int32).long()
    return group.long() * (1 << 32) + bits
