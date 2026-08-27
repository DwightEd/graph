"""Build and persist predecessor-aligned attention-routing lineage traces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from experiment_protocol import scalar_text, validate_complete_token_rows
from experiments.grounded_route.artifacts import load_npz, save_npz
from experiments.grounded_route.graph import TokenEdges, TokenGraph
from experiments.grounded_route.graph_effectiveness.data import load_bundle

from .lineage_controls import (
    LINEAGE_CONTROLS,
    apply_lineage_control,
    response_carrier_edges,
)
from .routing_dispersion import LOWER, UPPER, attention_routing_dispersion
from .routing_lineage import (
    DIRECT,
    ENDOGENOUS,
    INDIRECT,
    UNRESOLVED,
    ordered_routing_lineage,
)


TRACE_SCHEMA = "attention-routing-lineage-trace"
ARTIFACT_VERSION = 1
DEFAULT_EPSILON = 1e-8
PERSISTED_MASS_TOLERANCE = 4e-3

ROW_FIELDS = (
    "sample_id",
    "source_id",
    "task_type",
    "token_index",
    "response_length",
    "response_token_id",
    "prompt_length",
)


def encoded_to_token_graph(encoded) -> TokenGraph:
    """Reconstruct the exact token graph without reading ``node_embedding``."""

    edges = TokenEdges(
        source=encoded.edge_index[0].long(),
        target=encoded.edge_index[1].long(),
        layer=encoded.edge_layer.long(),
        head=encoded.edge_head.long(),
        weight=encoded.edge_weight.float(),
    )
    token_count = int(encoded.token_ids.numel())
    return TokenGraph(
        sample_id=str(encoded.sample_id),
        source_id=str(encoded.source_id),
        task_type=str(encoded.task_type),
        response_start=int(encoded.response_start),
        token_count=token_count,
        response_count=token_count - int(encoded.response_start),
        layer_count=int(encoded.layer_count),
        head_count=int(encoded.head_count),
        attention_floor=float(encoded.attention_floor),
        edges=edges,
        diagonal=encoded.diagonal.float(),
        unresolved=encoded.unresolved.float(),
        token_ids=encoded.token_ids.long(),
    ).check().canonicalize()


def direct_prompt_lookback(graph: TokenGraph) -> torch.Tensor:
    """Mean retained prompt mass in each cached response-query row.

    The returned rows still index queries.  A caller must shift row ``i`` to
    generated response token ``i + 1`` and drop the final cached query.  Sparse
    mass below the cache threshold remains unresolved and is never treated as
    observed zero prompt attention.
    """

    graph = graph.canonicalize()
    prompt_mass = graph.diagonal.new_zeros(
        (graph.response_count, graph.layer_count, graph.head_count)
    )
    prompt = graph.edges.source < graph.response_start
    if bool(prompt.any()):
        edges = graph.edges.select(prompt)
        prompt_mass.index_put_(
            (
                edges.target - graph.response_start,
                edges.layer,
                edges.head,
            ),
            edges.weight.to(prompt_mass),
            accumulate=True,
        )
    return prompt_mass.mean(dim=(1, 2))


def sample_seed(seed: int, sample_id: str) -> int:
    """Derive an order-independent deterministic seed for one graph."""

    digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def carrier_changed_fraction(original: TokenGraph, controlled: TokenGraph) -> float:
    """Fraction of retained response-carrier endpoints changed by the null."""

    def endpoints(graph: TokenGraph) -> set[tuple[int, int, int, int]]:
        edges = graph.edges.select(response_carrier_edges(graph))
        return set(
            zip(
                edges.source.tolist(),
                edges.target.tolist(),
                edges.layer.tolist(),
                edges.head.tolist(),
                strict=True,
            )
        )

    before = endpoints(original)
    if not before:
        return 0.0
    after = endpoints(controlled)
    return float(1.0 - len(before.intersection(after)) / len(before))


def complete_lineage_rows(lineage: torch.Tensor, response_count: int) -> torch.Tensor:
    """Insert the explicit unavailable boundary row before aligned lineage."""

    complete = lineage.new_zeros((response_count, 4))
    complete[0, UNRESOLVED] = 1.0
    if response_count > 1:
        complete[1:] = lineage
    return complete


def complete_trace_rows(trace: torch.Tensor, response_count: int) -> torch.Tensor:
    """Insert a U-only token-zero row into an ordered ``[R-1,L,4]`` trace."""

    complete = trace.new_zeros((response_count, trace.shape[1], 4))
    complete[0, :, UNRESOLVED] = 1.0
    if response_count > 1:
        complete[1:] = trace
    return complete


def complete_aligned_rows(value: torch.Tensor, response_count: int) -> torch.Tensor:
    """Insert an unavailable all-zero token-zero row before aligned values."""

    complete = value.new_zeros((response_count, *value.shape[1:]))
    if response_count > 1:
        complete[1:] = value
    return complete


def raw_takeover(lineage: torch.Tensor, epsilon: float = DEFAULT_EPSILON) -> torch.Tensor:
    """Preregistered response-rooted versus prompt-rooted log mass ratio."""

    return torch.log(
        (lineage[:, ENDOGENOUS] + float(epsilon))
        / (
            lineage[:, DIRECT]
            + lineage[:, INDIRECT]
            + float(epsilon)
        )
    )


@torch.no_grad()
def trace_graph(
    graph: TokenGraph,
    *,
    seed: int,
    carrier_rewire_passes: int = 4,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, np.ndarray]:
    """Compute all preregistered lineage views for one graph."""

    graph = graph.canonicalize()
    response_count = graph.response_count
    graph_seed = sample_seed(seed, graph.sample_id)
    result: dict[str, np.ndarray] = {
        "available": np.arange(response_count, dtype=np.int32) > 0,
        "predictor_token_index": np.arange(response_count, dtype=np.int32) - 1,
    }

    ordered_trace = None
    carrier_fraction = 0.0
    for control in LINEAGE_CONTROLS:
        # Layer-order nulls must use one experiment-wide permutation.  Only
        # endpoint rewiring is sample-seeded so graph iteration order cannot
        # change its realized swaps.
        control_seed = graph_seed if control == "carrier_rewire" else int(seed)
        controlled, order = apply_lineage_control(
            graph,
            control,
            seed=control_seed,
            carrier_rewire_passes=carrier_rewire_passes,
        )
        routed = ordered_routing_lineage(controlled, layer_order=order)
        complete = complete_lineage_rows(routed.token_lineage, response_count)
        takeover = raw_takeover(complete, epsilon)
        takeover[0] = 0.0
        result[f"{control}_lineage"] = complete.cpu().numpy().astype(np.float32)
        result[f"{control}_raw_takeover"] = takeover.cpu().numpy().astype(np.float32)
        padded_order = np.full(graph.layer_count, -1, dtype=np.int16)
        padded_order[: len(order)] = np.asarray(order, dtype=np.int16)
        result[f"{control}_layer_order"] = np.repeat(
            padded_order[None], response_count, axis=0
        )
        if control == "ordered":
            ordered_trace = complete_trace_rows(
                routed.query_trace[:-1], response_count
            )
            same_token = routed.query_lineage
            same_token_takeover = raw_takeover(same_token, epsilon)
            result["posthoc_same_token_lineage"] = (
                same_token.cpu().numpy().astype(np.float32)
            )
            result["posthoc_same_token_raw_takeover"] = (
                same_token_takeover.cpu().numpy().astype(np.float32)
            )
        if control == "carrier_rewire":
            carrier_fraction = carrier_changed_fraction(graph, controlled)

    if ordered_trace is None:
        raise ValueError("LINEAGE_CONTROLS must include the ordered path")
    result["ordered_representation"] = (
        ordered_trace.reshape(response_count, -1).cpu().numpy().astype(np.float32)
    )

    dispersion = attention_routing_dispersion(graph)
    entropy = complete_aligned_rows(
        dispersion.token_entropy_bounds,
        response_count,
    )
    concentration = complete_aligned_rows(
        dispersion.token_concentration_bounds,
        response_count,
    )
    role_mass = complete_aligned_rows(
        dispersion.token_role_mass,
        response_count,
    )
    role_std = complete_aligned_rows(
        dispersion.token_role_mass_disagreement,
        response_count,
    )
    role_js = complete_aligned_rows(
        dispersion.token_role_js_disagreement,
        response_count,
    )
    result["routing_entropy_lower"] = (
        entropy[..., LOWER].cpu().numpy().astype(np.float32)
    )
    result["routing_entropy_upper"] = (
        entropy[..., UPPER].cpu().numpy().astype(np.float32)
    )
    result["routing_concentration_lower"] = (
        concentration[..., LOWER].cpu().numpy().astype(np.float32)
    )
    result["routing_concentration_upper"] = (
        concentration[..., UPPER].cpu().numpy().astype(np.float32)
    )
    result["routing_role_mass"] = role_mass.cpu().numpy().astype(np.float32)
    result["routing_head_role_std"] = role_std.cpu().numpy().astype(np.float32)
    result["routing_role_js"] = role_js.cpu().numpy().astype(np.float32)
    routing_trace = torch.cat(
        (
            ordered_trace,
            entropy,
            concentration,
            role_mass,
            role_std,
            role_js[..., None],
        ),
        dim=-1,
    )
    result["routing_representation"] = (
        routing_trace.reshape(response_count, -1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    query_lookback = direct_prompt_lookback(graph)
    lookback = np.zeros(response_count, dtype=np.float32)
    if response_count > 1:
        lookback[1:] = query_lookback[:-1].cpu().numpy().astype(np.float32)
    result["direct_prompt_lookback"] = lookback
    result["known_mass"] = (
        1.0 - result["ordered_lineage"][:, UNRESOLVED]
    ).astype(np.float32)
    result["carrier_rewire_changed_fraction"] = np.full(
        response_count, carrier_fraction, dtype=np.float32
    )
    return result


def row_identity(graph: TokenGraph) -> dict[str, np.ndarray]:
    """Return complete response-token identity columns for one graph."""

    count = graph.response_count
    return {
        "sample_id": np.repeat(graph.sample_id, count),
        "source_id": np.repeat(graph.source_id, count),
        "task_type": np.repeat(graph.task_type, count),
        "token_index": np.arange(count, dtype=np.int32),
        "response_length": np.full(count, count, dtype=np.int32),
        "response_token_id": graph.response_token_ids.cpu().numpy().astype(np.int64),
        "prompt_length": np.full(count, graph.response_start, dtype=np.int32),
    }


def concatenate_rows(blocks: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate row-aligned graph blocks without changing their fields."""

    names = tuple(blocks[0])
    if any(tuple(block) != names for block in blocks[1:]):
        raise ValueError("lineage graph blocks expose different row fields")
    return {name: np.concatenate([block[name] for block in blocks]) for name in names}


def align_to_index(rows: dict[str, np.ndarray], index) -> dict[str, np.ndarray]:
    """Align generated sidecar rows to the embedding index token order."""

    generated_keys = list(
        zip(rows["sample_id"].astype(str), rows["token_index"].tolist(), strict=True)
    )
    locations = {key: position for position, key in enumerate(generated_keys)}
    index_keys = list(
        zip(index.sample_id.astype(str), index.token_index.tolist(), strict=True)
    )
    if len(locations) != len(generated_keys) or set(locations) != set(index_keys):
        raise ValueError("encoded graph sidecars do not match embedding-index rows")
    order = np.asarray([locations[key] for key in index_keys], dtype=np.int64)
    aligned = {name: value[order] for name, value in rows.items()}
    expected = {
        "sample_id": index.sample_id.astype(str),
        "source_id": index.source_id.astype(str),
        "task_type": index.task_type.astype(str),
        "token_index": index.token_index.astype(np.int32),
        "response_length": index.response_length.astype(np.int32),
        "response_token_id": index.response_token_id.astype(np.int64),
    }
    if any(not np.array_equal(aligned[name], value) for name, value in expected.items()):
        raise ValueError("encoded graph identity differs from embedding-index rows")
    return aligned


def copied_metadata(metadata: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Retain only provenance needed to bind calibration and evaluation."""

    names = (
        "dataset_manifest_sha256",
        "split",
        "scope",
        "audit_scope",
        "reserved_source_ids",
        "test_source_ids",
        "test_sample_ids",
        "calibration_sample_ids",
        "calibration_source_ids",
        "alignment",
        "prompt_partition",
        "functional_contribution_observed",
        "drift_observed",
        "dispersion_observed",
        "parametric_bias_observed",
        "layer_count",
        "head_count",
        "attention_floor",
        "seed",
        "carrier_rewire_passes",
        "takeover_epsilon",
    )
    return {name: np.asarray(metadata[name]) for name in names if name in metadata}


def sample_level_values(
    sample_id: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select one repeated sample-level value from each complete response."""

    first = []
    for sample in dict.fromkeys(np.asarray(sample_id).astype(str).tolist()):
        first.append(np.flatnonzero(sample_id == sample)[0])
    selected = np.asarray(first, dtype=np.int64)
    return np.asarray(sample_id)[selected], np.asarray(value)[selected]


def export_trace(
    index_path,
    output_path,
    *,
    seed: int = 20260827,
    carrier_rewire_passes: int = 4,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, object]:
    """Export a label-free trace from verified encoded-graph sidecars."""

    bundle = load_bundle(index_path)
    blocks: list[dict[str, np.ndarray]] = []
    layer_count = head_count = attention_floor = None
    for record in bundle.records:
        graph = encoded_to_token_graph(record.load())
        if graph.sample_id != record.sample_id or graph.source_id != record.source_id:
            raise ValueError("encoded graph identity differs from sidecar index")
        geometry = (graph.layer_count, graph.head_count, graph.attention_floor)
        if layer_count is None:
            layer_count, head_count, attention_floor = geometry
        elif geometry != (layer_count, head_count, attention_floor):
            raise ValueError("encoded graphs use different attention geometry")
        blocks.append(
            {
                **row_identity(graph),
                **trace_graph(
                    graph,
                    seed=seed,
                    carrier_rewire_passes=carrier_rewire_passes,
                    epsilon=epsilon,
                ),
            }
        )

    if not blocks:
        raise ValueError("embedding bundle contains no encoded graph sidecars")
    rows = align_to_index(concatenate_rows(blocks), bundle.index)
    validate_complete_token_rows(
        rows["sample_id"], rows["source_id"], rows["token_index"], rows["response_length"]
    )
    arrays = {
        **rows,
        **copied_metadata(bundle.metadata),
        "schema": np.asarray(TRACE_SCHEMA),
        "version": np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        "labels_included": np.asarray(False),
        "source_index_sha256": np.asarray(bundle.index_sha256),
        "controls": np.asarray(LINEAGE_CONTROLS),
        "layer_count": np.asarray(layer_count, dtype=np.int16),
        "head_count": np.asarray(head_count, dtype=np.int16),
        "attention_floor": np.asarray(attention_floor, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int64),
        "carrier_rewire_passes": np.asarray(carrier_rewire_passes, dtype=np.int16),
        "takeover_epsilon": np.asarray(epsilon, dtype=np.float64),
        "alignment": np.asarray("predecessor_response_query"),
        "prompt_partition": np.asarray("prompt_vs_response_only"),
        "functional_contribution_observed": np.asarray(False),
        "drift_observed": np.asarray(True),
        "dispersion_observed": np.asarray(True),
        "parametric_bias_observed": np.asarray(False),
    }
    save_npz(output_path, **arrays)
    _, changed = sample_level_values(
        rows["sample_id"], rows["carrier_rewire_changed_fraction"]
    )
    mass = rows["ordered_lineage"].sum(axis=1)
    role_mass_error = np.abs(
        rows["routing_role_mass"][rows["available"]].sum(axis=-1) - 1.0
    )
    return {
        "trace": str(Path(output_path).resolve()),
        "samples": len(changed),
        "nodes": len(rows["sample_id"]),
        "available_nodes": int(rows["available"].sum()),
        "representation_dimension": int(rows["ordered_representation"].shape[1]),
        "routing_representation_dimension": int(
            rows["routing_representation"].shape[1]
        ),
        "carrier_rewire_changed_fraction": float(changed.mean()),
        "carrier_rewire_nonzero_sample_fraction": float((changed > 0).mean()),
        "lineage_mass_max_error": float(np.max(np.abs(mass - 1.0))),
        "routing_role_mass_max_error": float(role_mass_error.max()),
        "labels_read": False,
    }


def validate_routing_fields(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate persisted dispersion intervals and representation geometry."""

    if "routing_entropy_lower" not in arrays:
        return
    row_count = len(arrays["sample_id"])
    layer_count = int(np.asarray(arrays["layer_count"]).item())
    expected = {
        "routing_entropy_lower": (row_count, layer_count),
        "routing_entropy_upper": (row_count, layer_count),
        "routing_concentration_lower": (row_count, layer_count),
        "routing_concentration_upper": (row_count, layer_count),
        "routing_role_mass": (row_count, layer_count, 4),
        "routing_head_role_std": (row_count, layer_count, 4),
        "routing_role_js": (row_count, layer_count),
        "ordered_representation": (row_count, layer_count * 4),
        "routing_representation": (row_count, layer_count * 17),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} has invalid shape or non-finite values")

    bounded = (
        "routing_entropy_lower",
        "routing_entropy_upper",
        "routing_concentration_lower",
        "routing_concentration_upper",
        "routing_role_mass",
        "routing_role_js",
    )
    if any(
        np.any(np.asarray(arrays[name]) < -1e-6)
        or np.any(np.asarray(arrays[name]) > 1.0 + 1e-6)
        for name in bounded
    ):
        raise ValueError("routing dispersion values must lie in [0, 1]")
    for prefix in ("routing_entropy", "routing_concentration"):
        if np.any(
            np.asarray(arrays[f"{prefix}_lower"])
            > np.asarray(arrays[f"{prefix}_upper"]) + 1e-6
        ):
            raise ValueError(f"{prefix} lower bound exceeds its upper bound")

    available = np.asarray(arrays["available"], dtype=bool)
    role_mass = np.asarray(arrays["routing_role_mass"])
    if bool(available.any()) and not np.allclose(
        role_mass[available].sum(axis=-1),
        1.0,
        atol=PERSISTED_MASS_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("available routing role mass does not sum to one")
    if np.any(np.asarray(arrays["routing_head_role_std"]) < -1e-6):
        raise ValueError("head-role disagreement cannot be negative")
    dispersion_fields = bounded + ("routing_head_role_std",)
    if any(np.any(np.asarray(arrays[name])[~available]) for name in dispersion_fields):
        raise ValueError("unavailable boundary rows must contain zero dispersion")


def require_artifact(path, schema: str) -> dict[str, np.ndarray]:
    """Load one complete-row lineage artifact and validate its boundary mask."""

    arrays = load_npz(path)
    if scalar_text(arrays, "schema") != schema:
        raise ValueError(f"artifact is not {schema}")
    if int(np.asarray(arrays["version"]).item()) != ARTIFACT_VERSION:
        raise ValueError("unsupported lineage artifact version")
    if bool(np.asarray(arrays["labels_included"]).item()):
        raise ValueError("lineage artifacts must not contain labels")
    validate_complete_token_rows(
        arrays["sample_id"], arrays["source_id"], arrays["token_index"], arrays["response_length"]
    )
    expected_available = np.asarray(arrays["token_index"]) > 0
    if not np.array_equal(np.asarray(arrays["available"], dtype=bool), expected_available):
        raise ValueError("availability mask does not encode predecessor-query boundary")
    expected_predictor = np.asarray(arrays["token_index"], dtype=np.int64) - 1
    if not np.array_equal(
        np.asarray(arrays["predictor_token_index"], dtype=np.int64),
        expected_predictor,
    ):
        raise ValueError("predictor rows are not predecessor-query aligned")
    validate_routing_fields(arrays)
    return arrays
