"""Aligned embedding views from independently encoded construction controls.

The saved ``node_embedding`` is already the readout of GroundedRoute's typed
message passing. Downstream audits compare separately trained construction
controls; they do not run a second graph network over the saved edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experiment_protocol import scalar_text, sha256_text

from .data import GraphBundle, load_bundle


VIEW_PROTOCOL = {
    "real": ("real", "neighbor"),
    "no_message": ("real", "row_local"),
    "endpoint_rewire": ("endpoint_rewire", "neighbor"),
    "weight_shuffle": ("weight_shuffle", "neighbor"),
}


@dataclass(frozen=True)
class EmbeddingView:
    """One independently encoded variant aligned to reference token rows."""

    variant: str
    bundle: GraphBundle
    row_order: np.ndarray
    changed_fraction: float
    graph_variant: str
    message_mode: str

    @property
    def embedding(self) -> np.ndarray:
        return self.bundle.index.embedding[self.row_order]


@dataclass(frozen=True)
class AlignedEmbeddingViews:
    """Variant embeddings sharing one exact response-token row identity."""

    reference_variant: str
    views: Mapping[str, EmbeddingView]

    @property
    def reference(self) -> EmbeddingView:
        return self.views[self.reference_variant]

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.views)

    def embedding(self, variant: str) -> np.ndarray:
        return self.views[variant].embedding


def load_embedding_views(
    variant_indices: Mapping[str, str],
    *,
    reference_variant: str = "real",
) -> AlignedEmbeddingViews:
    """Load, verify and row-align independently encoded variant bundles.

    ``variant_indices`` maps the declared construction view to its encoded
    ``index.npz``. A single real bundle is valid for representation probes;
    construction claims require independently fitted controls.
    """

    if reference_variant not in variant_indices:
        raise ValueError("the reference variant is missing")
    unknown = set(variant_indices).difference(VIEW_PROTOCOL)
    if unknown:
        raise ValueError(f"unknown graph variants: {sorted(unknown)}")

    bundles = {
        variant: load_bundle(index_path)
        for variant, index_path in variant_indices.items()
    }
    for declared, bundle in bundles.items():
        graph_variant = scalar_text(bundle.metadata, "variant")
        message_mode = _message_mode(bundle.metadata)
        if (graph_variant, message_mode) != VIEW_PROTOCOL[declared]:
            raise ValueError(
                f"bundle declared as {declared!r} has graph/message mode "
                f"{(graph_variant, message_mode)!r}"
            )

    reference = bundles[reference_variant]
    reference_keys = _row_keys(reference)
    views: dict[str, EmbeddingView] = {}
    for variant, bundle in bundles.items():
        _verify_common_identity(reference, bundle)
        order = _alignment(reference_keys, _row_keys(bundle))
        _verify_aligned_columns(reference, bundle, order)
        views[variant] = EmbeddingView(
            variant=variant,
            bundle=bundle,
            row_order=order,
            changed_fraction=_scalar_number(bundle.metadata, "changed_fraction"),
            graph_variant=scalar_text(bundle.metadata, "variant"),
            message_mode=_message_mode(bundle.metadata),
        )
    return AlignedEmbeddingViews(
        reference_variant=reference_variant,
        views=views,
    )


def _verify_common_identity(reference: GraphBundle, other: GraphBundle) -> None:
    for name in ("dataset_manifest_sha256", "graph_spec_sha256"):
        if sha256_text(reference.metadata, name) != sha256_text(other.metadata, name):
            raise ValueError(f"variant bundles use different {name}")
    for name in ("split", "scope"):
        if scalar_text(reference.metadata, name) != scalar_text(other.metadata, name):
            raise ValueError(f"variant bundles use different {name}")
    if reference.index.embedding.shape[1] != other.index.embedding.shape[1]:
        raise ValueError("variant bundles use different embedding dimensions")
    if "implementation_sha256" in reference.metadata or "implementation_sha256" in other.metadata:
        if sha256_text(reference.metadata, "implementation_sha256") != sha256_text(
            other.metadata,
            "implementation_sha256",
        ):
            raise ValueError("variant bundles use different encoder implementations")


def _row_keys(bundle: GraphBundle) -> tuple[tuple[str, int], ...]:
    index = bundle.index
    return tuple(
        zip(
            index.sample_id.astype(str).tolist(),
            index.token_index.astype(np.int64).tolist(),
            strict=True,
        )
    )


def _alignment(
    reference: tuple[tuple[str, int], ...],
    candidate: tuple[tuple[str, int], ...],
) -> np.ndarray:
    if len(set(candidate)) != len(candidate):
        raise ValueError("variant embedding rows are not uniquely identified")
    location = {key: row for row, key in enumerate(candidate)}
    if set(location) != set(reference):
        raise ValueError("variant bundles cover different response-token rows")
    return np.asarray([location[key] for key in reference], dtype=np.int64)


def _verify_aligned_columns(
    reference: GraphBundle,
    other: GraphBundle,
    order: np.ndarray,
) -> None:
    left = reference.index
    right = other.index
    for name in (
        "sample_id",
        "source_id",
        "task_type",
        "token_index",
        "response_length",
        "response_token_id",
    ):
        if not np.array_equal(getattr(left, name), getattr(right, name)[order]):
            raise ValueError(f"variant bundles disagree on aligned {name}")


def _scalar_number(mapping: Mapping[str, np.ndarray], name: str) -> float:
    value = np.asarray(mapping[name])
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"artifact field {name!r} must be a scalar number")
    return float(value.item())


def _message_mode(mapping: Mapping[str, np.ndarray]) -> str:
    """Treat pre-control GroundedRoute artifacts as the default neighbour mode."""

    if "message_mode" not in mapping:
        return "neighbor"
    return scalar_text(mapping, "message_mode")
