"""Label-sealed, source-cross-fitted detection from functional message traces.

The detector models the *joint* layer/head mechanism state rather than adding a
few hand-chosen scalar scores.  All fitted quantities are learned from source-
disjoint fitting records.  A separate calibration fold turns the raw innovation
energy into a conditional empirical-tail score; neither path reads labels.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCORE_NAMES = (
    "mechanism_innovation",
    "static_state",
    "confidence",
)
SCORE_DEFINITIONS = {
    "mechanism_innovation": (
        "position/length/full-confidence-conditioned PPCA/AR(1) innovation "
        "over the mechanism tensor and four causal auxiliary channels, "
        "calibrated without labels"
    ),
    "static_state": (
        "position/length/full-confidence-conditioned PPCA state energy without "
        "the temporal transition"
    ),
    "confidence": "negative full-branch target-token log probability",
}
FACTORIAL_CHANNELS = ("evidence", "history", "interaction")
AUXILIARY_CHANNELS = (
    *FACTORIAL_CHANNELS,
    "remaining_context_margin",
)
NUISANCE_COVARIATES = ("full_logprob", "full_margin")

_EPS = 1e-6


def _array(value: Any) -> np.ndarray:
    """Convert a tensor-like value without retaining its device storage."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite(name: str, value: Any) -> np.ndarray:
    array = _array(value).astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _load_artifact(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load one artifact lazily; only explicit mechanism fields are consumed."""

    artifact = record.get("artifact")
    if artifact is not None:
        return artifact
    path = Path(record["path"])
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before ``weights_only`` was introduced.
        import torch

        return torch.load(path, map_location="cpu")


def factorial_contrasts(artifact: Mapping[str, Any]) -> np.ndarray:
    """Return symmetric evidence, history and interaction effects ``[T, 3]``.

    The common shift shared by all four branches cancels exactly.  In
    particular, raw target likelihood is not allowed into the mechanism score.
    """

    inputs = artifact["score_inputs"]
    full = _finite("full_logprob", inputs["full_logprob"])
    no_evidence = _finite("no_evidence_logprob", inputs["no_evidence_logprob"])
    no_history = _finite("no_history_logprob", inputs["no_history_logprob"])
    neither = _finite(
        "no_evidence_history_logprob", inputs["no_evidence_history_logprob"]
    )
    if full.ndim != 1 or not len(full):
        raise ValueError("factorial branches must be nonempty token vectors")
    if any(branch.shape != full.shape for branch in (no_evidence, no_history, neither)):
        raise ValueError("factorial branches must have the same token length")
    evidence = 0.5 * ((full - no_evidence) + (no_history - neither))
    history = 0.5 * ((full - no_history) + (no_evidence - neither))
    interaction = full - no_evidence - no_history + neither
    return np.stack((evidence, history, interaction), axis=1)


def causal_auxiliary_channels(artifact: Mapping[str, Any]) -> np.ndarray:
    """Return E/H/I plus the absolute remaining-context target margin.

    The fourth channel is conditioned on full log probability and full margin
    by the fit-only nuisance regression before it enters PPCA.  It is not pure
    parametric knowledge: other prompt, predictor self, residual and MLP paths
    remain after evidence/history deletion.
    """

    contrasts = factorial_contrasts(artifact)
    inputs = artifact["score_inputs"]
    remaining_margin = _finite(
        "no_evidence_history_margin", inputs["no_evidence_history_margin"]
    )
    if remaining_margin.shape != (len(contrasts),):
        raise ValueError("causal margins must match the trace token length")
    return np.column_stack((contrasts, remaining_margin))


def mechanism_tensor(artifact: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return head-resolved state ``[T,L,H,13]`` and auxiliary ``[T,L,4]``.

    Attention role composition is represented in centered log-ratio coordinates.
    Its mismatch with functional edge energy and net write size is therefore
    available to the joint model without mistaking total attention scale for a
    mechanism change.
    """

    trace = artifact["trace"]
    attention = _finite("role_attention_mass", trace["role_attention_mass"])
    edge = _finite("edge_role_energy", trace["edge_role_energy"])
    write = _finite("head_role_write_norm", trace["head_role_write_norm"])
    entropy = _finite("head_source_entropy", trace["head_source_entropy"])
    coherence = _finite("role_head_coherence", trace["role_head_coherence"])
    if attention.ndim != 4 or attention.shape[-1] != 4 or not attention.shape[1]:
        raise ValueError("role tensors must have shape [L,T,H,4] with T>0")
    if edge.shape != attention.shape or write.shape != attention.shape:
        raise ValueError("attention, edge and write role tensors must share [L,T,H,4]")
    if entropy.shape != attention.shape[:3]:
        raise ValueError("head_source_entropy must have shape [L,T,H]")
    if coherence.shape != (*attention.shape[:2], 4):
        raise ValueError("role_head_coherence must have shape [L,T,4]")
    if np.any(attention < 0) or np.any(edge < 0) or np.any(write < 0):
        raise ValueError(
            "attention, edge and write mechanism values must be nonnegative"
        )
    if not np.allclose(attention.sum(axis=-1), 1.0, atol=2e-2, rtol=0):
        raise ValueError("attention source roles must partition every causal row")
    if np.any(entropy < -2e-3) or np.any(entropy > 1 + 2e-3):
        raise ValueError("normalized source entropy must lie in [0,1]")
    if np.any(coherence < -2e-3) or np.any(coherence > 1 + 2e-3):
        raise ValueError("across-head coherence must lie in [0,1]")
    # Capture layout is [L,T,H,(role)].  Entropy was normalized by its visible
    # causal-prefix maximum at capture time.
    attention = np.moveaxis(attention, 1, 0)
    edge = np.moveaxis(edge, 1, 0)
    write = np.moveaxis(write, 1, 0)
    entropy = np.moveaxis(entropy, 1, 0)
    coherence = np.moveaxis(coherence, 1, 0)
    log_attention = np.log(attention + _EPS)
    attention_clr = log_attention - log_attention.mean(axis=-1, keepdims=True)
    tensor = np.concatenate(
        (
            attention_clr,
            np.log(edge + _EPS),
            np.log(write + _EPS),
            entropy[..., None],
        ),
        axis=-1,
    )
    return tensor, coherence


def stable_source_hash(source_id: str, seed: int = 0) -> int:
    payload = f"{seed}\0{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def source_fold_assignments(
    source_ids: Sequence[str], *, folds: int = 5, seed: int = 0
) -> dict[str, int]:
    """Assign whole sources to deterministic, hash-ordered balanced folds."""

    unique = sorted(
        set(map(str, source_ids)),
        key=lambda source: (stable_source_hash(source, seed), source),
    )
    effective = max(1, min(int(folds), len(unique)))
    return {source: rank % effective for rank, source in enumerate(unique)}


def crossfit_partitions(
    source_ids: Sequence[str], *, folds: int = 5, seed: int = 0
) -> list[dict[str, Any]]:
    """Describe strict fit/calibration/test source partitions."""

    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    occupied = sorted(set(assignment.values()))
    if len(occupied) < 3:
        return []
    result = []
    for offset, test_fold in enumerate(occupied):
        calibration_fold = occupied[(offset + 1) % len(occupied)]
        result.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": tuple(
                    fold
                    for fold in occupied
                    if fold not in {test_fold, calibration_fold}
                ),
                "test_sources": tuple(
                    source for source, fold in assignment.items() if fold == test_fold
                ),
                "calibration_sources": tuple(
                    source
                    for source, fold in assignment.items()
                    if fold == calibration_fold
                ),
                "fit_sources": tuple(
                    source
                    for source, fold in assignment.items()
                    if fold not in {test_fold, calibration_fold}
                ),
            }
        )
    return result


def relative_positions(tokens: int) -> np.ndarray:
    """Return shared token-bin coordinates, including a half-bin correction."""

    return (np.arange(tokens, dtype=np.float64) + 0.5) / max(tokens, 1)


def _position_design(tokens: int, response_start: int) -> np.ndarray:
    index = np.arange(tokens, dtype=np.float64)
    relative = relative_positions(tokens)
    return np.column_stack(
        (
            np.ones(tokens),
            relative,
            relative**2,
            relative**3,
            np.log1p(index),
            np.full(tokens, np.log1p(tokens)),
            np.full(tokens, np.log1p(response_start)),
        )
    )


def _nuisance_covariates(artifact: Mapping[str, Any], tokens: int) -> np.ndarray:
    inputs = artifact["score_inputs"]
    full_logprob = _finite("full_logprob", inputs["full_logprob"])
    full_margin = _finite("full_margin", inputs["full_margin"])
    if full_logprob.shape != (tokens,) or full_margin.shape != (tokens,):
        raise ValueError("full confidence covariates must match trace token length")
    return np.column_stack((full_logprob, full_margin))


def _nuisance_design(
    tokens: int,
    response_start: int,
    covariates: np.ndarray,
    covariate_mean: np.ndarray,
    covariate_scale: np.ndarray,
) -> np.ndarray:
    standardized = (covariates - covariate_mean) / covariate_scale
    return np.column_stack(
        (_position_design(tokens, response_start), standardized)
    )


def _fix_eigenvector_sign(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.copy()
    for column in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return vectors


def _leading_eigenvectors(covariance: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(values)[::-1][: max(1, min(rank, len(values)))]
    return _fix_eigenvector_sign(vectors[:, order])


@dataclass
class _Model:
    channel_mean: np.ndarray
    channel_scale: np.ndarray
    layer_basis: np.ndarray
    head_basis: np.ndarray
    covariate_mean: np.ndarray
    covariate_scale: np.ndarray
    nuisance: np.ndarray
    residual_scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    noise: float
    transition: np.ndarray
    innovation_inverse: np.ndarray


def _record_weights(records: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    counts = Counter(str(record["source_id"]) for record in records)
    return {id(record): 1.0 / counts[str(record["source_id"])] for record in records}


def _fit_indices(tokens: int, maximum: int) -> np.ndarray:
    """Retain deterministic, response-wide coverage in bounded fit passes."""

    if maximum <= 0 or tokens <= maximum:
        return np.arange(tokens)
    return np.unique(np.linspace(0, tokens - 1, maximum).round().astype(int))


def _raw_parts(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    artifact = _load_artifact(record)
    tensor, coherence = mechanism_tensor(artifact)
    auxiliary = causal_auxiliary_channels(artifact)
    if len(auxiliary) != len(tensor):
        raise ValueError("trace and causal channels must have the same token length")
    covariates = _nuisance_covariates(artifact, len(tensor))
    return (
        tensor,
        coherence,
        auxiliary,
        int(artifact["response_start"]),
        covariates,
    )


def _project(
    record: Mapping[str, Any],
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
    layer_basis: np.ndarray,
    head_basis: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    tensor, coherence, contrasts, response_start, covariates = _raw_parts(record)
    tensor = (tensor - channel_mean) / channel_scale
    core = np.einsum(
        "tlhc,la,hb->tabc", tensor, layer_basis, head_basis, optimize=True
    ).reshape(tensor.shape[0], -1)
    coherent = np.einsum(
        "tlr,la->tar", coherence, layer_basis, optimize=True
    ).reshape(tensor.shape[0], -1)
    state = np.concatenate((core, coherent, contrasts), axis=1)
    return state, response_start, covariates


def _fit_model(
    records: Sequence[Mapping[str, Any]],
    *,
    layer_rank: int,
    head_rank: int,
    latent_rank: int,
    max_fit_tokens_per_response: int,
) -> _Model:
    weights = _record_weights(records)
    channel_sum = channel_square = None
    covariate_sum = covariate_square = None
    total = 0.0
    for record in records:
        tensor, _, _, _, covariates = _raw_parts(record)
        indices = _fit_indices(len(tensor), max_fit_tokens_per_response)
        tensor = tensor[indices]
        covariates = covariates[indices]
        weight = weights[id(record)]
        mean = tensor.mean(axis=(0, 1, 2))
        square = np.square(tensor).mean(axis=(0, 1, 2))
        channel_sum = (
            weight * mean if channel_sum is None else channel_sum + weight * mean
        )
        channel_square = (
            weight * square
            if channel_square is None
            else channel_square + weight * square
        )
        covariate_mean = covariates.mean(axis=0)
        covariate_second = np.square(covariates).mean(axis=0)
        covariate_sum = (
            weight * covariate_mean
            if covariate_sum is None
            else covariate_sum + weight * covariate_mean
        )
        covariate_square = (
            weight * covariate_second
            if covariate_square is None
            else covariate_square + weight * covariate_second
        )
        total += weight
    channel_mean = channel_sum / total
    variance = np.maximum(channel_square / total - channel_mean**2, _EPS)
    channel_scale = np.sqrt(variance)
    covariate_mean = covariate_sum / total
    covariate_variance = np.maximum(
        covariate_square / total - covariate_mean**2, _EPS
    )
    covariate_scale = np.sqrt(covariate_variance)

    layer_cov = head_cov = None
    for record in records:
        tensor, _, _, _, _ = _raw_parts(record)
        tensor = tensor[_fit_indices(len(tensor), max_fit_tokens_per_response)]
        x = (tensor - channel_mean) / channel_scale
        weight = weights[id(record)] / x.shape[0]
        current_layer = np.einsum("tlhc,tmhc->lm", x, x, optimize=True) / (
            x.shape[2] * x.shape[3]
        )
        current_head = np.einsum("tlhc,tlkc->hk", x, x, optimize=True) / (
            x.shape[1] * x.shape[3]
        )
        layer_cov = (
            weight * current_layer
            if layer_cov is None
            else layer_cov + weight * current_layer
        )
        head_cov = (
            weight * current_head
            if head_cov is None
            else head_cov + weight * current_head
        )
    layer_basis = _leading_eigenvectors(layer_cov / total, layer_rank)
    head_basis = _leading_eigenvectors(head_cov / total, head_rank)

    # The third and final artifact pass materializes only the projected state
    # for this fold.  All dense fitting below reuses it, then releases it before
    # calibration/test scoring.
    projected = []
    for record in records:
        state, response_start, covariates = _project(
            record, channel_mean, channel_scale, layer_basis, head_basis
        )
        projected.append(
            (
                record,
                state.astype(np.float32),
                response_start,
                covariates.astype(np.float32),
            )
        )

    design_dimension = _position_design(1, 1).shape[1] + len(
        NUISANCE_COVARIATES
    )
    xtx = np.zeros((design_dimension, design_dimension), dtype=np.float64)
    xty = None
    for record, state, response_start, covariates in projected:
        indices = _fit_indices(len(state), max_fit_tokens_per_response)
        design = _nuisance_design(
            len(state),
            response_start,
            covariates,
            covariate_mean,
            covariate_scale,
        )[indices]
        state = state[indices]
        weight = weights[id(record)] / len(indices)
        xtx += weight * design.T @ design
        contribution = weight * design.T @ state
        xty = contribution if xty is None else xty + contribution
    nuisance = np.linalg.solve(
        xtx + 1e-6 * np.eye(design_dimension), xty
    )

    dimension = nuisance.shape[1]
    residual_covariance = np.zeros((dimension, dimension), dtype=np.float64)
    for record, state, response_start, covariates in projected:
        indices = _fit_indices(len(state), max_fit_tokens_per_response)
        residual = state[indices] - (
            _nuisance_design(
                len(state),
                response_start,
                covariates,
                covariate_mean,
                covariate_scale,
            )[indices]
            @ nuisance
        )
        residual_covariance += (
            weights[id(record)] * residual.T @ residual / len(residual)
        )
    residual_covariance /= total
    residual_scale = np.sqrt(np.maximum(np.diag(residual_covariance), _EPS))
    covariance = residual_covariance / np.outer(residual_scale, residual_scale)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(values)[::-1]
    rank = max(1, min(latent_rank, dimension, np.count_nonzero(values > _EPS)))
    selected = order[:rank]
    components = _fix_eigenvector_sign(vectors[:, selected])
    eigenvalues = np.maximum(values[selected], _EPS)
    discarded = values[order[rank:]]
    noise = float(max(discarded.mean() if len(discarded) else _EPS, _EPS))

    xx = np.zeros((rank, rank), dtype=np.float64)
    xy = np.zeros((rank, rank), dtype=np.float64)
    pair_total = 0.0
    latents = []
    for record, state, response_start, covariates in projected:
        residual = (
            state
            - _nuisance_design(
                len(state),
                response_start,
                covariates,
                covariate_mean,
                covariate_scale,
            )
            @ nuisance
        ) / residual_scale
        latent = residual @ components
        latents.append((record, latent))
        if len(latent) > 1:
            pairs = _fit_indices(len(latent) - 1, max_fit_tokens_per_response)
            weight = weights[id(record)] / len(pairs)
            xx += weight * latent[pairs].T @ latent[pairs]
            xy += weight * latent[pairs].T @ latent[pairs + 1]
            pair_total += weights[id(record)]
    del projected
    if pair_total:
        transition = np.linalg.solve(xx + 1e-4 * np.eye(rank), xy).T
    else:
        transition = np.zeros((rank, rank))
    innovation_cov = np.zeros((rank, rank), dtype=np.float64)
    innovation_total = 0.0
    for record, latent in latents:
        if len(latent) <= 1:
            continue
        pairs = _fit_indices(len(latent) - 1, max_fit_tokens_per_response)
        innovation = latent[pairs + 1] - latent[pairs] @ transition.T
        innovation_cov += (
            weights[id(record)] * innovation.T @ innovation / len(innovation)
        )
        innovation_total += weights[id(record)]
    if innovation_total:
        innovation_cov /= innovation_total
    else:
        innovation_cov = np.diag(eigenvalues)
    innovation_inverse = np.linalg.pinv(
        innovation_cov + 1e-5 * np.eye(rank), hermitian=True
    )
    return _Model(
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        layer_basis=layer_basis,
        head_basis=head_basis,
        covariate_mean=covariate_mean,
        covariate_scale=covariate_scale,
        nuisance=nuisance,
        residual_scale=residual_scale,
        components=components,
        eigenvalues=eigenvalues,
        noise=noise,
        transition=transition,
        innovation_inverse=innovation_inverse,
    )


def _raw_scores(
    record: Mapping[str, Any], model: _Model
) -> tuple[np.ndarray, np.ndarray]:
    """Score state and AR innovation; token zero uses initial-state PPCA energy."""

    state, response_start, covariates = _project(
        record,
        model.channel_mean,
        model.channel_scale,
        model.layer_basis,
        model.head_basis,
    )
    residual = (
        state
        - _nuisance_design(
            len(state),
            response_start,
            covariates,
            model.covariate_mean,
            model.covariate_scale,
        )
        @ model.nuisance
    ) / model.residual_scale
    latent = residual @ model.components
    reconstruction = latent @ model.components.T
    perpendicular = np.square(residual - reconstruction).sum(axis=1) / model.noise
    static = (
        np.square(latent / np.sqrt(model.eigenvalues)).sum(axis=1) + perpendicular
    ) / residual.shape[1]
    innovation = static.copy()
    if len(latent) > 1:
        delta = latent[1:] - latent[:-1] @ model.transition.T
        dynamic = np.einsum(
            "ti,ij,tj->t", delta, model.innovation_inverse, delta, optimize=True
        )
        innovation[1:] = (dynamic + perpendicular[1:]) / residual.shape[1]
    return innovation, static


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return np.interp((0.2, 0.4, 0.6, 0.8), cumulative, values)


def _calibration_tables(
    records: Sequence[Mapping[str, Any]],
    raw: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]], np.ndarray]:
    record_weights = _record_weights(records)
    lengths = np.asarray(
        [len(raw[str(record["sample_id"])][0]) for record in records], dtype=float
    )
    length_weights = np.asarray([record_weights[id(r)] for r in records])
    thresholds = _weighted_quantiles(lengths, length_weights)
    cells: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        values = raw[str(record["sample_id"])][0]
        length_bin = int(np.searchsorted(thresholds, len(values), side="right"))
        token_weight = record_weights[id(record)] / len(values)
        position_bins = np.minimum(
            9, (10 * relative_positions(len(values))).astype(int)
        )
        for token, value in enumerate(values):
            position_bin = int(position_bins[token])
            for key in (
                ("cell", position_bin, length_bin),
                ("position", position_bin),
                ("length", length_bin),
                ("all",),
            ):
                cells[key].append((float(value), token_weight))
    tables = {}
    for key, pairs in cells.items():
        pairs.sort(key=lambda pair: pair[0])
        values = np.asarray([pair[0] for pair in pairs])
        weights = np.asarray([pair[1] for pair in pairs])
        tables[key] = values, np.cumsum(weights)
    return tables, thresholds


def _ecdf(
    values: np.ndarray,
    length: int,
    tables: Mapping[tuple[Any, ...], tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    length_bin = int(np.searchsorted(thresholds, length, side="right"))
    position_bins = np.minimum(9, (10 * relative_positions(length)).astype(int))
    for token, value in enumerate(values):
        position_bin = int(position_bins[token])
        candidates = (
            ("cell", position_bin, length_bin),
            ("position", position_bin),
            ("length", length_bin),
            ("all",),
        )
        calibration_values = cumulative = None
        for key in candidates:
            if key in tables:
                calibration_values, cumulative = tables[key]
                break
        left = np.searchsorted(calibration_values, value, side="left")
        right = np.searchsorted(calibration_values, value, side="right")
        below = cumulative[left - 1] if left else 0.0
        through = cumulative[right - 1] if right else below
        result[token] = (below + 0.5 * (through - below)) / cumulative[-1]
    return result


def _confidence(record: Mapping[str, Any]) -> np.ndarray:
    artifact = _load_artifact(record)
    return -_array(artifact["score_inputs"]["full_logprob"]).astype(
        np.float64, copy=False
    )


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    folds: int = 5,
    layer_rank: int = 8,
    head_rank: int = 8,
    latent_rank: int = 64,
    max_fit_tokens_per_response: int = 128,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Return per-sample out-of-fold scores and label-free fit metadata.

    ``path`` records are loaded one at a time on every streaming fit/project
    pass.  Only final one-dimensional scores are retained.  At most
    ``max_fit_tokens_per_response`` evenly spaced states and adjacent transition pairs
    per response enter each fit pass.  Source-equal record weights, rather than
    this per-response cap, ensure that a source cannot dominate the fit.
    """

    if folds < 3:
        raise ValueError("folds must be at least three for fit/calibration/test")
    if max_fit_tokens_per_response <= 0:
        raise ValueError("max_fit_tokens_per_response must be positive")
    records = list(records)
    if not records:
        return {}, {
            "crossfit_complete": False,
            "mechanism_scores_available": False,
            "partitions": [],
        }
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id must be unique")
    records.sort(
        key=lambda record: (str(record["source_id"]), str(record["sample_id"]))
    )
    tasks = {
        str(record["task_type"])
        for record in records
        if record.get("task_type") is not None
    }
    if len(tasks) > 1:
        raise ValueError("score_records requires one task_type at a time")
    source_ids = [str(record["source_id"]) for record in records]
    assignment = source_fold_assignments(source_ids, folds=folds, seed=seed)
    partitions = crossfit_partitions(source_ids, folds=folds, seed=seed)
    output: dict[str, dict[str, np.ndarray]] = {}

    if not partitions:
        for record in records:
            confidence = _confidence(record).astype(np.float32)
            unavailable = np.zeros_like(confidence)
            output[str(record["sample_id"])] = {
                "mechanism_innovation": unavailable.copy(),
                "static_state": unavailable.copy(),
                "confidence": confidence,
            }
        return output, {
            "crossfit_complete": False,
            "mechanism_scores_available": False,
            "reason": (
                "at least three distinct sources are required for "
                "fit/calibration/test"
            ),
            "source_folds": assignment,
            "partitions": [],
            "score_definitions": SCORE_DEFINITIONS,
            "causal_auxiliary_channels": list(AUXILIARY_CHANNELS),
            "nuisance_covariates": list(NUISANCE_COVARIATES),
        }

    partition_metadata = []
    for partition in partitions:
        fit_sources = set(partition["fit_sources"])
        calibration_sources = set(partition["calibration_sources"])
        test_sources = set(partition["test_sources"])
        fit_records = [r for r in records if str(r["source_id"]) in fit_sources]
        calibration_records = [
            r for r in records if str(r["source_id"]) in calibration_sources
        ]
        test_records = [r for r in records if str(r["source_id"]) in test_sources]
        model = _fit_model(
            fit_records,
            layer_rank=layer_rank,
            head_rank=head_rank,
            latent_rank=latent_rank,
            max_fit_tokens_per_response=max_fit_tokens_per_response,
        )
        calibration_raw = {
            str(record["sample_id"]): _raw_scores(record, model)
            for record in calibration_records
        }
        primary_tables, thresholds = _calibration_tables(
            calibration_records, calibration_raw
        )
        static_raw = {
            sample: (scores[1], scores[1])
            for sample, scores in calibration_raw.items()
        }
        static_tables, _ = _calibration_tables(calibration_records, static_raw)
        for record in test_records:
            innovation, static = _raw_scores(record, model)
            sample = str(record["sample_id"])
            output[sample] = {
                "mechanism_innovation": _ecdf(
                    innovation, len(innovation), primary_tables, thresholds
                ).astype(np.float32),
                "static_state": _ecdf(
                    static, len(static), static_tables, thresholds
                ).astype(np.float32),
                "confidence": _confidence(record).astype(np.float32),
            }
        partition_metadata.append(
            {
                "test_fold": partition["test_fold"],
                "calibration_fold": partition["calibration_fold"],
                "fit_folds": list(partition["fit_folds"]),
                "fit_sources": len(fit_sources),
                "calibration_sources": len(calibration_sources),
                "test_sources": len(test_sources),
                "layer_rank": int(model.layer_basis.shape[1]),
                "head_rank": int(model.head_basis.shape[1]),
                "latent_rank": int(model.components.shape[1]),
                "fit_artifact_passes": 3,
            }
        )
    return output, {
        "crossfit_complete": True,
        "mechanism_scores_available": True,
        "seed": int(seed),
        "requested_folds": int(folds),
        "effective_folds": len(partitions),
        "source_folds": assignment,
        "partitions": partition_metadata,
        "max_fit_tokens_per_response": int(max_fit_tokens_per_response),
        "fit_artifact_passes": 3,
        "score_definitions": SCORE_DEFINITIONS,
        "causal_auxiliary_channels": list(AUXILIARY_CHANNELS),
        "nuisance_covariates": list(NUISANCE_COVARIATES),
        "nuisance_fit": "fit_sources_only",
    }
