"""Separate within-head moments, cross-head moments, and temporal persistence."""

import numpy as np

from ..fixed_graph.reference import fit_reference
from .geometry import (
    head_views,
    layer_groups,
    projection,
    second_moments,
    symmetric_vector,
    window_counts,
)

PRIMARY = "pair_full"


def causal_mean(values, positions, counts, window):
    """Trailing mean; counts reset at gaps and never treat a numeric zero as missing."""
    indices = positions[:, None] - np.arange(window)
    valid = np.arange(window) < counts[positions, None]
    extra_axes = (1,) * (values.ndim - 1)
    history = values[np.maximum(indices, 0)]
    history = np.where(valid.reshape((*valid.shape, *extra_axes)), history, 0.0)
    denominator = counts[positions].reshape((-1, *extra_axes))
    return history.sum(axis=1) / denominator


def head_pair_mask(width, signals):
    row, column = np.triu_indices(width)
    return row // signals == column // signals


def moment_parts(moment, mean, signals):
    within = head_pair_mask(moment.shape[-1], signals)
    persistent = mean[..., :, None] * mean[..., None, :]
    vector = symmetric_vector(moment)
    persistence = symmetric_vector(persistent)[:, ~within]
    return dict(
        diagonal=vector[:, within],
        cross=vector[:, ~within],
        covariance=vector[:, ~within] - persistence,
        persistence=persistence,
    )


def allocate_parts(rows, width, args):
    cross_width = int((~head_pair_mask(width, len(args.signal_indices))).sum())
    legacy_width = width * (width + 1) // 2
    within_width = legacy_width - cross_width
    widths = dict(
        diagonal=within_width, instant_diagonal=within_width, moment=args.dimensions or legacy_width
    )
    for name in ("cross", "covariance", "persistence", "instant_cross"):
        widths[name] = args.dimensions or cross_width
    for name in ("moment_diagonal", "moment_cross"):
        widths[name] = widths["moment"]
    return {name: np.empty((rows, columns)) for name, columns in widths.items()}


def legacy_parts(moment, root, signals):
    """Ablate in original head coordinates, then apply the unchanged v2 whitening."""
    head_ids = np.arange(moment.shape[-1]) // signals
    within = head_ids[:, None] == head_ids[None, :]
    diagonal = np.where(within, moment, 0.0)
    identity = np.eye(moment.shape[-1])
    return dict(
        moment=symmetric_vector(root @ moment @ root - identity),
        moment_diagonal=symmetric_vector(root @ diagonal @ root - identity),
        moment_cross=symmetric_vector(root @ (moment - diagonal) @ root),
    )


def layer_parts(values, relative, positions, counts, model, index, layer, args):
    """Only cross-head entries share a sketch; same-head blocks stay exact."""
    width = values.shape[-1]
    output = allocate_parts(len(positions), width, args)
    maps = {}
    if args.dimensions:
        cross_width = int((~head_pair_mask(width, len(args.signal_indices))).sum())
        maps["cross"] = projection(cross_width, args.dimensions, args.seed, layer, 2)
        maps["moment"] = projection(width * (width + 1) // 2, args.dimensions, args.seed, layer, 1)
    for start in range(0, len(positions), 128):
        chosen = positions[start : start + 128]
        moment = second_moments(values, chosen, counts, args.window, 0.0)
        mean = causal_mean(values, chosen, counts, args.window)
        parts = moment_parts(moment, mean, len(args.signal_indices))
        current = values[chosen]
        instant = moment_parts(
            current[:, :, None] * current[:, None, :], current, len(args.signal_indices)
        )
        parts.update(instant_diagonal=instant["diagonal"], instant_cross=instant["cross"])
        legacy = second_moments(relative, chosen, counts, args.window, args.ridge)
        parts.update(legacy_parts(legacy, model["inverse_root"][index], len(args.signal_indices)))
        for name, matrix in parts.items():
            if args.dimensions and name not in ("diagonal", "instant_diagonal"):
                matrix = matrix @ maps["moment" if name.startswith("moment") else "cross"]
            output[name][start : start + len(chosen)] = matrix
    return output


def combine_layers(parts, indices, name, dimensions):
    matrices = [parts[index][name] for index in indices]
    if dimensions and name not in ("diagonal", "instant_diagonal"):
        # Match v2's layer-order float32 accumulation for the legacy bridge.
        combined = np.zeros(matrices[0].shape, np.float32)
        for matrix in matrices:
            combined += matrix / np.sqrt(len(indices))
        return combined
    return np.concatenate(matrices, axis=1)


def embed(observations, coverage, layers, model, args, positions=None):
    raw, contrast = head_views(observations, args.signal_indices)
    raw = (raw - model["raw_center"]) / model["raw_scale"]
    relative = (contrast - model["center"]) / model["scale"]
    values = raw if args.coordinates == "raw" else relative
    counts = window_counts(coverage, args.window)
    positions = np.flatnonzero(coverage) if positions is None else np.asarray(positions)
    parts = [
        layer_parts(
            values[:, index], relative[:, index], positions, counts, model, index, layer, args
        )
        for index, layer in enumerate(layers)
    ]
    output = {}
    for group, indices in layer_groups(layers, args.layer_bands).items():
        width = len(indices) * values.shape[-1]
        current = values[positions][:, indices].reshape(len(positions), width)
        blocks = {name: combine_layers(parts, indices, name, args.dimensions) for name in parts[0]}
        choices = dict(
            pair_state=(),
            pair_diagonal=("diagonal",),
            pair_cross=("cross",),
            pair_full=("diagonal", "cross"),
            pair_covariance=("diagonal", "covariance"),
            pair_persistence=("diagonal", "persistence"),
            pair_full_w1=("instant_diagonal", "instant_cross"),
        )
        for method, names in choices.items():
            selected = [current] + [blocks[name] for name in names]
            output[f"{group}__{method}"] = np.concatenate(selected, axis=1)
        output[f"{group}__raw"] = raw[positions][:, indices].reshape(len(positions), width)
        legacy_current = relative[positions][:, indices].reshape(len(positions), width)
        for name in ("moment", "moment_diagonal", "moment_cross"):
            output[f"{group}__{name}"] = np.concatenate((legacy_current, blocks[name]), axis=1)
    return {name: matrix.astype(np.float32) for name, matrix in output.items()}, positions, counts


def fit_references(arrays, neighbors):
    """Fit one block metric on full FIT moments and reuse its columns in every ablation."""
    references = {}
    scopes = sorted({name.split("__")[0] for name in arrays})
    for scope in scopes:
        local = {
            name.split("__")[1]: values
            for name, values in arrays.items()
            if name.startswith(scope + "__")
        }
        full = fit_reference(local["pair_full"], neighbors)
        current_end = local["pair_state"].shape[1]
        diagonal_end = local["pair_diagonal"].shape[1]
        boundaries = (0, current_end, diagonal_end, len(full["scale"]))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            full["scale"][start:end] *= np.sqrt(end - start)
        columns = dict(
            pair_state=np.arange(current_end),
            pair_diagonal=np.arange(diagonal_end),
            pair_cross=np.r_[np.arange(current_end), np.arange(diagonal_end, boundaries[-1])],
        )
        legacy = fit_reference(local["moment"], neighbors)
        for method, matrix in local.items():
            name = f"{scope}__{method}"
            if method == "raw":
                references[name] = fit_reference(matrix, neighbors)
                continue
            template = legacy if method.startswith("moment") else full
            chosen = columns.get(method, np.arange(len(template["scale"])))
            center, scale = template["center"][chosen], template["scale"][chosen]
            references[name] = dict(
                center=center,
                scale=scale,
                bank=((matrix - center) / scale).astype(np.float32),
                neighbors=full["neighbors"],
            )
        references[f"{scope}__pair_state_smooth"] = dict(smooth=np.array(True))
    return references


def add_smooth_scores(scores, counts, window):
    positions = np.flatnonzero(counts)
    for name in list(scores):
        if not name.endswith("__pair_state"):
            continue
        result = np.full(len(counts), np.nan, np.float32)
        result[positions] = causal_mean(scores[name], positions, counts, window)
        scores[name + "_smooth"] = result
