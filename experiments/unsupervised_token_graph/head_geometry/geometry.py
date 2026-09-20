"""Signed head coordinates and causal second moments; no temporal mean feature."""

import numpy as np
from scipy import sparse


METHODS = ("raw", "contrast", "moment", "log_moment", "log_diagonal")
PRIMARY = "log_moment"


def head_views(observations, signals):
    selected = observations[..., signals]
    contrast = selected - selected.mean(axis=2, keepdims=True)
    shape = (*selected.shape[:2], -1)
    return selected.reshape(shape), contrast.reshape(shape)


def robust_scale(values):
    center = np.median(values, axis=0)
    quartiles = np.quantile(values, [.25, .75], axis=0)
    return center, np.maximum(quartiles[1] - quartiles[0], 1e-3)


def matrix_function(matrices, function):
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    if np.any(eigenvalues <= 0):
        raise ValueError("SPD descriptor lost positive definiteness")
    return (eigenvectors * function(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


def fit_geometry(observations, signals, ridge):
    raw, contrast = head_views(observations, signals)
    raw_center, raw_scale = robust_scale(raw)
    center, scale = robust_scale(contrast)
    normalized = (contrast - center) / scale
    moment = np.einsum("nli,nlj->lij", normalized, normalized) / len(normalized)
    reference = moment + ridge * np.eye(moment.shape[-1])
    return dict(raw_center=raw_center, raw_scale=raw_scale, center=center, scale=scale,
                inverse_root=matrix_function(reference, lambda value: value ** -.5),
                reference_diagonal=np.diagonal(reference, axis1=-2, axis2=-1).copy())


def window_counts(coverage, window):
    positions = np.arange(len(coverage))
    last_gap = np.maximum.accumulate(np.where(coverage, -1, positions))
    return np.minimum(window, positions - last_gap)


def second_moments(values, positions, counts, window, ridge):
    """R^T R/K relative to the frozen FIT center, not rolling covariance."""
    indices = positions[:, None] - np.arange(window)[None, :]
    valid = np.arange(window)[None, :] < counts[positions, None]
    history = np.where(valid[..., None], values[np.maximum(indices, 0)], 0.)
    moment = np.einsum("nki,nkj->nij", history, history)
    moment /= counts[positions, None, None]
    return moment + ridge * np.eye(values.shape[1])


def symmetric_vector(matrices):
    row, column = np.triu_indices(matrices.shape[-1])
    values = matrices[..., row, column].copy()
    values[..., row != column] *= np.sqrt(2.)
    return values


def projection(width, dimensions, seed, layer, block):
    """Two signed hash destinations per coordinate; fixed before seeing data."""
    random = np.random.default_rng([seed, int(layer), block])
    rows = np.repeat(np.arange(width), 2)
    columns = random.integers(dimensions, size=len(rows))
    values = random.choice([-1., 1.], size=len(rows)) / np.sqrt(2.)
    return sparse.csr_matrix((values, (rows, columns)), shape=(width, dimensions))


def relation_descriptors(relative, positions, counts, model, layer, args):
    width = relative.shape[-1] * (relative.shape[-1] + 1) // 2
    relation_map = projection(width, args.dimensions, args.seed, layer, 1) if args.dimensions else None
    relations = {name: [] for name in METHODS[2:]}
    for start in range(0, len(positions), 128):
        chosen = positions[start:start + 128]
        moment = second_moments(relative, chosen, counts, args.window, args.ridge)
        whitened = model["inverse_root"] @ moment @ model["inverse_root"]
        linear = whitened - np.eye(moment.shape[-1])
        logarithm = matrix_function(whitened, np.log)
        diagonal = np.zeros_like(moment)
        index = np.arange(moment.shape[-1])
        diagonal[:, index, index] = np.log(
            np.diagonal(moment, axis1=-2, axis2=-1) / model["reference_diagonal"])
        for name, matrices in zip(METHODS[2:], (linear, logarithm, diagonal)):
            vector = symmetric_vector(matrices)
            relations[name].append(vector @ relation_map if args.dimensions else vector)
    return {name: np.concatenate(values) for name, values in relations.items()}


def layer_groups(layers, bands):
    result = {"all": np.arange(len(layers))}
    if bands and len(layers) >= 4:
        for indices in np.array_split(np.arange(len(layers)), 4):
            result[f"L{layers[indices[0]]}-{layers[indices[-1]]}"] = indices
    return result


def allocate_embeddings(raw, relative, positions, groups, dimensions):
    output, relations = {}, {}
    pair_width = raw.shape[-1] * (raw.shape[-1] + 1) // 2
    for group, indices in groups.items():
        current_width = len(indices) * raw.shape[-1]
        output[f"{group}__raw"] = raw[positions][:, indices].reshape(len(positions), current_width)
        output[f"{group}__contrast"] = relative[positions][:, indices].reshape(len(positions), current_width)
        width = dimensions or len(indices) * pair_width
        for method in METHODS[2:]:
            relations[f"{group}__{method}"] = np.zeros((len(positions), width), np.float32)
    return output, relations


def add_layer_relations(relations, values, groups, index, dimensions):
    for group, indices in groups.items():
        if index not in indices:
            continue
        for method, matrix in values.items():
            if dimensions:
                relations[f"{group}__{method}"] += matrix / np.sqrt(len(indices))
            else:
                slot = int(np.flatnonzero(indices == index)[0])
                begin = slot * matrix.shape[1]
                relations[f"{group}__{method}"][:, begin:begin + matrix.shape[1]] = matrix


def embed(observations, coverage, layers, model, args, positions=None):
    raw, contrast = head_views(observations, args.signal_indices)
    raw = (raw - model["raw_center"]) / model["raw_scale"]
    relative = (contrast - model["center"]) / model["scale"]
    counts = window_counts(coverage, args.window)
    positions = np.flatnonzero(coverage) if positions is None else np.asarray(positions)
    groups = layer_groups(layers, args.layer_bands)
    output, relations = allocate_embeddings(raw, relative, positions, groups, args.dimensions)
    for index, layer in enumerate(layers if len(positions) else []):
        local = {key: model[key][index] for key in ("inverse_root", "reference_diagonal")}
        values = relation_descriptors(relative[:, index], positions, counts, local, layer, args)
        add_layer_relations(relations, values, groups, index, args.dimensions)
    for name, matrix in relations.items():
        group = name.split("__")[0]
        output[name] = np.concatenate((output[f"{group}__contrast"], matrix), axis=1)
    output = {name: values.astype(np.float32) for name, values in output.items()}
    return output, positions, counts
