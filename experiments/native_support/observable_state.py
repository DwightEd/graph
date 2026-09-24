"""Structured conditional kernel reference; no truth labels or token likelihood fit.

Rows of each response matrix retain channel, physical head and source role.
The kernel compares their full Gram matrices without materializing H^2 graphs.
"""

import numpy as np
import torch

ROLES = ("source", "history", "self", "other")
CHANNELS = ("residual", "ffn")
CONTEXT = ("log1p_entropy", "entropy_change", "log1p_position", "log1p_prompt", "log1p_sources")
BANDWIDTH_FLOOR = 1e-6


def roles(values, count, axis):
    selected = np.moveaxis(values, axis, -1)
    result = np.stack((selected[..., :count].sum(-1), selected[..., count],
                       selected[..., count + 3], selected[..., count + 1:count + 3].sum(-1)), -1)
    return np.moveaxis(result, -1, axis)


def state_features(row, source_count):
    """Fixed head/role axes across prompts; detailed block responses stay in capture."""
    channels = [roles(row[f"response_{name}"], source_count, 2) for name in CHANNELS]
    matrix = np.stack(channels, axis=1)
    matrix = matrix.reshape(matrix.shape[0], -1, matrix.shape[-1])
    gram = np.swapaxes(matrix, -1, -2) @ matrix
    gram_norm = np.sqrt(np.square(gram).sum((-1, -2)))
    matrix = matrix / np.sqrt(np.maximum(gram_norm, 1e-30))[:, None, None]
    reading = roles(row["read_mass"], source_count, 2)
    reading = np.sqrt(reading / reading.shape[1]).reshape(reading.shape[0], -1)
    source = row["read_mass"][..., :source_count]
    mass = source.sum(-1, keepdims=True)
    distribution = np.divide(source, mass, out=np.zeros_like(source), where=mass > 0)
    total = roles(row["response_total"], source_count, 2)
    energy = np.linalg.norm(total, axis=-1).sum(1)
    route = (energy[:, 1] + energy[:, 2] - energy[:, 0]) / np.maximum(energy.sum(-1), 1e-30)
    return dict(matrix=matrix.astype(np.float32), reading=reading.astype(np.float32),
                response_scale=gram_norm, observable_route=float(route.mean()),
                source_read_mass=float(mass.mean()),
                source_concentration=float(np.square(distribution).sum(-1).mean()))


def matrix_distance(left, right):
    """Half squared Frobenius distance of normalized response Gram matrices."""
    cross = left.transpose(-1, -2) @ right
    left_self = left.transpose(-1, -2) @ left
    right_self = right.transpose(-1, -2) @ right
    distance = (left_self.square().sum((-1, -2)) + right_self.square().sum((-1, -2))
                - 2 * cross.square().sum((-1, -2))) / 2
    return distance.clamp_min(0).mean(-1)


def state_distance(left_matrix, left_read, right_matrix, right_read):
    transport = matrix_distance(left_matrix, right_matrix)
    reading = (left_read - right_read).square().sum(-1).mean(-1) / 2
    return (transport + reading) / 2


def observed_transitions(features):
    """Adjacent observed changes, not semantic boundary/reanchor labels."""
    matrix = torch.as_tensor(features["matrix"])
    response = matrix_distance(matrix[1:], matrix[:-1]).numpy()
    reading = features["reading"]
    read_change = np.square(reading[1:] - reading[:-1]).sum(-1).mean(-1) / 2
    return dict(response_change=np.r_[np.nan, response], read_change=np.r_[np.nan, read_change],
                source_mass_change=np.r_[np.nan, np.diff(features["source_read_mass"])],
                source_concentration_change=np.r_[np.nan, np.diff(features["source_concentration"])])


def context_features(entropy, target, prompt, source_count):
    delta = np.r_[0., np.diff(entropy)]
    return np.column_stack((np.log1p(entropy), delta, np.log1p(target),
                            np.full(len(target), np.log1p(prompt)),
                            np.full(len(target), np.log1p(source_count))))


def reference_bandwidth(matrix, reading, sources, seed=37):
    """Source-disjoint random pairs; adjacent rows do not define the scale."""
    generator = np.random.default_rng(seed)
    pairs = []
    for source in np.unique(sources):
        left = np.flatnonzero(sources == source)
        right = np.flatnonzero(sources != source)
        if not len(right):
            continue
        count = min(len(left), 128)
        pairs.extend(zip(generator.choice(left, count, replace=False), generator.choice(right, count)))
    if not pairs:
        raise ValueError("Conditional reference requires at least two independent reference sources")
    pair = np.asarray(pairs)
    distances = state_distance(matrix[pair[:, 0]], reading[pair[:, 0]],
                               matrix[pair[:, 1]], reading[pair[:, 1]])
    positive = distances[distances > 0]
    scale = float(positive.median()) if len(positive) else BANDWIDTH_FLOOR
    return max(scale, BANDWIDTH_FLOOR)


def context_neighbors(context, reference, source_ids, has_previous, previous_mask, count):
    """Same initialization status, up to count nearest rows per reference source."""
    distance = np.square(reference - context).mean(-1)
    chosen = []
    for source in np.unique(source_ids):
        eligible = np.flatnonzero((source_ids == source) & (previous_mask == has_previous))
        order = np.argsort(distance[eligible], kind="stable")[:count]
        chosen.extend(eligible[order])
    indices = np.asarray(chosen, dtype=int)
    return indices, distance[indices]


def kernel_anomaly(distance, log_weight, bandwidth):
    """Negative log conditional kernel affinity, not a calibrated density/probability."""
    return torch.logsumexp(log_weight, 0) - torch.logsumexp(log_weight - distance / bandwidth, 0)


def score_queries(query, reference, neighbors=16, device="cpu", seed=37):
    """Reference is already source-disjoint. Labels cannot enter this function."""
    center = np.median(reference["context"], axis=0)
    spread = np.std(reference["context"], axis=0).clip(1e-6)
    query = device_state(query, center, spread, device)
    reference = device_state(reference, center, spread, device)
    scale = reference_bandwidth(reference["matrix"], reference["reading"], reference["source"], seed)
    scores, diagnostics = [], []
    for target in range(len(query["matrix"])):
        row, detail = score_query(target, query, reference, neighbors, scale)
        scores.append(row)
        diagnostics.append(detail)
    result = {name: np.asarray([row[name] for row in scores]) for name in scores[0]}
    protocol = dict(bandwidth=scale, context_center=center.tolist(), context_scale=spread.tolist(),
                    reference_sources=np.unique(reference["source"]).tolist(),
                    reference_tokens=len(reference["matrix"]))
    return result, diagnostics, protocol


def device_state(features, center, spread, device):
    return {**features, "matrix": torch.as_tensor(features["matrix"], device=device),
            "reading": torch.as_tensor(features["reading"], device=device),
            "context": (features["context"] - center) / spread}


def score_query(target, query, reference, neighbors, scale):
    matrix, reading = reference["matrix"], reference["reading"]
    previous = query["previous"][target]
    indices, context_distance = context_neighbors(
        query["context"][target], reference["context"], reference["source"], previous >= 0,
        reference["previous"] >= 0, neighbors,
    )
    if not len(indices):
        raise ValueError("No reference rows with matching initial/continuation status")
    selected_sources = reference["source"][indices]
    counts = {source: sum(selected_sources == source) for source in np.unique(selected_sources)}
    log_weight = -context_distance / 2 - np.log([counts[source] for source in selected_sources])
    # Cancel common large context penalties before float32 log-sum-exp subtraction.
    log_weight -= log_weight.max()
    log_weight = torch.as_tensor(log_weight, device=matrix.device, dtype=matrix.dtype)
    distance = state_distance(query["matrix"][target], query["reading"][target],
                              matrix[indices], reading[indices])
    context_score = kernel_anomaly(distance, log_weight, scale)
    if previous >= 0:
        prior_indices = reference["previous"][indices]
        prior_distance = state_distance(query["matrix"][previous], query["reading"][previous],
                                        matrix[prior_indices], reading[prior_indices])
        log_weight = log_weight - prior_distance / scale
    conditional = kernel_anomaly(distance, log_weight, scale)
    weights = log_weight.softmax(0)
    detail = dict(target=target, reference_response=reference["response"][indices].tolist(),
                  reference_target=reference["target"][indices].tolist(),
                  weight=weights.cpu().tolist(), distance=distance.cpu().tolist(),
                  minimum_context_distance=float(context_distance.min()),
                  effective_neighbors=float(1 / weights.square().sum()))
    return dict(transport_conditional=float(conditional), transport_context=float(context_score)), detail
