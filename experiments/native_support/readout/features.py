"""Label-independent observations and within-answer temporal features."""

import numpy as np

ROLES = ("source", "history", "self", "other")
SCALARS = ("raw_route", "observable_route", "raw_attention", "entropy",
           "source_read_mass", "source_concentration", "response_change", "read_change",
           "source_mass_change", "source_concentration_change")
OPERATORS = ("sketch_effective_rank", "pullback_energy", "skip_energy",
             "ffn_pullback_energy", "pullback_alignment")


def context_features(response, source_count):
    count = len(response["token_ids"]) - response["prompt_length"]
    target = np.arange(count)
    values = np.column_stack((target / max(count - 1, 1), np.log1p(target),
                              np.full(count, np.log1p(count)),
                              np.full(count, np.log1p(response["prompt_length"])),
                              np.full(count, np.log1p(source_count))))
    return values, ["relative_position", "log_position", "log_answer_length",
                    "log_prompt_length", "log_source_count"]


def temporal_features(values, names, window):
    """Current, past mean, future mean and innovation; no gold boundaries or labels."""
    count = len(values)
    finite = np.isfinite(values)
    sums = np.vstack((np.zeros(values.shape[1]), np.cumsum(np.where(finite, values, 0), axis=0)))
    counts = np.vstack((np.zeros(values.shape[1]), np.cumsum(finite, axis=0)))
    target = np.arange(count)
    means = []
    for left, right in ((np.maximum(0, target - window), target),
                        (target + 1, np.minimum(count, target + window + 1))):
        total, observed = sums[right] - sums[left], counts[right] - counts[left]
        means.append(np.divide(total, observed, out=np.full_like(total, np.nan), where=observed > 0))
    past, future = means
    matrix = np.column_stack((values, past, future, values - past, target > 0, target + 1 < count))
    columns = [f"{channel}/{name}" for channel in ("current", "past", "future", "innovation") for name in names]
    return matrix, columns + ["has_past", "has_future"]


def role_responses(values, source_count):
    return np.stack((values[..., :source_count, :].sum(-2), values[..., source_count, :],
                     values[..., source_count + 3, :],
                     values[..., source_count + 1:source_count + 3, :].sum(-2)), axis=-2)


def cosine(left, right):
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide((left * right).sum(-1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def block_state(attention, response, memory, previous):
    """A statistical source memory, not inheritance of a previous factual label."""
    mass = attention.sum(-1, keepdims=True)
    distribution = np.divide(attention, mass, out=np.zeros_like(attention), where=mass > 0)
    energy = np.linalg.norm(response, axis=-1)
    total = energy.sum(-1, keepdims=True)
    response_distribution = np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)
    memory_mass = memory.sum(-1, keepdims=True)
    retained = np.divide(memory, memory_mass, out=np.zeros_like(memory), where=memory_mass > 0)
    channels = {
        "source_concentration": np.square(distribution).sum(-1),
        "source_entropy": -(distribution * np.log(distribution.clip(1e-30))).sum(-1),
        "read_response_overlap": np.sqrt(distribution * response_distribution).sum(-1),
        "read_memory_overlap": np.sqrt(distribution * retained).sum(-1),
        "previous_read_overlap": np.sqrt(distribution * previous).sum(-1),
        "memory_mass": memory_mass[..., 0],
        "source_response_log_magnitude": np.log1p(total[..., 0]),
        "source_response_coherence": np.divide(np.linalg.norm(response.sum(-2), axis=-1),
            total[..., 0], out=np.zeros_like(total[..., 0]), where=total[..., 0] > 0),
    }
    # Attention has total mass one: current source mass gates this statistical update.
    updated = (1 - mass) * memory + attention
    return channels, updated, distribution


def head_features(rows, source_count, layer_indices):
    """Per physical head: readings, response magnitudes, interactions and source memory."""
    output, columns = [], None
    memory, previous = None, None
    for row in rows:
        read = row["read_mass"][layer_indices].astype(np.float64)
        residual = role_responses(row["response_residual"][layer_indices].astype(float), source_count)
        ffn = role_responses(row["response_ffn"][layer_indices].astype(float), source_count)
        total = residual + ffn
        source_read = read[..., :source_count]
        source_response = row["response_total"][layer_indices, :, :source_count].astype(float)
        if memory is None:
            memory, previous = np.zeros_like(source_read), np.zeros_like(source_read)
        channels, memory, previous = block_state(source_read, source_response, memory, previous)
        role_read = role_responses(read[..., None], source_count)[..., 0]
        for role_index, role in enumerate(ROLES):
            channels[f"read/{role}"] = role_read[..., role_index]
            for name, value in (("total", total), ("residual", residual), ("ffn", ffn)):
                channels[f"log_response/{name}/{role}"] = np.log1p(np.linalg.norm(value[..., role_index, :], axis=-1))
            channels[f"ffn_interaction/{role}"] = cosine(residual[..., role_index, :], ffn[..., role_index, :])
        for role_index, role in enumerate(ROLES[1:], start=1):
            channels[f"source_alignment/{role}"] = cosine(total[..., 0, :], total[..., role_index, :])
        output.append(np.concatenate([value.ravel() for value in channels.values()]))
        if columns is None:
            columns = [f"{name}/L{layer}/H{head}" for name in channels
                       for layer in layer_indices for head in range(read.shape[1])]
    return np.asarray(output, dtype=np.float32), columns
