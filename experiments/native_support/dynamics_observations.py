"""Head-resolved read/response profiles and offline reuse, without annotations."""

import numpy as np
from scipy.special import xlogy
from state_audit.storage import read_arrays, write_arrays

from .dynamics_core import effect_source_moments


def partition_prompt(response, source_mask, special_ids, block_tokens):
    """Text-boundary blocks with a token cap; no claim or annotation boundaries."""
    prompt = response["prompt_length"]
    ordinary = ~np.isin(response["token_ids"][:prompt], special_ids)
    evidence = ordinary if source_mask is None else ordinary & source_mask
    block_ids = np.full(prompt, -1, dtype=np.int64)
    blocks, current = [], []
    for position in range(prompt):
        if current and (not evidence[position] or position != current[-1] + 1):
            blocks.append(current)
            current = []
        if evidence[position]:
            current.append(position)
            piece = response["token_text"][position].strip()
            if len(current) >= block_tokens or (len(current) >= 8 and piece.endswith((".", "?", "!"))):
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{response['id']}: no ordinary source positions")
    for index, positions in enumerate(blocks):
        block_ids[positions] = index
    block_ids[block_ids < 0] = len(blocks) + 2
    block_ids[~ordinary] = len(blocks) + 3
    return block_ids, blocks


def distribution(mass):
    total = mass.sum(-1)
    probability = mass / np.maximum(total[..., None], 1e-12)
    return probability, total, -xlogy(probability, probability).sum(-1)


def head_profile(current, previous, sources, rank):
    """Layer/head axes survive; only within-head source distributions are reduced."""
    read, read_total, read_entropy = distribution(current["group_attention"][..., :sources])
    effects = current["group_effect"][..., :sources, :rank]
    size = effects.shape
    moments = effect_source_moments(effects.reshape(-1, sources, rank))
    effect_mass = np.linalg.norm(effects, axis=-1)
    _probability, effect_total, effect_entropy = distribution(effect_mass)
    coherent = np.linalg.norm(effects.sum(-2), axis=-1) / np.maximum(effect_total, 1e-12)
    covariance = np.nan_to_num(moments["covariance"]).reshape(*size[:2], rank, rank)
    source_variance = np.trace(covariance, axis1=-2, axis2=-1)
    previous_read = read if previous is None else distribution(previous["group_attention"][..., :sources])[0]
    mixture = .5 * (read + previous_read)
    js = (-xlogy(mixture, mixture).sum(-1)
          + .5 * xlogy(read, read).sum(-1) + .5 * xlogy(previous_read, previous_read).sum(-1))
    mass = current["group_attention"]
    history_effect = np.linalg.norm(current["group_effect"][..., sources, :rank], axis=-1)
    values = [read_total, mass[..., sources], mass[..., sources + 1], mass[..., sources + 2],
              mass[..., sources + 3], read_entropy, read.max(-1), js,
              np.zeros_like(read_total) if previous is None else read_total - previous["group_attention"][..., :sources].sum(-1),
              np.log1p(effect_total), effect_entropy, coherent, source_variance,
              np.log1p(history_effect), (effect_total > 0).astype(float)]
    choice = current["group_effect"][..., :sources, rank:]
    # Choice slots are local to a token; these summaries do not treat slot 1 as a fixed semantic axis.
    values.extend([np.log1p(np.maximum(choice[..., 0], 0).sum(-1)),
                   np.log1p(np.maximum(-choice[..., 0], 0).sum(-1)),
                   np.log1p(np.linalg.norm(choice, axis=-1).sum(-1))])
    denominator = np.log(max(sources, 2))
    values.extend([read_entropy / denominator, effect_entropy / denominator,
                   (read_total > 0).astype(float), np.full_like(read_total, previous is not None)])
    source_keys = current["group_ids"] < sources
    edge_strength = np.sqrt(current["edge_response_energy"])[..., source_keys].sum(-1)
    values.extend([np.log1p(edge_strength), effect_total / np.maximum(edge_strength, 1e-12)])
    return np.stack(values, -1).astype(np.float32)


PROFILE_CHANNELS = (
    "source_read", "history_read", "self_read", "other_read", "special_read",
    "source_read_entropy", "source_max_share", "source_distribution_change", "source_read_change",
    "log_source_response", "source_response_entropy", "source_direction_coherence",
    "source_direction_variance", "log_history_response", "source_response_observed",
    "log_observed_choice_positive", "log_observed_choice_negative", "log_choice_response",
    "source_read_entropy_fraction", "source_response_entropy_fraction", "source_read_observed",
    "previous_read_observed",
    "log_source_edge_response", "within_block_direction_coherence",
    "source_log_distance", "future_mean_attention", "future_mean_response", "future_observed",
)


def build_observations(response, directory, sources, rank, source_mask, special_ids):
    from .routes import route_scores

    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    profiles, states, inputs, histories, ffns, rows = [], [], [], [], [], []
    future_attention, future_response = None, None
    previous = None
    for target in range(count):
        current = read_arrays(directory / f"token_{target:06d}.npz")
        if int(current["query"]) != prompt + target - 1 or int(current["token_id"]) != response["token_ids"][prompt + target]:
            raise ValueError(f"{response['id']}: dynamics token alignment mismatch")
        attention = current["attention"]
        if future_attention is None:
            future_attention = np.zeros((count, *attention.shape[:2]), dtype=np.float32)
            future_response = np.zeros_like(future_attention)
        # Target j occupies KEY P+j; strictly later query is t>j+1. Exclude self-read.
        previous_keys = max(target - 1, 0)
        future_attention[:previous_keys] += attention[..., prompt:-1].transpose(2, 0, 1)
        future_response[:previous_keys] += np.sqrt(current["edge_response_energy"][..., prompt:-1]).transpose(2, 0, 1)
        profile = head_profile(current, previous, sources, rank)
        selected = current["group_ids"] < sources
        distance = np.log1p(int(current["query"]) - np.arange(attention.shape[-1]))
        remote = (attention * selected * distance).sum(-1)
        profiles.append(np.concatenate((profile, remote[..., None]), -1))
        history_state = np.zeros((*attention.shape[:2], rank), dtype=np.float32)
        if target > 1:
            history_state = np.einsum("lhj,jr->lhr", attention[..., prompt:-1], np.asarray(states[1:]))
        histories.append(history_state.reshape(-1))
        states.append(current["state"])
        inputs.append(current["group_effect"][..., :sources, :rank].sum(-2).reshape(-1))
        ffns.append(current["ffn_effect"].reshape(-1))
        groups = np.zeros(attention.shape[-1], dtype=int)
        groups[prompt:] = 1
        groups[np.isin(response["token_ids"][:len(groups)], special_ids)] = 2
        metrics = route_scores(attention, np.sqrt(current["edge_value_energy"]), groups, prompt, source_mask)
        rows.append({**metrics, "entropy": float(current["entropy"]), "surprisal": float(current["surprisal"]),
                     "candidate_tail_mass": float(current["candidate_tail_mass"])})
        previous = current
    observed = np.maximum(count - np.arange(count) - 2, 0)
    divisor = np.maximum(observed, 1)[:, None, None]
    future = np.stack((future_attention / divisor, future_response / divisor,
                       np.broadcast_to((observed > 0)[:, None, None], future_attention.shape)), -1)
    profiles = np.concatenate((np.stack(profiles), future), -1)
    arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    arrays.update(profile=profiles, state=np.stack(states), source_input=np.stack(inputs), history=np.stack(histories),
                  ffn=np.stack(ffns), future_query_count=observed,
                  profile_channels=np.asarray(PROFILE_CHANNELS), token_id=np.asarray(response["token_ids"][prompt:]),
                  query=np.arange(prompt - 1, prompt + count - 1), target=np.arange(count))
    write_arrays(directory / "observations.npz", **arrays)
    return arrays


def project_source_covariance(directory, count, sources, rank, components, scale):
    """Project blocks jointly across heads, retaining same-source cross-head terms."""
    results = []
    weights = (components / scale).reshape(len(components), -1, rank)
    for target in range(count):
        raw = read_arrays(directory / f"token_{target:06d}.npz")["group_effect"]
        effects = raw[..., :sources, :rank].reshape(-1, sources, rank)
        blocks = np.einsum("uhr,hbr->bu", weights, effects)
        moments = effect_source_moments(blocks[None])
        covariance = np.nan_to_num(moments["covariance"][0]) * moments["effect_scale"][0] ** 2
        results.append(covariance)
    return np.asarray(results, dtype=np.float32)
