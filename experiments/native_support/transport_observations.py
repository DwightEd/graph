"""Read source-resolved native responses without learned compression or labels."""

import numpy as np
from state_audit.storage import read_arrays, read_json

from .routes import route_scores


def routing_observations(current, prompt, source_count, source_mask):
    """Preserve the original route numerator and all-edge norm denominator."""
    positions = np.arange(len(current["group_ids"]))
    special = current["group_ids"] == source_count + 3
    groups = (positions >= prompt).astype(int)
    groups[special] = 2
    source = (positions < prompt) & ~special
    history = (positions >= prompt) & ~special
    if source_mask is not None:
        source[:prompt] = source_mask
        source[-1] = False
        history = positions >= prompt

    route_groups = np.full(len(positions), source_count + 2, dtype=int)
    route_groups[history] = source_count + 1
    route_groups[source] = source_count  # Source positions outside automatic blocks.
    resolved = source & (current["group_ids"] < source_count)
    route_groups[resolved] = current["group_ids"][resolved]
    masks = np.eye(source_count + 3, dtype=np.float32)[route_groups]
    magnitude = np.sqrt(current["edge_value_energy"])
    metrics = route_scores(current["attention"], magnitude, groups, prompt, source_mask)
    prefix = "prompt_" if source_mask is None else ""
    return {"routing_budget": magnitude @ masks,
            "routing_attention": current["attention"] @ masks,
            "raw_routing": np.asarray(metrics[prefix + "routing_imbalance"]),
            "raw_attention": np.asarray(metrics[prefix + "attention_displacement"])}


def observe_token(current, prompt, source_count, rank, source_mask=None):
    """Keep layer/head/group axes; choice coordinates belong to this query only."""
    masks = np.eye(source_count + 4, dtype=np.float32)[current["group_ids"]]
    message_size = np.sqrt(current["edge_value_energy"])
    choice = current["group_effect"][..., rank:]
    ffn_choice = current["ffn_effect"][..., rank:]
    observed = {
        "group_effect": current["group_effect"][..., :rank],
        "choice_effect": choice,
        "choice_contrast": choice[..., :1] - choice[..., 1:],
        "group_attention": current["group_attention"],
        "message_budget": message_size @ masks,
        "ffn_effect": current["ffn_effect"],
        "ffn_state": current["ffn_effect"][..., :rank],
        "ffn_choice_contrast": ffn_choice[..., :1] - ffn_choice[..., 1:],
        # Answer key P+j represents target j, while its native state is row j+1.
        "history_attention": current["attention"][..., prompt:],
    }
    for name in ("state", "candidate_ids", "candidate_logits", "entropy", "surprisal",
                 "candidate_tail_mass", "query", "target", "token_id"):
        observed[name] = np.asarray(current[name])
    observed.update(routing_observations(current, prompt, source_count, source_mask))
    return observed


def describe_sources(response, blocks):
    """Attach original text to automatic blocks, without inventing semantic roles."""
    return [
        {"id": index, "positions": positions,
         "text": "".join(response["token_text"][position] for position in positions),
         "semantic_type": "unassigned"}
        for index, positions in enumerate(blocks)
    ]


def read_token(response, directory, target, prompt_groups, source_count, special_ids):
    """Check the cache's token and group alignment once at the input boundary."""
    current = read_arrays(directory / f"token_{target:06d}.npz")
    prompt = response["prompt_length"]
    query = prompt + target - 1
    token_id = response["token_ids"][prompt + target]
    if (int(current["query"]) != query or int(current["target"]) != target
            or int(current["token_id"]) != token_id
            or int(current["candidate_ids"][0]) != token_id):
        raise ValueError(f"{response['id']}: transport token alignment mismatch at {target}")

    groups = np.full(query + 1, source_count, dtype=np.int64)
    groups[:prompt] = prompt_groups
    groups[query] = source_count + 1
    groups[np.isin(response["token_ids"][:query + 1], special_ids)] = source_count + 3
    if not np.array_equal(groups, current["group_ids"]):
        raise ValueError(f"{response['id']}: transport source alignment mismatch at {target}")
    return current


def load_token_rows(response, directory, rank, sources, special_ids, source_mask):
    """Load each potentially large raw attention array once, then release it."""
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    source_count = len(sources["blocks"])
    rows = []
    for target in range(count):
        current = read_token(response, directory, target, sources["prompt_groups"],
                             source_count, special_ids)
        if len(current["state"]) != rank:
            raise ValueError(f"{response['id']}: rank does not match native capture")
        row = observe_token(current, prompt, source_count, rank, source_mask)
        if target == 0:
            shape = (count, *current["attention"].shape[:2])
            history = np.zeros((*shape, count), dtype=np.float32)
            future_response = np.zeros(shape, dtype=np.float32)
        history[target, ..., :target] = row.pop("history_attention")
        # Target j's key is P+j; only rows t>j+1 are strictly later queries.
        prior_count = max(target - 1, 0)
        response_size = np.sqrt(current["edge_response_energy"])[..., prompt:-1]
        future_response[:prior_count] += response_size.transpose(2, 0, 1)
        rows.append(row)
    arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    arrays["history_attention"] = history
    return arrays, future_response


def add_observation_masks(arrays, future_response):
    """Missing first differences and future windows stay separate from features."""
    count = len(arrays["target"])
    read_change = np.full_like(arrays["group_attention"], np.nan)
    read_change[1:] = np.diff(arrays["group_attention"], axis=0)
    arrays["read_change"] = read_change
    arrays["read_change_observed"] = np.arange(count) > 0

    query_index = np.arange(count)[:, None]
    key_index = np.arange(count)[None, :]
    future_window = query_index > key_index + 1
    future_count = future_window.sum(axis=0)
    future_attention = np.einsum("tlhj,tj->jlh", arrays["history_attention"], future_window)
    divisor = future_count[:, None, None]
    for name, total in (("future_attention_mean", future_attention),
                        ("future_response_mean", future_response)):
        mean = np.full_like(total, np.nan)
        np.divide(total, divisor, out=mean, where=divisor > 0)
        arrays[name] = mean
    arrays["future_query_count"] = future_count
    arrays["future_observed"] = future_count > 0


def load_observations(response, directory, rank, special_ids, source_mask=None):
    """Return arrays and readable source rows; the caller owns saving outputs."""
    sources = read_json(directory / "sources.json")
    arrays, future_response = load_token_rows(response, directory, rank, sources,
                                             special_ids, source_mask)
    add_observation_masks(arrays, future_response)
    arrays["source_count"] = np.asarray(len(sources["blocks"]))
    arrays["routing_source_count"] = np.asarray(len(sources["blocks"]) + 1)
    return arrays, describe_sources(response, sources["blocks"])
