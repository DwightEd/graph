"""Inspect CHARM states, not the LLM residual stream or causal truth content."""

import numpy as np

from ..graph import degree, node_statistics
from .common import membership


def trace_model(model, graph, block_relay=False, block_sender=False):
    """Block inherited neighbor information, retaining each sender's own depth updates."""
    import torch

    model.eval()
    device = model.in_proj.weight.device
    with torch.no_grad():
        state = model.in_proj(torch.as_tensor(graph["x"], dtype=torch.float32, device=device)).relu()
        isolated = state
        divisor = torch.as_tensor(degree(graph, model.normalization), device=device)[:, None]
        states = [state.cpu().numpy()]
        for layer in model.mp_layers:
            sender = isolated if block_relay else state
            if block_sender:
                sender = torch.zeros_like(sender)
            aggregated = aggregate_messages(layer, sender, graph, model.edge_chunk)
            update = layer.up_mlp(torch.cat((state, aggregated / divisor), dim=-1))
            state = (state + update if layer.residual else update).relu()
            states.append(state.cpu().numpy())
            if block_relay:
                local_update = layer.up_mlp(torch.cat((isolated, torch.zeros_like(isolated)), dim=-1))
                isolated = (isolated + local_update if layer.residual else local_update).relu()
        score = torch.sigmoid(model.pred(state).view(-1)).cpu().numpy()
    prompt = int(graph["prompt_length"])
    return score[prompt:], [state[prompt:] for state in states]


def aggregate_messages(layer, sender, graph, chunk):
    import torch

    result = torch.zeros_like(sender)
    source, target = graph["edge_index"]
    for start in range(0, len(source), chunk):
        stop = start + chunk
        src = torch.as_tensor(source[start:stop], device=sender.device)
        dst = torch.as_tensor(target[start:stop], device=sender.device)
        attr = torch.as_tensor(graph["edge_attr"][start:stop], device=sender.device, dtype=sender.dtype)
        mark = torch.as_tensor(graph["edge_mark"][start:stop], device=sender.device, dtype=sender.dtype)
        message = layer.msg_mlp(torch.cat((sender[src], attr, mark), dim=-1))
        result.index_add_(0, dst, message)
    return result


def retained_entropy(graph, chunk=4096):
    """Entropy conditional on retained edge weights + self diagonal, NOT full entropy."""
    diagonal = np.asarray(graph["x"], dtype=np.float64)
    mass = diagonal.copy()
    weight_log_weight = diagonal * np.log(np.maximum(diagonal, 1e-300))
    target = graph["edge_index"][1]
    for start in range(0, len(target), chunk):
        values = graph["edge_attr"][start:start + chunk].astype(np.float64)
        np.add.at(mass, target[start:start + chunk], values)
        np.add.at(weight_log_weight, target[start:start + chunk], values * np.log(np.maximum(values, 1e-300)))
    entropy = np.log(np.maximum(mass, 1e-300)) - weight_log_weight / np.maximum(mass, 1e-300)
    entropy[mass == 0] = 0.
    prompt = int(graph["prompt_length"])
    return entropy[prompt:].astype(np.float32), mass[prompt:].astype(np.float32)


def head_overlap(graph):
    """Mean pairwise head overlap per layer; no head averaging in the actual model."""
    prompt = int(graph["prompt_length"])
    layers, heads = int(graph["layers"]), int(graph["heads"])
    count = len(graph["x"]) - prompt
    result = np.zeros((count, layers), np.float32)
    source, target = graph["edge_index"]
    order = np.argsort(target, kind="stable")
    bounds = np.searchsorted(target[order], np.arange(prompt, prompt + count + 1))
    for token in range(count):
        ids = order[bounds[token]:bounds[token + 1]]
        values = np.vstack((graph["edge_attr"][ids], graph["x"][prompt + token])).reshape(-1, layers, heads)
        normalized = values / np.maximum(values.sum(axis=0, keepdims=True), 1e-30)
        numerator = np.sum(normalized.sum(axis=2) ** 2 - (normalized ** 2).sum(axis=2), axis=0)
        result[token] = numerator / max(heads * (heads - 1), 1)
    return result


def features(graph, states):
    """Frozen-state diagnostic probes; all feature construction is label-blind."""
    prompt = int(graph["prompt_length"])
    count = len(graph["x"]) - prompt
    stats = node_statistics(graph)
    entropy, mass = retained_entropy(graph)
    position = np.stack((np.arange(count), np.arange(count) / max(count - 1, 1)), axis=1).astype(np.float32)
    marginal = np.stack((stats["in_rp"], stats["in_rr"], stats["local_rr_fraction"],
                         stats["self_attention_mean"], entropy.mean(axis=1), entropy.std(axis=1),
                         mass.mean(axis=1)), axis=1).astype(np.float32)
    node = graph["x"][prompt:]
    result = dict(position=position, retained_entropy=entropy.mean(axis=1, keepdims=True),
                  head_marginals=np.column_stack((mass, entropy)).astype(np.float32),
                  node=node, local_causal=marginal,
                  local_with_future_degree=np.column_stack((marginal, stats["out_degree"])).astype(np.float32),
                  node_local=np.column_stack((node, marginal)).astype(np.float32),
                  head_overlap=head_overlap(graph), projected=states[0])
    result.update({"layer_" + str(index): state for index, state in enumerate(states[1:], 1)})
    return result


def relation_rows(sample, graph, feature_values, token_ids, max_lag=32):
    """Compare pairs at EXACTLY the same answer and token distance."""
    prompt = int(graph["prompt_length"])
    count = len(sample["gold"])
    member = membership(sample)
    response_ids = np.asarray(token_ids)[prompt:]
    gold = sample["gold"].astype(bool)
    source, target = graph["edge_index"]
    keep = source >= prompt
    edge_keys = np.sort((target[keep] - prompt) * count + source[keep] - prompt)
    selected_features = {k: v for k, v in feature_values.items() if k in ("node", "projected") or k.startswith("layer_")}
    normalized = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for k, v in selected_features.items()}
    for lag in range(1, min(max_lag + 1, count)):
        left, right = np.arange(count - lag), np.arange(lag, count)
        connected = np.isin(right * count + left, edge_keys)
        groups = dict(same_span=gold[left] & gold[right] & (member[left] == member[right]),
                      different_error_spans=gold[left] & gold[right] & (member[left] != member[right]),
                      normal_normal=~gold[left] & ~gold[right], mixed=gold[left] != gold[right])
        for relation, selected in groups.items():
            for edge in (False, True):
                for repeat in (False, True):
                    mask = selected & (connected == edge) & ((response_ids[left] == response_ids[right]) == repeat)
                    if not mask.any():
                        continue
                    row = dict(id=sample["id"], source_id=sample["source_id"], lag=lag, relation=relation,
                               connected=edge, same_token_id=repeat, pairs=int(mask.sum()),
                               mean_score_gap=float(np.mean(abs(sample["score"][left[mask]] - sample["score"][right[mask]]))))
                    for name, values in normalized.items():
                        row["cosine_" + name] = float(np.mean(np.sum(values[left[mask]] * values[right[mask]], axis=1)))
                    yield row


def matched_relation_contrasts(rows):
    """No cross-answer or cross-distance substitution when matches are missing."""
    grouped = {}
    for row in rows:
        key = (row["id"], row["lag"], row["connected"], row["same_token_id"])
        grouped.setdefault(key, {})[row["relation"]] = row
    result = []
    for (identity, lag, connected, same_token), group in grouped.items():
        if "same_span" not in group or "different_error_spans" not in group:
            continue
        same, different = group["same_span"], group["different_error_spans"]
        row = dict(id=identity, source_id=same["source_id"], lag=lag, connected=connected, same_token_id=same_token,
                   same_span_pairs=same["pairs"], different_span_pairs=different["pairs"])
        for name in same:
            if name.startswith("cosine_") or name == "mean_score_gap":
                row[name + "_same_minus_different"] = same[name] - different[name]
        result.append(row)
    return result
