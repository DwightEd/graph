"""Frozen CHARM activation interventions on one interval. No new features."""

import numpy as np
import torch

from .model import degree


def factual_states(model, graph):
    """Exact original computation; store K+1 small node-state matrices only."""
    device = model.in_proj.weight.device
    values = torch.as_tensor(graph['x'], dtype=torch.float32, device=device)
    state = model.in_proj(values).relu()
    divisor = torch.as_tensor(degree(graph, 'in'), device=device)[:, None]
    states = [state]
    for layer in model.mp_layers:
        messages = layer.aggregate(state, graph, model.edge_chunk, 'full') / divisor
        state = layer.update(state, messages)
        states.append(state)
    return states


def interval_view(graph, start, end):
    """All incoming edges to [start,end), not just an induced internal subgraph."""
    source, target = graph['edge_index']
    selected = (target >= start) & (target < end)
    view = dict(edge_index=graph['edge_index'][:, selected],
                edge_attr=graph['edge_attr'][selected],
                edge_mark=graph['edge_mark'][selected])
    view['start'] = start
    view['end'] = end
    view['internal'] = view['edge_index'][0] >= start
    view['divisor'] = degree(graph, 'in')[start:end]
    view['prompt'] = int(graph['prompt_length'])
    return view


def aggregate_groups(layer, sender, view, chunk, muted):
    """Zero entire selected messages, including MLP bias; retain original degree."""
    length = view['end'] - view['start']
    total = sender.new_zeros((length, sender.shape[1]))
    internal = torch.zeros_like(total)
    external = torch.zeros_like(total)
    for left in range(0, len(view['internal']), chunk):
        right = left + chunk
        message = layer.messages(sender, view, left, right, 'full')
        keep = torch.as_tensor(~muted[left:right], device=sender.device)
        inside = torch.as_tensor(view['internal'][left:right], device=sender.device)
        target = torch.as_tensor(view['edge_index'][1, left:right] - view['start'], device=sender.device)
        message = message * keep[:, None]
        total.index_add_(0, target, message)
        internal.index_add_(0, target, message * inside[:, None])
        external.index_add_(0, target, message * (~inside)[:, None])
    divisor = torch.as_tensor(view['divisor'], device=sender.device)[:, None]
    return total / divisor, internal / divisor, external / divisor


def readout(model, state):
    """An exact signed decomposition of the final logit, not saliency."""
    hidden = model.pred[1](model.pred[0](state))
    contribution = hidden * model.pred[3].weight[0]
    logits = model.pred(state).view(-1)
    reconstructed = contribution.sum(dim=-1) + model.pred[3].bias[0]
    torch.testing.assert_close(logits, reconstructed, atol=2e-5, rtol=1e-5)
    return dict(logits=logits, contribution=contribution)


def update_trace(layer, state, total):
    joined = torch.cat((state, total), dim=-1)
    gate = layer.up_mlp[0](joined) > 0
    updated = layer.update(state, total)
    return updated, gate


def span_forward(model, base, view, muted=None, pulse=None, reference=None):
    """Pruned computation is exact on a causal interval; earlier sources stay factual.

    reference is the cut-internal trajectory: internal senders cannot retransmit
    state acquired via earlier same-span messages. Their external context remains.
    """
    start, end = view['start'], view['end']
    state = base[0][start:end]
    trajectory = [state]
    empty = np.zeros(len(view['internal']), bool)
    selected = empty if muted is None else muted
    for step, layer in enumerate(model.mp_layers):
        sender = base[step].clone()
        sender[start:end] = state if reference is None else reference['states'][step]
        active = selected if pulse is None or step == pulse else empty
        total, internal, external = aggregate_groups(layer, sender, view, model.edge_chunk, active)
        own = state
        state, gate = update_trace(layer, own, total)
        trajectory.append(state)
    return dict(readout(model, state), states=trajectory, own=own, gate=gate,
                total=total, internal=internal, external=external)


def last_swap(model, recipient, donor, branch):
    """Swap factual matched activations only at the last update; no future mixing."""
    own = recipient['own']
    total = recipient['total']
    if branch == 'own':
        own = donor['own']
    else:
        total = total - recipient[branch] + donor[branch]
    state, gate = update_trace(model.mp_layers[-1], own, total)
    return dict(readout(model, state), gate=gate)


def random_internal_mask(view, token_ids, seed):
    """Same target/lag-band/copy counts. Nonexchangeable strata remain disclosed."""
    source, target = view['edge_index']
    history = source >= view['prompt']
    indices = np.flatnonzero(history)
    distance = np.ceil(np.log2(target[indices] - source[indices])).astype(int)
    copy = token_ids[source[indices]] == token_ids[target[indices]]
    strata = np.column_stack((target[indices], distance, copy))
    _, labels = np.unique(strata, axis=0, return_inverse=True)
    order = np.argsort(labels, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    mask = np.zeros(len(source), bool)
    exchangeable = np.zeros(view['end'] - view['start'], int)
    rng = np.random.default_rng(seed)
    for group in groups:
        edges = indices[group]
        number = int(view['internal'][edges].sum())
        if not number:
            continue
        mask[rng.choice(edges, number, replace=False)] = True
        if number < len(edges):
            exchangeable[target[edges[0]] - view['start']] += number
    return mask, exchangeable


def mask_counts(view, mask):
    source, target = view['edge_index']
    baseline = view['internal']
    return dict(edges=int(mask.sum()), internal_edges=int(baseline.sum()),
                changed_slots=int(np.count_nonzero(mask != baseline)),
                overlap=int((mask & baseline).sum()),
                retained_weight_sum=float(view['edge_attr'][mask].sum()),
                mean_lag=float((target[mask] - source[mask]).mean()) if mask.any() else None)


def intervene_span(model, graph, base, start, end, token_ids, repeats, seed):
    view = interval_view(graph, start, end)
    inner = view['internal']
    runs = {'full': span_forward(model, base, view)}
    runs['cut_internal'] = span_forward(model, base, view, inner)
    runs['cut_external'] = span_forward(model, base, view, ~inner)
    runs['cut_all'] = span_forward(model, base, view, np.ones(len(inner), bool))
    runs['no_reuse'] = span_forward(model, base, view, reference=runs['cut_internal'])
    for step in range(len(model.mp_layers)):
        runs[f'pulse_{step}'] = span_forward(model, base, view, inner, pulse=step)
    controls = [dict(experiment='cut_internal', **mask_counts(view, inner))]
    exchangeable = np.zeros(end - start, int)
    for repeat in range(repeats):
        mask, exchangeable = random_internal_mask(view, token_ids, seed + repeat)
        name = f'random_{repeat}'
        runs[name] = span_forward(model, base, view, mask)
        controls.append(dict(experiment=name, **mask_counts(view, mask)))
    return runs, controls, exchangeable


def pair_forward(model, graph, sample, base, pair, repeats, seed):
    """Two separate counterfactual worlds, then symmetric final-cell exchanges."""
    runs, controls, exchangeable = [], [], []
    for role, key in enumerate(('error_start', 'normal_start')):
        start = int(graph['prompt_length']) + pair[key]
        side, counts, eligible = intervene_span(model, graph, base, start, start + pair['length'],
                                                sample['token_ids'], repeats, seed)
        for row in counts:
            row['side'] = role
        runs.append(side)
        controls.extend(counts)
        exchangeable.append(eligible)
    for role in range(2):
        for branch in ('internal', 'external', 'own'):
            runs[role]['last_swap_' + branch] = last_swap(
                model, runs[role]['full'], runs[1 - role]['full'], branch)
    names = list(runs[0])
    logits, contributions, gates = [], [], []
    for side in runs:
        logits.append(torch.stack([side[name]['logits'] for name in names]).cpu().numpy())
        contributions.append(torch.stack([side[name]['contribution'] for name in names]).cpu().numpy())
        gates.append(torch.stack([(side[name]['gate'] != side['full']['gate']).float().mean(-1)
                                  for name in names]).cpu().numpy())
    return dict(names=np.asarray(names), logits=np.stack(logits), contributions=np.stack(contributions),
                gate_flips=np.stack(gates), exchangeable=np.stack(exchangeable),
                readout_bias=model.pred[3].bias.detach().cpu().numpy()), controls
