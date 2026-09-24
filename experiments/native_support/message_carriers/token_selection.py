"""Source-conditioned edge derivatives on shared, query-resolved history candidates."""

import numpy as np
import torch

from state_audit.capture import numpy
from state_audit.attribution import _projection_gram
from state_audit.functional_capture import attention_rows


def receiver_candidates(observations, layer, heads, width, target, budget):
    """Always include current receiver; screen earlier writes by gradient/state norms.

    Screening is an explicit budget, not a complete enumeration of causal paths.
    Its union is shared between source conditions before comparing edge derivatives.
    """
    chosen = [{target - 1} if target else set() for _ in range(heads)]
    for condition in observations:
        record, prompt = condition["layers"][layer], condition["prompt"]
        gradient = record["gradient"][0, prompt:].float().reshape(target, heads, width)
        output = record["head"][0, prompt:].float().reshape(target, heads, width)
        relevance = numpy(gradient.norm(dim=-1) * output.norm(dim=-1))
        for head in range(heads):
            order = np.argsort(-relevance[:-1, head], kind="stable")[:budget - 1]
            chosen[head].update(int(query) for query in order if relevance[query, head] > 0)
    return [sorted(queries) for queries in chosen]


@torch.no_grad()
def measure_layer_edges(model, layer, observation, receivers, target):
    """Return signed gate derivatives, route mass and native projected message energy."""
    device = model.native.device
    record = {name: None if value is None else value.to(device)
              for name, value in observation["layers"][layer].items()}
    positions = sorted({query for selected in receivers for query in selected})
    queries = torch.tensor(positions, device=device) + observation["prompt"]
    rotary = tuple(value.to(device) for value in observation["rotary"])
    heads, kv_heads, width = model.head_layout(layer)
    attention = attention_rows(model, layer, record, rotary, queries).to(record["value"].dtype).float()
    values = record["value"][0].float().reshape(-1, kv_heads, width).repeat_interleave(heads // kv_heads, 1)
    gradient = record["gradient"][0, queries].float().reshape(-1, heads, width)
    keys = torch.arange(target, device=device) + observation["prompt"]
    pairs = [(head, positions.index(query), query) for head, selected in enumerate(receivers) for query in selected]
    head_ids = torch.tensor([pair[0] for pair in pairs], device=device)
    row_ids = torch.tensor([pair[1] for pair in pairs], device=device)
    mass = attention[row_ids, head_ids][:, keys]
    derivative = torch.einsum("nkd,nd->nk", values[keys][:, head_ids].transpose(0, 1), gradient[row_ids, head_ids]) * mass
    norms = torch.einsum("khd,hde,khe->kh", values[keys], _projection_gram(model, layer), values[keys])
    energy = mass.square() * norms[:, head_ids].T
    rows = [(layer, head, query, key) for head, _, query in pairs for key in range(target)]
    reconstructed = torch.einsum("qhk,khd->qhd", attention, values)
    error = (reconstructed - record["head"][0, queries].float().reshape(-1, heads, width)).abs().max()
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4), numpy(derivative).ravel(), numpy(mass).ravel(), numpy(energy).ravel(), float(error)


def candidate_table(model, observations, target, receiver_budget):
    identities, gradients, masses, energies, errors = [], [], [], [], []
    for layer in range(len(model.layers)):
        heads, _, width = model.head_layout(layer)
        receivers = receiver_candidates(observations, layer, heads, width, target, receiver_budget)
        measured = [measure_layer_edges(model, layer, condition, receivers, target) for condition in observations]
        identities.append(measured[0][0])
        gradients.append(np.stack([value[1] for value in measured], axis=1))
        masses.append(np.stack([value[2] for value in measured], axis=1))
        energies.append(np.stack([value[3] for value in measured], axis=1))
        errors.append([value[4] for value in measured])
    return dict(edges=np.concatenate(identities), gradients=np.concatenate(gradients),
                route_mass=np.concatenate(masses), message_energy=np.concatenate(energies),
                reconstruction_error=np.asarray(errors).T)


def select_edges(table, top_k, seed, selection):
    derivatives = table["gradients"]
    if not np.isfinite(derivatives).all():
        raise ValueError("Nonfinite token-choice derivatives")
    importance = (abs(derivatives[:, 0] - derivatives[:, 1]) if selection == "conditional"
                  else abs(derivatives).max(1))
    eligible = table["route_mass"].max(1) > 0
    order = np.argsort(-importance, kind="stable")
    chosen, used = [], set()
    for index in order:
        head = tuple(table["edges"][index, :2])
        if eligible[index] and head not in used:
            chosen.append(int(index))
            used.add(head)
            if len(chosen) == top_k:
                break
    random, sham = np.random.default_rng(seed), []
    for index in chosen:
        # Match physical head AND receiver, varying the history key only.
        same = np.all(table["edges"][:, :3] == table["edges"][index, :3], axis=1)
        alternatives = np.flatnonzero(same & eligible & (np.arange(len(eligible)) != index))
        sham.append(int(random.choice(alternatives)) if len(alternatives) else index)
    return dict(edges=table["edges"][chosen], sham_edges=table["edges"][sham],
        approximation=derivatives[chosen], sham_approximation=derivatives[sham],
        route_mass=table["route_mass"][chosen], message_energy=table["message_energy"][chosen],
        sham_route_mass=table["route_mass"][sham], sham_message_energy=table["message_energy"][sham],
        importance=importance[chosen], candidate_count=int(eligible.sum()),
        reconstruction_error=table["reconstruction_error"], reconstruction_measured=True)
