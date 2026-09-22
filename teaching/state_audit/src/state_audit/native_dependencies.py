"""Local dependencies at unchanged activations; neither ablations nor path flows."""

import numpy as np
import torch

from .capture import numpy


def sender_sites(records, receiver_layer, include_current_heads):
    sites = []
    for layer, record in records.items():
        if layer < receiver_layer or (layer == receiver_layer and include_current_heads):
            sites.append((layer, "head_readout", record["head_readout"]))
        if layer < receiver_layer:
            sites.append((layer, "mlp_write", record["mlp_write"]))
    return sites


def sender_sensitivity(objective, sites, records, group_readouts, shape, *, retain_graph=True):
    """The source partition lives in head-value space, before its W_O projection."""
    layers, heads, groups = shape
    head_effect = np.zeros((layers, heads, groups), dtype=np.float32)
    mlp_effect = np.zeros(layers, dtype=np.float32)
    head_observed = np.zeros((layers, heads), dtype=bool)
    mlp_observed = np.zeros(layers, dtype=bool)
    if sites:
        gradients = torch.autograd.grad(
            objective, [site[2] for site in sites], retain_graph=retain_graph
        )
        for (layer, name, _), gradient in zip(sites, gradients):
            if name == "head_readout":
                head_effect[layer] = np.einsum(
                    "hd,hgd->hg", numpy(gradient[-1]), group_readouts[layer]
                )
                head_observed[layer] = True
            else:
                mlp_effect[layer] = float((gradient[-1] * records[layer][name][-1]).detach().sum())
                mlp_observed[layer] = True
    return head_effect, mlp_effect, head_observed, mlp_observed


def select_route_receivers(group_mass, positive_groups, negative_groups, budget):
    """One head per layer, ranked by unsigned reviewed-role exposure, not output effects."""
    selected_groups = list(positive_groups) + list(negative_groups)
    mass = group_mass[..., selected_groups].sum(-1)
    candidates = [
        (float(mass[layer, head]), layer, int(head))
        for layer, head in enumerate(mass.argmax(-1))
        if layer > 0
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(layer, head) for exposure, layer, head in candidates[:budget] if exposure > 0]


def dependency_objectives(records, arrays, masks, positive_groups, negative_groups, budget):
    receivers = select_route_receivers(
        arrays["group_route_mass"], positive_groups, negative_groups, budget
    )
    route_direction = masks[:, list(positive_groups)].sum(-1)
    route_direction -= masks[:, list(negative_groups)].sum(-1)
    for layer, head in receivers:
        objective = records[layer]["attention"][head, -1].float() @ route_direction
        yield "route_mass_difference", layer, head, objective, False
    direction = torch.as_tensor(arrays["direction"], device=masks.device)
    for layer, record in records.items():
        objective = record["mlp_write"][-1].float() @ direction
        yield "mlp_logit_write", layer, -1, objective, True


def capture_dependencies(records, arrays, masks, positive_groups, negative_groups, budget):
    shape = arrays["group_route_mass"].shape
    names, layers, heads, values, effects = [], [], [], [], []
    objectives = dependency_objectives(
        records, arrays, masks, positive_groups, negative_groups, budget
    )
    for name, layer, head, objective, current_heads in objectives:
        sites = sender_sites(records, layer, current_heads)
        effects.append(
            sender_sensitivity(objective, sites, records, arrays["group_readouts"], shape)
        )
        names.append(name)
        layers.append(layer)
        heads.append(head)
        values.append(float(objective.detach()))
    fields = (
        "dependency_head_effect",
        "dependency_mlp_effect",
        "dependency_head_observed",
        "dependency_mlp_observed",
    )
    result = {name: np.stack([item[i] for item in effects]) for i, name in enumerate(fields)}
    result.update(
        receiver_kind=np.asarray(names),
        receiver_layer=np.asarray(layers),
        receiver_head=np.asarray(heads),
        receiver_value=np.asarray(values),
    )
    return result
