from types import SimpleNamespace

import torch


def routing_state():
    response_idx = 2
    tokens = 3
    layers = 3
    heads = 2
    events = []
    for layer in range(layers):
        for head in range(heads):
            events.extend(
                [
                    (layer, head, 0, 0, 0.65),
                    (layer, head, 1, 1, 0.30),
                    (layer, head, 1, 2, 0.35),
                    (layer, head, 2, 1, 0.20),
                    (layer, head, 2, 3, 0.45),
                ]
            )
    layer = torch.tensor([row[0] for row in events], dtype=torch.long)
    head = torch.tensor([row[1] for row in events], dtype=torch.long)
    query = torch.tensor([row[2] for row in events], dtype=torch.long)
    source = torch.tensor([row[3] for row in events], dtype=torch.long)
    weight = torch.tensor([row[4] for row in events], dtype=torch.float32)
    self_mass = torch.full((tokens, layers, heads), 0.20)
    prompt_mass = torch.zeros(tokens, layers, heads)
    response_mass = torch.zeros_like(prompt_mass)
    for current_layer, current_head, current_query, current_source, current_weight in events:
        if current_source < response_idx:
            prompt_mass[current_query, current_layer, current_head] += current_weight
        else:
            response_mass[current_query, current_layer, current_head] += current_weight
    unresolved = 1.0 - prompt_mass - response_mass - self_mass
    role = torch.stack((prompt_mass, response_mass, self_mass, unresolved), dim=-1)
    edges = SimpleNamespace(
        num_layers=layers,
        num_heads=heads,
        num_response_tokens=tokens,
        num_tokens=response_idx + tokens,
        response_idx=response_idx,
        layer=layer,
        head=head,
        query=query,
        source=source,
        weight=weight,
        device=weight.device,
    )
    return SimpleNamespace(
        edges=edges,
        edge_weight=weight,
        prompt_mass=prompt_mass,
        response_mass=response_mass,
        self_mass=self_mass,
        unresolved_mass=unresolved,
        role_probability=role,
    )
