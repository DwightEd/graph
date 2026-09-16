"""Operate on the trained model itself; do not reimplement its forward formula."""

import numpy as np
import torch


@torch.no_grad()
def predict(model, graph):
    model.eval()
    logits = model(graph)
    prompt = int(graph["prompt_length"])
    return logits[prompt:].cpu().numpy()


@torch.no_grad()
def capture_states(model, graph):
    """Read actual inputs of each GNN layer. All hooks are removed on exit."""
    states = []
    handles = []
    def capture(module, arguments):
        states.append(arguments[0].detach().cpu().numpy().copy())
    try:
        for layer in model.mp_layers:
            handles.append(layer.register_forward_pre_hook(capture))
        logits = predict(model, graph)
    finally:
        for handle in handles:
            handle.remove()
    return logits, states


class MessageInputPatch:
    """Replace only source-state coordinates as native edge chunks enter msg_mlp."""

    def __init__(self, states, graph, groups, donors=None):
        self.cursor = 0
        source = graph["edge_index"][0]
        self.values = states[source].copy()
        if donors is not None:
            self.values = states[donors].copy()
        else:
            for group in groups:
                self.values[group] = self.values[group].mean(axis=0)

    def __call__(self, module, arguments):
        incoming = arguments[0]
        stop = self.cursor + len(incoming)
        values = torch.as_tensor(self.values[self.cursor:stop], device=incoming.device, dtype=incoming.dtype)
        changed = incoming.clone()
        changed[:, :values.shape[1]] = values
        self.cursor = stop
        return (changed,)


@torch.no_grad()
def replace_source_states(model, graph, layer_index, states, groups, donors=None):
    """One GNN layer at a time; downstream layers run normally with fixed original degree."""
    patch = MessageInputPatch(states[layer_index], graph, groups, donors)
    handle = model.mp_layers[layer_index].msg_mlp.register_forward_pre_hook(patch)
    try:
        logits = predict(model, graph)
    finally:
        handle.remove()
    if patch.cursor != graph["edge_index"].shape[1]:
        raise RuntimeError("message chunk order does not match the saved graph")
    return logits


def state_summary(graph, states):
    """Similarity BEFORE each GNN layer, compared with a simple neighboring-token chain."""
    prompt = int(graph["prompt_length"])
    labels = graph["gold"].astype(bool)
    source, target = graph["edge_index"]
    rr = source >= prompt
    pairs = {"attention": (source[rr] - prompt, target[rr] - prompt),
             "adjacent": (np.arange(len(labels) - 1), np.arange(1, len(labels)))}
    rows = []
    for layer_index, state in enumerate(states):
        values = state[prompt:]
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = values / np.maximum(norms, 1e-12)
        for name, (left, right) in pairs.items():
            cosine = (normalized[left] * normalized[right]).sum(axis=1)
            for category, selected in {"error_error": labels[left] & labels[right],
                                      "normal_normal": ~labels[left] & ~labels[right],
                                      "mixed": labels[left] != labels[right]}.items():
                rows.append(dict(gnn_layer=layer_index, pair=name, category=category,
                    count=int(selected.sum()), cosine=float(cosine[selected].mean()) if selected.any() else None))
    return rows


class RemoveHeadInteraction:
    """Remove only the two edge channels' conditional non-additive MLP term.

    interaction = M(full) - M(A=0) - M(B=0) + M(A=B=0).
    The source state and all other edge coordinates are held fixed inside M.
    This uses zero as a declared reference, not a natural-attention intervention.
    """

    def __init__(self, hidden_width, first_channel, second_channel):
        self.first = hidden_width + first_channel
        self.second = hidden_width + second_channel
        self.norm_sum = 0.
        self.edges = 0

    def __call__(self, module, arguments, output):
        first_zero = arguments[0].clone()
        first_zero[:, self.first] = 0
        second_zero = arguments[0].clone()
        second_zero[:, self.second] = 0
        both_zero = first_zero.clone()
        both_zero[:, self.second] = 0
        # Direct forward bypasses this Sequential's hook, preventing recursion.
        additive = module.forward(first_zero) + module.forward(second_zero) - module.forward(both_zero)
        interaction = output - additive
        self.norm_sum += float(torch.linalg.vector_norm(interaction, dim=1).sum().cpu())
        self.edges += len(output)
        return additive


@torch.no_grad()
def remove_head_interaction(model, graph, gnn_layer, first_channel, second_channel):
    intervention = RemoveHeadInteraction(model.in_proj.out_features, first_channel, second_channel)
    handle = model.mp_layers[gnn_layer].msg_mlp.register_forward_hook(intervention)
    try:
        logits = predict(model, graph)
    finally:
        handle.remove()
    diagnostics = dict(edges=intervention.edges,
        mean_message_interaction_norm=intervention.norm_sum / intervention.edges if intervention.edges else None,
        reference="zero only the two selected edge channels; keep source state and all other channels")
    return logits, diagnostics
