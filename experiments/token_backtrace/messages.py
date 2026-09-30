"""Native whole-head carriers and signed, target-specific directional derivatives.

The supported model has Llama-style decoder layers. A carrier is the complete
head output before W_O, not one attention edge. These measurements describe
influence under fixed text; they do not determine factual correctness.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class MessageTrace:
    """Messages are exact projection inputs [1, P+T, H*d], retaining autograd."""
    prompt: tuple
    answer: tuple
    logits: torch.Tensor
    messages: tuple
    residuals: tuple
    heads: int
    head_dim: int

    @property
    def prediction_positions(self):
        return torch.arange(len(self.answer)) + len(self.prompt) - 1

    @property
    def carrier_positions(self):
        return torch.arange(len(self.answer)) + len(self.prompt)


@contextmanager
def capture_messages(model):
    """Observe native tensors without replacing attention or its derivatives."""
    layers = model.model.layers
    messages, residuals = [None] * len(layers), [None] * len(layers)
    handles = []

    def remember(destination, index, detach=False):
        def observe(module, inputs):
            destination[index] = inputs[0].detach() if detach else inputs[0]
        return observe

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            remember(messages, index)))
        handles.append(layer.input_layernorm.register_forward_pre_hook(
            remember(residuals, index, detach=True)))
    try:
        yield messages, residuals
    finally:
        for handle in handles:
            handle.remove()


def native_trace(model, prompt, answer, gradients=True):
    """Keep all tokens; the final carrier has no later answer target to affect."""
    tokens = torch.tensor([list(prompt) + list(answer)], device=model.device)
    with torch.set_grad_enabled(gradients), capture_messages(model) as captured:
        embeddings = model.model.embed_tokens(tokens).detach().requires_grad_(gradients)
        hidden = model.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        prediction_hidden = hidden[0, len(prompt) - 1:len(prompt) + len(answer) - 1]
        logits = model.lm_head(prediction_hidden)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    messages, residuals = captured
    return MessageTrace(tuple(prompt), tuple(answer), logits, tuple(messages),
                        tuple(residuals), model.config.num_attention_heads,
                        model.model.layers[0].self_attn.head_dim)


@torch.no_grad()
def top_heads(model, original, count=4):
    """Select O-only physical (layer, head) addresses per carrier; ties are lexical.

    Scores [T,L,H] are ||W_O^h m_qh|| / (||pre-attn residual_q||+1e-8).
    The head Gram matrix avoids allocating [T,H,model_dim] projected vectors.
    """
    layer_scores = []
    positions = original.carrier_positions.to(original.logits.device)
    for index, layer in enumerate(model.model.layers):
        weight = layer.self_attn.o_proj.weight.detach().float()
        weight = weight.reshape(-1, original.heads, original.head_dim).permute(1, 0, 2)
        gram = weight.transpose(1, 2) @ weight
        message = original.messages[index][0, positions].float()
        message = message.reshape(-1, original.heads, original.head_dim)
        square_norm = torch.einsum('qhd,hde,qhe->qh', message, gram, message)
        residual_norm = original.residuals[index][0, positions].float().norm(dim=-1)
        layer_scores.append(square_norm.clamp_min(0).sqrt() / (residual_norm[:, None] + 1e-8))
    scores = torch.stack(layer_scores, dim=1)
    ranked = torch.argsort(scores.flatten(1), dim=1, descending=True, stable=True)[:, :count]
    selection = torch.stack((ranked // original.heads, ranked % original.heads), dim=-1)
    return selection.cpu(), scores.cpu()


def _selected_messages(trace, selection, answer_positions):
    """Gather [T,K,d_head] at explicit answer positions; -1 rows remain NaN."""
    device = trace.logits.device
    selection = selection.to(device)
    answer_positions = answer_positions.to(device)
    valid = answer_positions >= 0
    values = trace.messages[0].new_full((*selection.shape[:2], trace.head_dim), torch.nan)
    for layer, message in enumerate(trace.messages):
        rows, slots = torch.where((selection[:, :, 0] == layer) & valid[:, None])
        heads = selection[rows, slots, 1]
        carriers = answer_positions[rows] + len(trace.prompt)
        values[rows, slots] = message[0].reshape(-1, trace.heads, trace.head_dim)[carriers, heads]
    return values


def aligned_deltas(original, donors, alignments, selection):
    """Return signed donor-minus-O messages [donor,T,K,d] and joint availability."""
    selection = torch.as_tensor(selection, dtype=torch.long)
    maps = {name: torch.as_tensor(alignments[name], dtype=torch.long) for name in donors}
    joint = torch.stack([mapping >= 0 for mapping in maps.values()]).all(dim=0)
    with torch.no_grad():
        baseline = _selected_messages(original, selection, torch.arange(len(original.answer)))
        baseline = baseline.to(original.logits.dtype)
        differences = [_selected_messages(donor, selection, maps[name]).to(baseline) - baseline
                       for name, donor in donors.items()]
        deltas = torch.stack(differences)
        deltas[:, ~joint] = torch.nan
    return deltas, joint, maps


def direction_vector(original_logp, donor_logp):
    """Frozen full-vocabulary a = p_O * (log p_P - log p_O - its p_O mean)."""
    probability = original_logp.detach().exp()
    change = donor_logp.detach() - original_logp.detach()
    return (probability * (change - (probability * change).sum(dim=-1, keepdim=True))).detach()


def _selected_gradients(gradients, original, selection):
    """Gather dF/dm for the same physical heads chosen using O alone."""
    device = original.logits.device
    selection = selection.to(device)
    positions = original.carrier_positions.to(device)
    values = gradients[0].new_empty((*selection.shape[:2], original.head_dim))
    for layer, gradient in enumerate(gradients):
        rows, slots = torch.where(selection[:, :, 0] == layer)
        heads = selection[rows, slots, 1]
        values[rows, slots] = gradient[0].reshape(-1, original.heads, original.head_dim)[positions[rows], heads]
    return values


def directional_vjp(original, donors, alignments, selection):
    """Measure [direction,donor,carrier,selected-head,target] signed eta.

    One target VJP reaches every earlier native carrier simultaneously. Each
    donor defines its own fixed direction, including E/U self-aligned controls.
    NaN plus `valid` distinguishes unmeasured or noncausal edges from measured 0.
    """
    selection = torch.as_tensor(selection)
    deltas, aligned, maps = aligned_deltas(original, donors, alignments, selection)
    selection = selection.long()
    names, token_count = tuple(donors), len(original.answer)
    shape = (len(names), len(names), token_count, selection.shape[1], token_count)
    effects = torch.full(shape, torch.nan, dtype=original.logits.dtype)
    indices = torch.arange(token_count)
    eligible = aligned[:, None] & aligned[None, :] & (indices[:, None] < indices[None, :])
    valid = eligible[:, None, :].expand(-1, selection.shape[1], -1).clone()
    original_logp = original.logits.detach().log_softmax(-1)
    backward_calls = 0
    for direction_index, name in enumerate(names):
        donor_logp = donors[name].logits.detach().log_softmax(-1)
        for target in torch.where(eligible.any(dim=0))[0].tolist():
            direction = direction_vector(original_logp[target], donor_logp[maps[name][target]])
            objective = (direction * original.logits[target]).sum()
            gradients = torch.autograd.grad(objective, original.messages, retain_graph=True)
            selected = _selected_gradients(gradients, original, selection).to(deltas)
            response = (deltas * selected[None]).sum(dim=-1).detach().cpu()
            response[:, ~valid[:, :, target]] = torch.nan
            effects[direction_index, :, :, :, target] = response
            backward_calls += 1
    return dict(eta=effects, valid=valid, aligned=aligned, donor_names=names,
                selection=selection, backward_calls=backward_calls,
                measurement='native directional first derivative; not finite effect')


@contextmanager
def patch_message(model, layer, head, receiver, delta, alpha=1.):
    """Add alpha*delta to one whole pre-W_O head; recompute native descendants."""
    attention = model.model.layers[layer].self_attn
    delta = torch.as_tensor(delta)

    def replace(module, inputs):
        changed = inputs[0].clone()
        begin = head * attention.head_dim
        changed[0, receiver, begin:begin + attention.head_dim] += alpha * delta.to(changed)
        return (changed, *inputs[1:])

    handle = attention.o_proj.register_forward_pre_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def finite_effect(model, original, layer, head, carrier, delta, direction, alpha=1.):
    """Return exact finite delta F for fixed directions [T,V] at all targets."""
    receiver = len(original.prompt) + carrier
    with patch_message(model, layer, head, receiver, delta, alpha):
        changed = native_trace(model, original.prompt, original.answer, gradients=False)
    return ((changed.logits - original.logits.detach()) * direction).sum(dim=-1).cpu()
