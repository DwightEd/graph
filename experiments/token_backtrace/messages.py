"""Native whole-head carriers and signed, target-specific directional derivatives.

The supported model has Llama-style decoder layers. A carrier is the complete
head output before W_O, not one attention edge. These measurements describe
influence under fixed text; they do not determine factual correctness.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import torch


def site_index(value, name, size):
    """A native address cannot use Python's negative/boolean indexing semantics."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
        raise ValueError(f'{name} must be an integer in [0, {size})')
    return int(value)


def validate_selection(selection, trace):
    """Check cached physical head coordinates once before gathering messages."""
    selection = torch.as_tensor(selection)
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if selection.dtype not in integer_types or selection.ndim != 3 or selection.shape[2] != 2:
        raise ValueError('selection requires integer physical addresses [T,K,2]')
    if selection.shape[0] != len(trace.answer) or selection.shape[1] == 0:
        raise ValueError('selection must cover every original token with at least one head')
    layer, head = selection[..., 0], selection[..., 1]
    if ((layer < 0) | (layer >= len(trace.messages)) | (head < 0) | (head >= trace.heads)).any():
        raise ValueError('selection contains an out-of-range layer/head address')
    ordered = torch.sort(layer * trace.heads + head, dim=1).values
    if (torch.diff(ordered, dim=1) == 0).any():
        raise ValueError('selection repeats a physical head at the same carrier')
    return selection.long()


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
    if model.training:
        raise ValueError('Native tracing requires model.eval() for replay consistency.')
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


def validate_alignment(original, donors, alignments):
    """Maps original answer indices to donor indices; -1 explicitly means unknown.

    Eligible carriers share original token IDs and a common position displacement
    in all donor worlds. Upstream edit alignment supplies the exact suffix maps.
    """
    token_count = len(original.answer)
    maps = [torch.as_tensor(alignments[name]) for name in donors]
    if any(mapping.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
           for mapping in maps):
        raise ValueError('Alignment indices must be integers, with -1 for unknown.')
    maps = [mapping.long() for mapping in maps]
    if any(mapping.shape != (token_count,) for mapping in maps):
        raise ValueError('Every donor alignment must cover every original answer token.')
    original_ids = torch.tensor(original.answer)
    for mapping, donor in zip(maps, donors.values()):
        valid = mapping >= 0
        if torch.any(mapping < -1) or torch.any(mapping[valid] >= len(donor.answer)):
            raise ValueError('Alignment index is outside its donor answer.')
        if not torch.equal(original_ids[valid], torch.tensor(donor.answer)[mapping[valid]]):
            raise ValueError('Aligned carrier token IDs differ between original and donor.')
        if valid.any():
            first = int(torch.where(valid)[0][0])
            expected = torch.arange(int(mapping[first]), len(donor.answer))
            if not torch.equal(mapping[first:], expected):
                raise ValueError('Carrier alignment must be an exact contiguous suffix.')
    joint = torch.stack([mapping >= 0 for mapping in maps]).all(dim=0)
    if any(not torch.equal(mapping[joint], maps[0][joint]) for mapping in maps[1:]):
        raise ValueError('Donor alignments do not have the same position displacement.')
    if any(donor.prompt != original.prompt for donor in donors.values()):
        raise ValueError('Carrier worlds must preserve the complete original prompt.')
    return dict(zip(donors, maps)), joint


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
    selection = validate_selection(selection, original)
    maps, joint = validate_alignment(original, donors, alignments)
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
    layer = site_index(layer, 'layer', len(model.model.layers))
    head = site_index(head, 'head', model.config.num_attention_heads)
    if isinstance(receiver, bool) or not isinstance(receiver, Integral) or receiver < 0:
        raise ValueError('receiver must be a nonnegative integer native position')
    attention = model.model.layers[layer].self_attn
    delta = torch.as_tensor(delta)
    if delta.shape != (attention.head_dim,) or not torch.isfinite(delta).all() or not isfinite(alpha):
        raise ValueError('patch needs a finite [head_dim] displacement and finite dose')

    def replace(module, inputs):
        site_index(receiver, 'receiver', inputs[0].shape[1])
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
    carrier = site_index(carrier, 'carrier', len(original.answer))
    if direction.shape != original.logits.shape or not torch.isfinite(direction).all():
        raise ValueError('finite directions must be finite with the original [T,V] shape')
    receiver = len(original.prompt) + carrier
    with patch_message(model, layer, head, receiver, delta, alpha):
        changed = native_trace(model, original.prompt, original.answer, gradients=False)
    return ((changed.logits - original.logits.detach()) * direction).sum(dim=-1).cpu()
