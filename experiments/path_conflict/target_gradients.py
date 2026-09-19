"""Trace a final contrast through native Llama layers, one layer on GPU at a time.

Scores are derivatives of message gates at the observed computation, not finite
causal effects or a conservation law. No source roles enter this computation.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .operators import grouped_values, vocabulary_logits


def move_tensors(value, device):
    """Move captured layer arguments, including the RoPE pair and causal mask."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    return value


@contextmanager
def installed_hooks(handles):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_layers(model, prefix_ids):
    """Keep CPU checkpoints of the actual decoder calls, without a full tape."""
    checkpoints = []
    final = []

    def save_input(module, args, kwargs):
        # The model-level collector stays off; only this layer exposes A.
        kwargs = dict(kwargs, output_attentions=True)
        checkpoints.append((move_tensors(args, "cpu"), move_tensors(kwargs, "cpu")))
        return args, kwargs

    def save_final(module, args):
        final.append(args[0].detach().cpu())

    handles = [layer.register_forward_pre_hook(save_input, with_kwargs=True)
               for layer in model.model.layers]
    handles.append(model.model.norm.register_forward_pre_hook(save_final))
    device = next(model.parameters()).device
    ids = torch.tensor([prefix_ids], device=device)
    with installed_hooks(handles), torch.no_grad():
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, output_attentions=False, return_dict=True)
    return checkpoints, final[0]


def final_direction(model, state, candidate_ids):
    """Differentiate the real final norm and FP32 two-token output contrast."""
    state = state.detach().to(next(model.parameters()).device).requires_grad_(True)
    correct, wrong = candidate_ids
    direction = model.lm_head.weight[correct].float() - model.lm_head.weight[wrong].float()
    normalized = model.model.norm(state)
    margin = normalized[0, -1].float() @ direction
    if model.lm_head.bias is not None:
        margin = margin + model.lm_head.bias[correct] - model.lm_head.bias[wrong]
    gradient, = torch.autograd.grad(margin, state)
    return gradient.detach(), float(margin.detach())


def tail_direction(model, state, prefix_length, candidate, sign):
    """Tail log-probability VJP without retaining FP32 copies of the whole head."""
    device = next(model.parameters()).device
    state = state.detach().to(device).requires_grad_(True)
    normalized = model.model.norm(state)
    positions = torch.arange(prefix_length, state.shape[1], device=device)
    targets = torch.tensor(candidate[1:], device=device)
    with torch.no_grad():
        logits = vocabulary_logits(model, normalized[0, positions])
        logp = logits.double().log_softmax(-1)
        probabilities = logp.exp().float()
        mean_weight = torch.zeros(len(positions), state.shape[-1], device=device)
        for start in range(0, logits.shape[-1], 4096):
            stop = start + 4096
            mean_weight += probabilities[:, start:stop] @ model.lm_head.weight[start:stop].float()
        selected_weight = model.lm_head.weight[targets].float()
        seed = torch.zeros_like(normalized)
        seed[0, positions] = (sign * (selected_weight - mean_weight)).to(seed)
        value = sign * float(logp[torch.arange(len(targets), device=device), targets].sum())
    gradient, = torch.autograd.grad(normalized, state, seed)
    return gradient.detach(), value


def prepare_branches(model, probe, objective):
    prefix = list(probe["prefix_ids"])
    checkpoints, final = capture_layers(model, prefix)
    incoming, first = final_direction(model, final, [tokens[0] for tokens in probe["candidates"]])
    branches = [dict(checkpoints=checkpoints, final=final, incoming=incoming)]
    value = first
    if objective == "sequence_margin":
        for candidate, sign in zip(probe["candidates"], (1, -1)):
            if len(candidate) == 1:
                continue
            checkpoints, final = capture_layers(model, prefix + list(candidate[:-1]))
            incoming, tail = tail_direction(model, final, len(prefix), candidate, sign)
            branches.append(dict(checkpoints=checkpoints, final=final, incoming=incoming))
            value += tail
    return branches, first, value


def replay_layer(layer, checkpoint, incoming):
    """Exact layer VJP; Q/K, softmax, V, both norms and MLP stay differentiable."""
    args, kwargs = move_tensors(checkpoint, incoming.device)
    hidden = args[0].requires_grad_(True)
    captured = {}

    def save_heads(module, args):
        captured["heads"] = args[0]

    def save_values(module, args, output):
        captured["values"] = output.detach()

    def save_attention(module, args, output):
        captured["attention"] = output[1].detach()

    def save_mlp(module, args, output):
        captured["mlp"] = output

    handles = [layer.self_attn.o_proj.register_forward_pre_hook(save_heads),
               layer.self_attn.v_proj.register_forward_hook(save_values),
               layer.self_attn.register_forward_hook(save_attention),
               layer.mlp.register_forward_hook(save_mlp)]
    with installed_hooks(handles), torch.enable_grad():
        output = layer(hidden, *args[1:], **kwargs)
        output = output[0] if isinstance(output, tuple) else output
        gradients = torch.autograd.grad(
            output, (hidden, captured["heads"], captured["mlp"]), incoming)
    captured = {key: value.detach() for key, value in captured.items()}
    return gradients, captured, output.detach()


def edge_scores(attention, values, head_gradient):
    """[H,Q,S] gate derivative: A[h,q,s] * <dM/dz[h,q], V[h,s]>."""
    return attention.float() * torch.einsum("hqd,hsd->hqs", head_gradient.float(), values.float())


def selected_edges(layer, attention, values, scores, weight, top_k):
    """Retain both signs per physical head; never average heads or source IDs."""
    rows = []
    heads, length, _ = scores.shape
    width = values.shape[-1]
    for head in range(heads):
        candidates = set()
        for sign in (-1, 1):
            ranked, indices = (sign * scores[head]).flatten().topk(min(top_k, length * length))
            candidates.update(int(index) for index, value in zip(indices, ranked) if value > 0)
        for index in sorted(candidates):
            receiver, source = divmod(index, length)
            mass = attention[head, receiver, source]
            value = values[head, source]
            block = weight[:, head * width:(head + 1) * width]
            message = F.linear(mass * value, block)
            unit = f"L{layer}H{head}Q{receiver}S{source}"
            rows.append(dict(unit=unit, layer=layer, head=head, receiver=receiver,
                             source=source, attention_mass=float(mass),
                             value_energy=float(value.float().square().sum()),
                             message_norm=float(message.float().norm()),
                             final_linear_support=float(scores[head, receiver, source])))
    return rows


def branch_scores(model, captured, gradients, prefix_length):
    heads = model.config.num_attention_heads
    values = grouped_values(captured["values"], heads, model.config.num_key_value_heads)[0]
    gradient = gradients[1][0].view(-1, heads, values.shape[-1]).transpose(0, 1)
    attention = captured["attention"][0]
    # Counterfactual continuation is used for the objective, never as a cut site.
    scores = edge_scores(attention[:, :prefix_length, :prefix_length],
                         values[:, :prefix_length], gradient[:, :prefix_length])
    outputs = captured["heads"][0, :prefix_length].float()
    direct = (outputs * gradients[1][0, :prefix_length].float()).view(prefix_length, heads, -1).sum(-1).T
    mlp = (captured["mlp"].float() * gradients[2].float()).sum(-1)[0, :prefix_length]
    return scores, direct, mlp, attention[:, :prefix_length, :prefix_length], values[:, :prefix_length]


def replay_branches(model, index, branches, prefix_length):
    scores, direct, mlp = None, None, None
    replay_error = 0.
    for number, branch in enumerate(branches):
        checkpoints = branch["checkpoints"]
        gradients, captured, output = replay_layer(
            model.model.layers[index], checkpoints[index], branch["incoming"])
        expected = branch["final"] if index == len(checkpoints) - 1 else checkpoints[index + 1][0][0]
        replay_error = max(replay_error, float((output - expected.to(output)).float().abs().max()))
        measured, head_direct, mlp_direct, attention, values = branch_scores(
            model, captured, gradients, prefix_length)
        if number == 0:
            scores, direct, mlp = measured, head_direct, mlp_direct
            base_attention, base_values = attention, values
        else:
            scores += measured
            direct += head_direct
            mlp += mlp_direct
        branch["incoming"] = gradients[0].detach()
    return scores, direct, mlp, base_attention, base_values, replay_error


def layer_arrays(scores, attention, mlp):
    arrays = dict(node_signed=scores.sum(-1), node_positive=scores.clamp_min(0).sum(-1),
                  node_negative=scores.clamp_max(0).sum(-1),
                  target_sources=scores[:, -1], target_attention=attention[:, -1],
                  mlp_signed=mlp)
    return {key: value.float().cpu().numpy() for key, value in arrays.items()}


def trace_target(model, probe, top_k=2, progress=True, objective="sequence_margin"):
    """One prefix, one final contrast; CPU checkpoints plus one layer's tape."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Target tracing requires eval() and frozen model parameters")
    branches, margin, objective_value = prepare_branches(model, probe, objective)
    candidates = [tokens[0] for tokens in probe["candidates"]]
    rows, measured_layers, diagnostics = [], {}, []
    layers = model.model.layers
    for index in tqdm(reversed(range(len(layers))), total=len(layers),
                      desc="final-target VJP", disable=not progress, leave=False):
        scores, direct, mlp, attention, values, replay_error = replay_branches(
            model, index, branches, len(probe["prefix_ids"]))
        rows.extend(selected_edges(index, attention, values, scores,
                                   layers[index].self_attn.o_proj.weight, top_k))
        measured_layers[index] = layer_arrays(scores, attention, mlp)
        diagnostics.append(dict(layer=index, replay_max_error=replay_error,
                                edge_sum_max_error=float((scores.sum(-1) - direct).abs().max())))
    arrays = {key: np.stack([measured_layers[index][key] for index in range(len(layers))])
              for key in measured_layers[0]}
    return pd.DataFrame(rows), arrays, dict(
        next_margin=margin, diagnostics=diagnostics, candidate_first_ids=candidates,
        objective=objective, objective_value=objective_value,
        meaning="native_final_gate_derivative_not_finite_effect",
        axes="node_*: layer,head,receiver; target_*: layer,head,source")
