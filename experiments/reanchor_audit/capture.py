"""Collect each native Llama layer once; retain sparse routes, not hidden tapes."""

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..path_conflict.native import request_layer_attention
from ..path_conflict.operators import grouped_values, vocabulary_logits


def lookback_gain(attention, prompt_length, window, special):
    """Compare adjacent receivers over exactly the same earlier key set."""
    heads, length, _ = attention.shape
    positions = torch.arange(length, device=attention.device)
    ends = torch.maximum(positions - window, torch.full_like(positions, prompt_length)) - 1
    valid = torch.tensor(~np.asarray(special), device=attention.device)
    cumulative = (attention.float() * valid).cumsum(-1)
    keys = ends[None, :, None].expand(heads, -1, 1)
    current = cumulative.gather(-1, keys).squeeze(-1)
    previous = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1)
    gain = current - previous.gather(-1, keys).squeeze(-1)
    gain[:, 0] = float("nan")
    return gain.cpu().numpy()


def projected_norms(values, weight):
    width = values.shape[-1]
    norms = []
    for head in range(len(values)):
        block = weight[:, head * width:(head + 1) * width]
        transformed = F.linear(values[head].float(), block.float())
        norms.append(transformed.norm(dim=-1))
    return torch.stack(norms)


def layer_record(module, attention, values, residual_norm, output, top_k, prompt_length, window, special):
    mass, source = attention.topk(min(top_k, attention.shape[-1]), dim=-1)
    projected = projected_norms(values, module.o_proj.weight)
    full_norm_sum = torch.einsum("hqs,hs->q", attention.float(), projected)
    head_output = (attention @ values).transpose(0, 1).reshape(attention.shape[1], -1)
    rebuilt = F.linear(head_output, module.o_proj.weight, module.o_proj.bias)
    error = (rebuilt.float() - output.float()).norm(dim=-1)
    error = error / output.float().norm(dim=-1).clamp_min(1e-30)
    return dict(source=source.cpu().numpy().astype(np.int32),
        attention=mass.float().cpu().numpy(), retained_mass=mass.float().sum(-1).cpu().numpy(),
        value_energy=values.float().square().sum(-1).cpu().numpy(),
        projected_value_norm=projected.cpu().numpy(), residual_norm=residual_norm,
        full_message_norm_sum=full_norm_sum.cpu().numpy(),
        lookback_gain=lookback_gain(attention, prompt_length, window, special),
        reconstruction_error=error.cpu().numpy())


def install_capture(model, records, pending, probe, top_k, window, special, progress):
    handles = []
    for index, layer in enumerate(model.model.layers):
        def residual(module, inputs, index=index):
            pending[index] = dict(residual_norm=inputs[0][0].float().norm(dim=-1).cpu().numpy())
        def values(module, inputs, output, index=index):
            pending[index]["values"] = output
        def attention(module, inputs, output, index=index):
            state = pending.pop(index)
            config = model.config
            value = grouped_values(state["values"], config.num_attention_heads, config.num_key_value_heads)[0]
            records[index] = layer_record(module, output[1][0], value, state["residual_norm"],
                output[0][0], top_k, probe["prompt_length"], window, special)
            progress.update(1)
        handles.extend((layer.register_forward_pre_hook(request_layer_attention, with_kwargs=True),
            layer.input_layernorm.register_forward_pre_hook(residual),
            layer.self_attn.v_proj.register_forward_hook(values),
            layer.self_attn.register_forward_hook(attention)))
    return handles


def capture_graph(model, probe, top_k=8, window=10, special=None):
    ids = torch.tensor([probe["prefix_ids"]], device=next(model.parameters()).device)
    special = np.zeros(ids.shape[1], dtype=bool) if special is None else np.asarray(special)
    records, pending = {}, {}
    with tqdm(total=len(model.model.layers), desc="native route capture", leave=False) as progress:
        handles = install_capture(model, records, pending, probe, top_k, window, special, progress)
        try:
            with torch.inference_mode():
                result = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, output_attentions=False, return_dict=True)
                logp = vocabulary_logits(model, result.last_hidden_state[0, -1]).double().log_softmax(-1)
        finally:
            for handle in handles:
                handle.remove()
    arrays = {key: np.stack([records[layer][key] for layer in range(len(records))]) for key in records[0]}
    arrays.update(prefix_ids=np.array(probe["prefix_ids"]), prompt_length=np.array(probe["prompt_length"]),
                  candidate_first_logp=logp[[tokens[0] for tokens in probe["candidates"]]].cpu().numpy())
    return arrays
