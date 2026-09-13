"""Lossless attributed token graphs of frozen Llama computation.

Tokens are nodes; residual/MLP tensors are high-dimensional node attributes.
Attention[layer, head, receiver, sender] stores directed, typed edge weights.
Layer channels also permit an unfolded computational DAG without duplicating
the archive. No candidate projection, head average, edge pruning, or labels.
"""

import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def capture_sample_graph(model, token_ids, prompt_length, source_mask, special_ids=()):
    """Capture one complete fixed continuation in the same forward operation.

    y[t] is predicted by node prompt_length+t-1; the last target is not an input
    node. All visible prompt and response connections are retained. The final
    residual is captured before model.model.norm, unlike HF's final hidden state.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    source = np.asarray(source_mask, dtype=bool)
    if (
        ids.ndim != 1
        or not 0 < prompt_length < len(ids)
        or source.shape != (prompt_length,)
    ):
        raise ValueError("inconsistent full token sequence and source mask")
    if (
        model.config.model_type != "llama"
        or model.config._attn_implementation != "eager"
    ):
        raise ValueError("this capture requires eager Llama attention")
    layers, length = len(model.model.layers), len(ids) - 1
    heads, hidden = model.config.num_attention_heads, model.config.hidden_size
    kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    graph = {
        "input_ids": ids[:-1],
        "target_ids": ids[prompt_length:],
        "prompt_length": np.array(prompt_length),
        "roles": np.full(length, 2, dtype=np.int8),
        "residual": np.empty((layers + 1, length, hidden), dtype=np.float32),
        "post_attention": np.empty((layers, length, hidden), dtype=np.float32),
        "mlp_update": np.empty((layers, length, hidden), dtype=np.float32),
        "values": np.empty((layers, length, kv_heads, head_dim), dtype=np.float32),
        "attention": np.empty((layers, heads, length, length), dtype=np.float32),
    }
    graph["roles"][:prompt_length] = np.where(source, 0, 1)
    graph["roles"][np.isin(ids[:-1], list(special_ids))] = 3
    handles = []

    def save(name, layer, transform=lambda value: value):
        def hook(module, args, output):
            tensor = transform(output)
            graph[name][layer] = tensor[0].float().cpu().numpy()

        return hook

    def before(name, layer):
        def hook(module, args, kwargs):
            state = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            graph[name][layer] = state[0].float().cpu().numpy()

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.extend(
            [
                layer.register_forward_pre_hook(
                    before("residual", index), with_kwargs=True
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    before("post_attention", index), with_kwargs=True
                ),
                layer.mlp.register_forward_hook(save("mlp_update", index)),
                layer.self_attn.v_proj.register_forward_hook(
                    save(
                        "values",
                        index,
                        lambda value: value.reshape(1, length, kv_heads, head_dim),
                    )
                ),
                layer.self_attn.register_forward_hook(
                    save("attention", index, lambda output: output[1])
                ),
            ]
        )
    handles.append(
        model.model.layers[-1].register_forward_hook(save("residual", layers))
    )
    inputs = torch.as_tensor(ids[:-1], device=model.device)[None]
    try:
        output = model(inputs, use_cache=False, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = output.logits[0, prompt_length - 1 :].float()
    top = logits.topk(8)
    logp = logits.log_softmax(-1)
    graph.update(
        top_ids=top.indices.cpu().numpy(),
        top_logits=top.values.cpu().numpy(),
        log_normalizer=logits.logsumexp(-1).cpu().numpy(),
        entropy_nats=(-(logp.exp() * logp).sum(-1)).cpu().numpy(),
    )
    return graph


def validate_graph(graph):
    """Check complete topology, raw state semantics, and prediction alignment."""
    residual = graph["residual"]
    layers, length, hidden = residual.shape
    layers -= 1
    prompt = int(graph["prompt_length"])
    if not 0 < prompt <= length or graph["input_ids"].shape != (length,):
        raise ValueError("invalid token/prompt alignment")
    if graph["target_ids"].shape != (length - prompt + 1,):
        raise ValueError("missing prediction targets")
    if not np.array_equal(graph["input_ids"][prompt:], graph["target_ids"][:-1]):
        raise ValueError("response node and next-token target identities differ")
    if (
        graph["roles"].shape != (length,)
        or not np.isin(graph["roles"], [0, 1, 2, 3]).all()
    ):
        raise ValueError("invalid node roles")
    for name in ("post_attention", "mlp_update"):
        if graph[name].shape != (layers, length, hidden):
            raise ValueError(f"invalid {name} features")
    attention = graph["attention"]
    if (
        attention.ndim != 4
        or attention.shape[0] != layers
        or attention.shape[2:] != (length, length)
    ):
        raise ValueError("missing full prompt/response attention topology")
    values = graph["values"]
    if (
        values.ndim != 4
        or values.shape[:2] != (layers, length)
        or attention.shape[1] % values.shape[2]
        or values.shape[-1] * attention.shape[1] != hidden
    ):
        raise ValueError("invalid value projection/GQA attributes")
    targets = length - prompt + 1
    for name in ("entropy_nats", "log_normalizer"):
        if graph[name].shape != (targets,) or not np.isfinite(graph[name]).all():
            raise ValueError(f"invalid {name} target alignment")
    maximum_row_error, maximum_add_error = 0.0, 0.0
    # Layer-wise checks avoid allocating a float32 copy of the full graph.
    future = np.triu_indices(length, 1)
    for layer in range(layers):
        weights = attention[layer].astype(np.float32)
        if not np.isfinite(weights).all() or weights.min() < 0:
            raise ValueError("nonfinite or negative edge weights")
        if np.any(weights[:, future[0], future[1]]):
            raise ValueError("attention contains future-to-past edges")
        maximum_row_error = max(
            maximum_row_error, float(abs(weights.sum(-1) - 1).max())
        )
        if maximum_row_error > 0.01:
            raise ValueError("attention rows do not conserve probability mass")
        for name in ("residual", "post_attention", "mlp_update", "values"):
            if not np.isfinite(graph[name][layer]).all():
                raise ValueError(f"nonfinite {name}")
        actual = residual[layer + 1].astype(np.float32)
        expected = graph["post_attention"][layer].astype(np.float32)
        expected += graph["mlp_update"][layer].astype(np.float32)
        maximum_add_error = max(maximum_add_error, float(abs(actual - expected).max()))
    if not np.isfinite(residual[-1]).all():
        raise ValueError("nonfinite final raw residual")
    return {
        "nodes": length,
        "layers": layers,
        "heads": attention.shape[1],
        "hidden_dimension": hidden,
        "full_prompt_topology": True,
        "maximum_attention_row_sum_error": maximum_row_error,
        "maximum_raw_mlp_residual_addition_error": maximum_add_error,
        "array_bytes": sum(array.nbytes for array in graph.values()),
    }


def save_graph(directory, graph, metadata):
    """Store full attributes as mmap-readable arrays; refuse existing paths."""
    directory = Path(directory)
    checks = validate_graph(graph)
    directory.mkdir(parents=True, exist_ok=False)
    for name, array in graph.items():
        np.save(directory / f"{name}.npy", array, allow_pickle=False)
    (directory / "metadata.json").write_text(
        json.dumps({**metadata, "checks": checks}, ensure_ascii=False, indent=2) + "\n"
    )
    return checks
