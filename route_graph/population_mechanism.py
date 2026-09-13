"""Full-response finite message interventions; no factuality labels or detector."""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

CONDITIONS = (
    "base",
    "sham",
    "source_01",
    "source_1",
    "history_01",
    "history_1",
    "source_permute",
    "mlp_01",
)


def endpoint_permutation(count, identity, seed=20260912):
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(
        count
    )


@torch.inference_mode()
def response_forward(
    model,
    ids,
    prompt_length,
    source_mask,
    condition,
    permutation,
    history_window=16,
    chunk=64,
):
    """Intervene from query P-1 onward, retaining the complete causal input.

    Eager attention lives for only its current layer. O is rerun at the original
    full shape for exact zero-delta sham. Baseline profiles use fp32 A*V/O.
    """
    if condition not in CONDITIONS or not 1 <= prompt_length <= len(ids):
        raise ValueError("invalid condition or prompt length")
    if chunk < 1 or history_window < 1:
        raise ValueError("positive query block and history window required")
    if (
        model.config.model_type != "llama"
        or model.config._attn_implementation != "eager"
    ):
        raise ValueError("this protocol requires Llama eager attention")
    mask = np.asarray(source_mask, dtype=bool)
    if mask.shape != (prompt_length,) or not mask.any():
        raise ValueError("nonempty prompt-aligned source mask required")
    sources = np.flatnonzero(mask).tolist()
    if sorted(permutation) != list(range(len(sources))):
        raise ValueError("invalid source endpoint permutation")
    start, length = prompt_length - 1, len(ids)
    queries = list(range(start, length))
    layers, heads = len(model.model.layers), model.config.num_attention_heads
    profiles = {}
    if condition == "base":
        for name in ("source_attention", "remote_history_attention"):
            profiles[name] = np.empty((layers, len(queries), heads), np.float32)
        for name in ("source_message_norm", "remote_history_message_norm", "mlp_norm"):
            profiles[name] = np.empty((layers, len(queries)), np.float32)
    handles = []
    mass_error = 0.0
    permutation = torch.as_tensor(permutation, device=model.device)
    positions = torch.arange(length, device=model.device)

    def register_layer(index, layer):
        cache = {}

        def value_hook(module, args, output):
            cache["values"] = output[0].reshape(
                length, model.config.num_key_value_heads, -1
            )

        def context_hook(module, args):
            cache["context"] = args[0]

        def attention_hook(module, args, output):
            nonlocal mass_error
            changed = cache["context"].clone() if condition != "base" else None
            value = (
                cache["values"]
                .float()
                .repeat_interleave(module.num_key_value_groups, dim=1)
            )
            for left in range(start, length, chunk):
                right = min(left + chunk, length)
                weights = output[1][0, :, left:right].float()
                source = weights[:, :, sources]
                remote_mask = (positions[None, :] >= prompt_length) & (
                    positions[None, :]
                    <= torch.arange(left, right, device=model.device)[:, None]
                    - history_window
                )
                remote = weights * remote_mask[None]
                need_source = condition in {
                    "base",
                    "source_01",
                    "source_1",
                    "source_permute",
                }
                need_history = condition in {"base", "history_01", "history_1"}
                source_context = (
                    torch.einsum("hqk,khd->qhd", source, value[sources])
                    if need_source
                    else None
                )
                history_context = (
                    torch.einsum("hqk,khd->qhd", remote, value)
                    if need_history
                    else None
                )
                if condition == "base":
                    target = slice(left - start, right - start)
                    profiles["source_attention"][index, target] = (
                        source.sum(-1).T.cpu().numpy()
                    )
                    profiles["remote_history_attention"][index, target] = (
                        remote.sum(-1).T.cpu().numpy()
                    )
                    for name, context in (
                        ("source", source_context),
                        ("remote_history", history_context),
                    ):
                        message = F.linear(
                            context.reshape(right - left, -1),
                            module.o_proj.weight.float(),
                        )
                        profiles[f"{name}_message_norm"][index, target] = (
                            message.norm(dim=-1).cpu().numpy()
                        )
                    continue
                if condition.startswith("source_") and condition != "source_permute":
                    delta = -(0.1 if condition.endswith("01") else 1.0) * source_context
                elif condition.startswith("history_"):
                    delta = (
                        -(0.1 if condition.endswith("01") else 1.0) * history_context
                    )
                elif condition == "source_permute":
                    permuted = source[:, :, permutation]
                    mass_error = max(
                        mass_error,
                        float((permuted.sum(-1) - source.sum(-1)).abs().max()),
                    )
                    delta = (
                        torch.einsum("hqk,khd->qhd", permuted, value[sources])
                        - source_context
                    )
                else:
                    delta = torch.zeros(
                        (right - left, heads, module.head_dim), device=model.device
                    )
                changed[0, left:right] = (
                    changed[0, left:right].float() + delta.reshape(right - left, -1)
                ).to(changed.dtype)
            if changed is not None:
                projected = module.o_proj(changed)
                cache.clear()
                return projected, output[1]
            cache.clear()

        def mlp_hook(module, args, output):
            if condition == "base":
                profiles["mlp_norm"][index] = (
                    output[0, start:].float().norm(dim=-1).cpu().numpy()
                )
            elif condition == "mlp_01":
                changed = output.clone()
                changed[0, start:] = (output[0, start:].float() * 0.9).to(output.dtype)
                return changed

        if condition != "mlp_01":
            handles.append(layer.self_attn.v_proj.register_forward_hook(value_hook))
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(context_hook)
            )
            handles.append(layer.self_attn.register_forward_hook(attention_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))

    try:
        for index, layer in enumerate(model.model.layers):
            register_layer(index, layer)
        output = model.model(
            torch.as_tensor(ids, device=model.device)[None],
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        states = output.last_hidden_state[0].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(states).all():
        raise ValueError("nonfinite final hidden states")
    return states, profiles, {"source_mass_max_error": mass_error}


@torch.inference_mode()
def compare_states(model, baseline, altered, saved_tokens, chunk=64):
    """Actual full-vocabulary effects, fixed baseline top2, in bounded blocks."""
    if baseline.shape != altered.shape or len(saved_tokens) != len(baseline):
        raise ValueError("state and actual-response token identity differ")
    values = {
        name: []
        for name in (
            "entropy",
            "argmax",
            "max_ties",
            "saved_logp",
            "margin",
            "js",
            "saved_logp_change",
            "margin_change",
            "argmax_changed",
            "top2_ids",
            "top2_logits",
        )
    }
    for start in range(0, len(baseline), chunk):
        stop = min(start + chunk, len(baseline))
        original = model.lm_head(
            baseline[None, start:stop].to(model.device, model.dtype)
        )[0].float()
        current = model.lm_head(
            altered[None, start:stop].to(model.device, model.dtype)
        )[0].float()
        pair = original.topk(2, dim=-1).indices
        logp, logq = original.log_softmax(-1), current.log_softmax(-1)
        mixture = torch.logaddexp(logp, logq) - np.log(2)
        js = 0.5 * (
            (logp.exp() * (logp - mixture)).sum(-1)
            + (logq.exp() * (logq - mixture)).sum(-1)
        )
        identical = (original == current).all(-1)
        js = torch.where(identical, 0.0, js.clamp_min(0))
        rows = torch.arange(stop - start, device=model.device)
        tokens = torch.as_tensor(saved_tokens[start:stop], device=model.device)
        native_pair, altered_pair = original.gather(-1, pair), current.gather(-1, pair)
        batch = {
            "entropy": -(logq.exp() * logq).sum(-1),
            "argmax": current.argmax(-1),
            "max_ties": (current == current.max(-1, keepdim=True).values).sum(-1),
            "saved_logp": logq[rows, tokens],
            "margin": altered_pair[:, 0] - altered_pair[:, 1],
            "js": js,
            "saved_logp_change": logq[rows, tokens] - logp[rows, tokens],
            "margin_change": (altered_pair[:, 0] - altered_pair[:, 1])
            - (native_pair[:, 0] - native_pair[:, 1]),
            "argmax_changed": current.argmax(-1) != original.argmax(-1),
            "top2_ids": pair,
            "top2_logits": altered_pair,
        }
        for name, tensor in batch.items():
            values[name].append(tensor.cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in values.items()}
