"""Signed material provenance on the captured response computation DAG.

This is a conditional value-path decomposition, not a counterfactual forward.
Native attention and RMS denominators are fixed. SwiGLU's product is allocated
equally to its two branches. Prompt V states are boundary inputs, not unmixed
input-token identities. Labels and peak detection never enter this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attention_audit import GROUPS

SCHEMA = 1


class CheckpointWeights:
    """Load only the requested layer; no model construction or inference."""

    def __init__(self, directory, device="cpu"):
        self.directory, self.device = Path(directory), device
        self.config = json.loads((self.directory / "config.json").read_text())
        index = self.directory / "model.safetensors.index.json"
        self.files = json.loads(index.read_text())["weight_map"] if index.exists() else {}
        if not self.files and not (self.directory / "model.safetensors").is_file():
            raise ValueError("lineage requires the capture model's local safetensors checkpoint")
        if (self.config.get("model_type") != "llama"
                or self.config.get("hidden_act", "silu") != "silu"
                or self.config.get("attention_bias", False)
                or self.config.get("mlp_bias", False)):
            raise ValueError("this decomposition supports bias-free Llama with SwiGLU")

    def get(self, name):
        from safetensors import safe_open
        if name == "lm_head.weight" and self.config.get("tie_word_embeddings", False):
            name = "model.embed_tokens.weight"
        filename = self.files.get(name, "model.safetensors")
        with safe_open(self.directory / filename, framework="pt", device="cpu") as archive:
            return archive.get_tensor(name).to(self.device)


def norm_component(part, reference, weight, epsilon):
    """Linear allocation under the reference RMS denominator, retaining sign."""
    return part * weight.float() * torch.rsqrt(reference.float().square().mean(-1, keepdim=True) + epsilon)


def swiglu_component(part, reference, weights, epsilon, rule="symmetric"):
    """Allocate both gate and up branches; report the native arithmetic residual.

    For g=Gz, u=Uz, a=SiLU(g), k=a/g (k(0)=1/2):
      M(v) = D [ (a * Uv + u * k * Gv) / 2 ].
    M is linear in v and M(z)=D(a*u). The one-branch rule is exposed only
    for attribution-rule sensitivity; neither rule is an MLP intervention.
    """
    norm, gate, up, down = (weights[k] for k in ("post_norm", "gate", "up", "down"))
    dtype = gate.dtype
    z = norm_component(reference, reference, norm, epsilon).to(dtype)
    v = norm_component(part, reference, norm, epsilon)
    g, u = F.linear(z, gate).float(), F.linear(z, up).float()
    a = F.silu(g.to(dtype)).float()
    up_part = a * F.linear(v, up.float())
    if rule == "symmetric":
        k = torch.where(g != 0, a / torch.where(g != 0, g, 1), .5)
        up_part = .5 * (up_part + u * k * F.linear(v, gate.float()))
    elif rule != "up":
        raise ValueError("MLP rule must be symmetric or up")
    return F.linear(up_part, down.float()), F.linear((a * u).to(dtype), down).float()


def readout_directions(trace, states, weights, chunk=16):
    """Same observed-versus-runner contrast as the native capture, no labels."""
    device = weights.device
    raw = torch.as_tensor(states["final_residual"], device=device)
    count = len(raw) - 1
    ids = torch.as_tensor(trace["token_ids"][trace["row_position"][:count] + 1], device=device)
    norm, unembed = weights.get("model.norm.weight"), weights.get("lm_head.weight")
    epsilon = weights.config["rms_norm_eps"]
    runner = torch.empty(count, device=device, dtype=torch.long)
    direction = torch.zeros_like(raw, dtype=torch.float32)
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        x, target = raw[begin:end], ids[begin:end]
        if "readout_runner_id" in trace:
            other = torch.as_tensor(trace["readout_runner_id"][begin:end], device=device)
        else:
            # Legacy v3 captures did not save runner IDs. Reuse final states;
            # this is only the final RMSNorm/unembedding, not another forward.
            z = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)).to(norm.dtype)
            logits = F.linear(z * norm, unembed).float()
            logits.scatter_(1, target[:, None], -torch.inf)
            other = logits.argmax(-1)
        runner[begin:end] = other
        difference = unembed[target].float() - unembed[other].float()
        direction[begin:end] = norm_component(difference, x, norm, epsilon)
    return direction, runner.cpu().numpy()


def attention_chunks(trace, qk, history, layer, device, chunk):
    """All sources, with captured native history replacing reconstructed rows."""
    query = torch.as_tensor(qk[f"query_{layer}"], device=device)
    key = torch.as_tensor(qk[f"key_{layer}"], device=device)
    dtype = getattr(torch, str(qk[f"dtype_{layer}"]))
    h, r, _ = query.shape
    key = key.repeat_interleave(h // len(key), 0).to(dtype)
    rows = torch.as_tensor(trace["row_position"], device=device)
    native_history = torch.as_tensor(history[f"L{layer}"], device=device)
    source = torch.arange(key.shape[1], device=device)
    for begin in range(0, r, chunk):
        end = min(begin + chunk, r)
        scores = (query[:, begin:end].to(dtype) @ key.transpose(-1, -2)) * float(qk[f"scale_{layer}"])
        scores.masked_fill_(source[None, None] > rows[None, begin:end, None], torch.finfo(dtype).min)
        a = scores.softmax(-1, dtype=torch.float32).to(dtype).float()
        a[..., rows] = native_history[:, begin:end]
        yield begin, end, a


def source_groups(trace, rows, device):
    source = torch.arange(len(trace["token_ids"]), device=device)
    special = torch.as_tensor(trace["special_mask"], device=device)
    material = torch.as_tensor(trace["evidence_mask"], device=device) & ~special
    start = int(trace["response_start"])
    window = json.loads(str(trace["settings"]))["local_window"]
    lag = rows[:, None] - source[None]
    ordinary = ~special[None] & (lag >= 0)
    return torch.stack((
        special[None] & (lag >= 0), ordinary & material[None],
        ordinary & (source < start)[None] & ~material[None],
        ordinary & (source >= start)[None] & (lag > window),
        ordinary & (source >= start)[None] & (lag > 0) & (lag <= window),
        ordinary & (source >= start)[None] & (lag == 0),
    ), -1)


def display_edges(native, material, eligible, k):
    """A bounded display only: computation and statistics use every edge."""
    size = min(k, material.shape[-1])
    score = material.abs().masked_fill(~eligible[None], -1)
    strength, position = score.topk(size, -1)
    valid = strength > 0
    return (torch.where(valid, position, -1).int().cpu().numpy(),
            torch.where(valid, native.gather(-1, position), 0).cpu().numpy(),
            torch.where(valid, material.gather(-1, position), 0).cpu().numpy())


@torch.inference_mode()
def propagate(path, weights, *, chunk=16, top_k=4, mlp_rule="symmetric", progress=None):
    """Follow ordinary-material boundary messages through every response layer.

    Only one material vector [response-row, hidden] is propagated. Native minus
    this vector is the unassigned remainder, including other sources and native
    rounding. This is not a statement that the remainder is a parametric prior.
    """
    path, device = Path(path), weights.device
    with np.load(path, allow_pickle=False) as archive:
        trace = dict(archive)
    layers, heads, r = trace["head_margin"].shape
    cfg = weights.config
    if (layers, heads) != (cfg["num_hidden_layers"], cfg["num_attention_heads"]):
        raise ValueError("checkpoint layers/heads differ from capture")
    dim, start = cfg["hidden_size"], int(trace["response_start"])
    rows = torch.as_tensor(trace["row_position"], device=device)
    special = torch.as_tensor(trace["special_mask"], device=device)
    roots = torch.as_tensor(trace["evidence_mask"], device=device) & ~special
    carrier = rows >= start
    source = torch.arange(len(special), device=device)
    result = {"lineage_schema": np.array(SCHEMA), "row_position": trace["row_position"],
              "token_ids": trace["token_ids"], "response_start": trace["response_start"],
              "source_group_names": np.array(GROUPS), "mlp_rule": np.array(mlp_rule),
              "labels_used": np.array(False), "material_roots": roots.cpu().numpy(),
              "source_margin": np.zeros((layers, heads, r, len(GROUPS)), np.float32)}
    for name in ("material_history_margin", "material_self_margin", "material_special_margin", "carrier_signed", "carrier_absolute"):
        result[name] = np.zeros((layers, heads, r), np.float32)
    result["material_residual_margin"] = np.zeros((layers + 1, r), np.float32)
    result["material_mlp_margin"] = np.zeros((layers, r), np.float32)
    for kind in ("relay", "direct"):
        for name in ("position", "native", "material"):
            shape = (layers, heads, r, min(top_k, len(special)))
            result[f"top_{kind}_{name}"] = np.full(shape, -1 if name == "position" else 0,
                                                  np.int32 if name == "position" else np.float32)
    errors = np.zeros((layers, 3), np.float32)
    p = torch.zeros((r, dim), device=device)
    with np.load(path.with_suffix(".states.npz")) as states, np.load(path.with_suffix(".qk.npz")) as qk, np.load(path.with_suffix(".history.npz")) as history:
        direction, runners = readout_directions(trace, states, weights, chunk)
        result["readout_runner_id"] = runners
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            names = {"input_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
                     "value": "self_attn.v_proj.weight", "output": "self_attn.o_proj.weight",
                     "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
            w = {name: weights.get(prefix + key) for name, key in names.items()}
            reference = torch.as_tensor(states[f"residual_{layer}"], device=device)
            value = torch.as_tensor(states[f"value_{layer}"], device=device).float()
            kv, _, hd = value.shape
            part_v = F.linear(norm_component(p, reference, w["input_norm"], cfg["rms_norm_eps"]), w["value"].float())
            part_v = part_v.reshape(r, kv, hd).permute(1, 0, 2).repeat_interleave(heads // kv, 0)
            # P-1 is a prompt boundary source, never a response carrier.
            part_v[:, ~carrier] = 0
            value = value.repeat_interleave(heads // kv, 0)
            output = w["output"].float()
            projected = (direction @ output).reshape(r, heads, hd).permute(1, 0, 2)
            p_attention = torch.zeros_like(p)
            native_attention = torch.zeros_like(p)
            for begin, end, a in attention_chunks(trace, qk, history, layer, device, chunk):
                q = rows[begin:end]
                native_edge = a * (projected[:, begin:end] @ value.transpose(-1, -2))
                lineage_edge = a[..., rows] * (projected[:, begin:end] @ part_v.transpose(-1, -2))
                groups = source_groups(trace, q, device)
                result["source_margin"][layer, :, begin:end] = torch.einsum("hqs,qsg->hqg", native_edge, groups.float()).cpu().numpy()
                for name, mask in (("history", groups[..., rows, 3] | groups[..., rows, 4]),
                                   ("self", groups[..., rows, 5]), ("special", groups[..., rows, 0])):
                    result[f"material_{name}_margin"][layer, :, begin:end] = (lineage_edge * mask[None]).sum(-1).cpu().numpy()
                ordinary_past = carrier[None] & ~special[rows][None] & (rows[None] < q[:, None])
                valid_target = (q + 1 < len(special)) & ~special[q]
                valid_target &= ~special[(q + 1).clamp_max(len(special) - 1)]
                outgoing = lineage_edge * ordinary_past[None] * valid_target[None, :, None]
                result["carrier_signed"][layer] += outgoing.sum(1).cpu().numpy()
                result["carrier_absolute"][layer] += outgoing.abs().sum(1).cpu().numpy()
                full_lineage = torch.zeros_like(native_edge)
                full_lineage[..., rows] = lineage_edge
                relay = (source >= start)[None] & ~special[None] & (source[None] < q[:, None])
                for kind, component, eligible in (("relay", full_lineage, relay), ("direct", native_edge, roots[None].expand(len(q), -1))):
                    edge = display_edges(native_edge, component, eligible, top_k)
                    for name, data in zip(("position", "native", "material"), edge):
                        result[f"top_{kind}_{name}"][layer, :, begin:end] = data
                code = (a * roots[None, None]) @ value + a[..., rows] @ part_v
                p_attention[begin:end] = F.linear(code.permute(1, 0, 2).reshape(end - begin, -1), output)
                native_code = (a @ value).permute(1, 0, 2).reshape(end - begin, -1)
                native_attention[begin:end] = F.linear(native_code, output)
            native_write = torch.as_tensor(states[f"attention_{layer}"], device=device)
            dtype = getattr(torch, str(qk[f"dtype_{layer}"]))
            post = (reference.to(dtype) + native_write.to(dtype)).float()
            p_post = p + p_attention
            p_mlp, replay_mlp = swiglu_component(p_post, post, w, cfg["rms_norm_eps"], mlp_rule)
            native_mlp = torch.as_tensor(states[f"mlp_{layer}"], device=device)
            errors[layer, 0] = float((native_attention - native_write).norm() / native_write.norm().clamp_min(1e-12))
            errors[layer, 1] = float((replay_mlp - native_mlp).norm() / native_mlp.norm().clamp_min(1e-12))
            expected = torch.as_tensor(trace["head_margin"][layer, :, :-1], device=device)
            replay = torch.as_tensor(result["source_margin"][layer, :, :-1].sum(-1), device=device)
            errors[layer, 2] = float((replay - expected).norm() / expected.norm().clamp_min(1e-12))
            if not np.isfinite(errors[layer]).all() or errors[layer].max() > .05:
                raise ValueError(f"layer {layer}: native replay mismatch {errors[layer].tolist()}; verify the capture checkpoint/dtype")
            p = p_post + p_mlp
            result["material_mlp_margin"][layer] = (p_mlp * direction).sum(-1).cpu().numpy()
            result["material_residual_margin"][layer + 1] = (p * direction).sum(-1).cpu().numpy()
            if progress:
                progress(layer + 1, layers)
            del w, part_v, value, projected
    result["replay_relative_error"] = errors
    result["replay_error_names"] = np.array(("attention", "mlp", "head_readout"))
    result["remainder_history_margin"] = result["source_margin"][..., 3:5].sum(-1) - result["material_history_margin"]
    result["final_material_margin"] = result["material_residual_margin"][-1]
    result["final_remainder_margin"] = trace["residual_margin"][-1] - result["final_material_margin"]
    result.update(scores(trace, result))
    return result


def scores(trace, lineage):
    """One fixed hypothesis score and controls; no fitting, no sign selection."""
    final = trace["residual_margin"][-1]
    material = lineage["final_material_margin"]
    direct = lineage["source_margin"][..., 1].sum((0, 1))

    def deficit(component):
        total = np.abs(component) + np.abs(final - component)
        result = np.divide(-component, total, out=np.full_like(component, np.nan), where=total > 0)
        if not lineage["material_roots"].any():
            result[:] = np.nan
        return result

    return {"score_material_deficit": deficit(material), "score_direct_deficit": deficit(direct),
            "score_negative_logprob": -trace["predictor_logprob"],
            "score_position": np.arange(len(final), dtype=np.float32)}
