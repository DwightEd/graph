"""One native forward: ordinary-token reads, full history edges and real writes.

Labels never enter this module. Special tokens stay in the model input; their
mass is measured separately and excluded from the ordinary-token estimands.
History matrices are saved one layer at a time, allowing new offline horizons.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

GROUPS = ("special", "evidence", "other_prompt", "history_far", "history_local", "self")
SCALES = (4, 10, 32, 0)  # 0 means untruncated
TOP_GROUPS = ("ordinary_past", "evidence", "history")
SCHEMA = 3


@dataclass(frozen=True)
class AuditConfig:
    local_window: int = 10
    top_k: int = 8
    query_chunk: int = 32
    save_states: bool = True
    full_attention: bool = False

    def __post_init__(self):
        if min(self.local_window, self.top_k, self.query_chunk) < 1:
            raise ValueError("window, top_k and query_chunk must be positive")


def write_array(archive, name, value):
    """Stream an array into an ordinary NPZ; no pickles or whole-run buffers."""
    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def special_token_mask(tokenizer, token_ids, extra_ids=()):
    ids = set(getattr(tokenizer, "all_special_ids", ())) | set(extra_ids)
    ids.update(int(i) for i,token in getattr(tokenizer, 'added_tokens_decoder', {}).items()
               if getattr(token, 'special', False))
    return np.isin(token_ids, list(ids))


def divide(a, b):
    return torch.where(b > 0, a / b.clamp_min(1e-30), torch.nan)


class AuditObserver:
    def __init__(self, layers, heads, token_ids, start, evidence, special, unit_id,
                 config, history_archive=None, state_archive=None, full_archive=None,
                 progress=None):
        self.ids = np.asarray(token_ids)
        self.start, self.n = int(start), len(token_ids)
        if not 1 <= self.start < self.n:
            raise ValueError("a nonempty prompt and response are required")
        self.rows = np.arange(start - 1, self.n)
        self.evidence, self.special = np.asarray(evidence, bool), np.asarray(special, bool)
        self.unit_id = np.asarray(unit_id, int)
        if any(x.shape != (self.n,) for x in (self.evidence, self.special, self.unit_id)):
            raise ValueError("source masks must cover the original token sequence")
        if self.evidence[start:].any():
            raise ValueError("evidence identifies prompt positions only")
        self.units = max(0, int(self.unit_id.max()) + 1)
        self.config, self.layers, self.heads = config, layers, heads
        self.history_archive, self.state_archive = history_archive, state_archive
        self.full_archive, self.progress = full_archive, progress
        self.qk_archive = None
        shape = (layers, heads, len(self.rows))
        self.data = {name: np.full(shape, np.nan, np.float32) for name in
                     ("entropy", "entropy_normalized", "top1", "change_tv", "repeat_mass",
                      "head_norm", "ordinary_mass", "message_ordinary_mass",
                      "target_literal_evidence_mass", "target_literal_history_mass")}
        for name in ("mass", "message_mass"):
            self.data[name] = np.zeros((*shape, len(GROUPS)), np.float32)
        for name in ("distance", "message_distance", "distance_with_special"):
            self.data[name] = np.full((*shape, len(SCALES)), np.nan, np.float32)
        self.data["unit_mass"] = np.zeros((*shape, self.units), np.float32)
        top_shape = (*shape, len(TOP_GROUPS), config.top_k)
        for name in ("top_position", "top_message_position"):
            self.data[name] = np.full(top_shape, -1, np.int32)
        for name in ("top_attention", "top_message_strength"):
            self.data[name] = np.zeros(top_shape, np.float32)
        self.layer = -1
        self.seen = np.zeros((layers, len(self.rows)), bool)
        self.mass_error = self.future_leak = 0.

    def flush_layer(self):
        if self.layer < 0:
            return
        if self.history_archive is not None:
            write_array(self.history_archive, f"L{self.layer}", self.history)
        if self.full_archive is not None:
            write_array(self.full_archive, f"L{self.layer}", self.full)
        if self.state_archive is not None:
            write_array(self.state_archive, f"head_{self.layer}", self.head_code)
        if self.progress is not None:
            self.progress('forward', self.layer + 1, self.layers)

    def begin_layer(self, layer):
        if layer == self.layer:
            return
        self.flush_layer()
        self.layer, self.previous = layer, None
        self.history = np.zeros((self.heads, len(self.rows), len(self.rows)), np.float32)
        if self.full_archive is not None:
            self.full = np.zeros((self.heads, len(self.rows), self.n), np.float32)

    def observe_layer_input(self, layer, hidden):
        self.begin_layer(layer)
        if self.state_archive is not None:
            write_array(self.state_archive, f"residual_{layer}", hidden[0, self.rows].float().cpu().numpy())

    def observe_qk(self, layer, query, key, scaling):
        """Retain post-RoPE Q/K, enough to reconstruct every response source row.

        Queries are response predictors/carriers; keys cover the full sequence.
        This is much smaller than keeping every dense [H,response,all-source]
        matrix. Original compute dtype and scale are explicit, including GQA.
        """
        if self.qk_archive is not None:
            write_array(self.qk_archive, f'query_{layer}', query[0,:,self.rows].float().cpu().numpy())
            write_array(self.qk_archive, f'key_{layer}', key[0].float().cpu().numpy())
            write_array(self.qk_archive, f'scale_{layer}', np.array(scaling))
            write_array(self.qk_archive, f'dtype_{layer}', np.array(str(query.dtype).split('.')[-1]))

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        if begin + probability.shape[2] <= self.start - 1:
            return
        self.begin_layer(layer)
        if not hasattr(self, "norm_layer") or self.norm_layer != layer:
            h, d = value.shape[1], value.shape[-1]
            blocks = output_weight.detach().float().reshape(-1, h, d).permute(1, 0, 2)
            self.gram = blocks.transpose(-1, -2) @ blocks
            v = value[0].detach().float()
            self.norm = torch.einsum("hsd,hde,hse->hs", v, self.gram, v).clamp_min(0).sqrt()
            self.norm_layer = layer
            self.head_code = np.zeros((h, len(self.rows), d), np.float32)
            if self.config.save_states and self.state_archive is not None:
                # Preserve every actual per-head V; GQA repetitions compress well.
                write_array(self.state_archive, f"value_{layer}", v[::getattr(self, 'kv_stride', 1)].cpu().numpy())
        self.observe_rows(layer, begin, probability[0], self.norm)

    @torch.no_grad()
    def observe_rows(self, layer, begin, probability, value_norm):
        first, stop = max(begin, self.start - 1), min(begin + probability.shape[1], self.n)
        if stop <= first:
            return
        self.begin_layer(layer)
        saved = slice(first - self.start + 1, stop - self.start + 1)
        if self.seen[layer, saved].any():
            raise ValueError("duplicate native rows")
        self.seen[layer, saved] = True
        a = probability[:, first - begin:stop - begin].detach().float()
        if a.shape[-1] != self.n or value_norm.shape != (self.heads, self.n):
            raise ValueError("uncensored, complete source rows and value norms are required")
        device = a.device
        q = torch.arange(first, stop, device=device)
        s = torch.arange(self.n, device=device)
        lag = q[:, None] - s[None]
        legal = lag >= 0
        special = torch.as_tensor(self.special, device=device)
        evidence = torch.as_tensor(self.evidence, device=device) & ~special
        ordinary = legal & ~special[None]
        prompt = s < self.start
        groups = torch.stack((legal & special, legal & evidence,
                              legal & prompt & ~special & ~evidence,
                              (lag > self.config.local_window) & ~prompt & ~special,
                              (lag > 0) & (lag <= self.config.local_window) & ~prompt & ~special,
                              (lag == 0) & ~prompt & ~special)).float()
        self.mass_error = max(self.mass_error, float((a.sum(-1) - 1).abs().max()))
        self.future_leak = max(self.future_leak, float(a.masked_fill(legal[None], 0).abs().max()))
        transport = a * value_norm.to(device)[:, None]
        clean = a * ordinary[None]
        total = clean.sum(-1, keepdim=True)
        normalized = divide(clean, total)
        for prefix, weights in (("", a), ("message_", transport)):
            self.data[prefix + "mass"][layer, :, saved] = torch.einsum("hqs,gqs->hqg", weights, groups).cpu().numpy()
            retained = (weights * ordinary[None]).sum(-1, keepdim=True)
            self.data[prefix + "ordinary_mass"][layer, :, saved] = retained[..., 0].cpu().numpy()
            wn = divide(weights * ordinary[None], retained)
            for k, scale in enumerate(SCALES):
                dist = lag.clamp_min(0).float()
                if scale:
                    dist = dist.clamp_max(scale)
                self.data[prefix + "distance"][layer, :, saved, k] = (wn * dist).sum(-1).cpu().numpy()
                if not prefix:
                    self.data['distance_with_special'][layer, :, saved, k] = (a * dist).sum(-1).cpu().numpy()
        entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(-1)
        choices = ordinary.sum(-1).float()
        norm_entropy = torch.where(choices > 1, entropy / choices.clamp_min(2).log(),
                                   torch.where((choices == 1) & (total[..., 0] > 0), 0., torch.nan))
        ids = torch.as_tensor(self.ids, device=device)
        repeated = (ids[s][None] == ids[q][:, None]) & (lag > 0) & ordinary
        target_equal = (ids[s][None] == ids[(q+1).clamp_max(self.n-1)][:,None]) & ordinary
        for name, mask in (("evidence", target_equal & evidence),
                           ("history", target_equal & ~prompt & (lag > 0))):
            strength = (a * mask[None]).sum(-1)
            strength[:, q == self.n-1] = torch.nan
            self.data[f'target_literal_{name}_mass'][layer, :, saved] = strength.cpu().numpy()
        for name, array in (("entropy", entropy), ("entropy_normalized", norm_entropy),
                            ("top1", normalized.amax(-1)),
                            ("repeat_mass", (clean * repeated[None]).sum(-1))):
            self.data[name][layer, :, saved] = array.cpu().numpy()
        previous = torch.cat((self.previous[:, None] if self.previous is not None
                              else torch.full_like(normalized[:, :1], torch.nan), normalized[:, :-1]), dim=1)
        self.data["change_tv"][layer, :, saved] = (.5 * (normalized - previous).abs().sum(-1)).cpu().numpy()
        self.previous = normalized[:, -1].clone()
        if self.units:
            unit = torch.as_tensor(self.unit_id, device=device)
            mask = (unit >= 0) & ~special
            unit_mass = torch.zeros((*a.shape[:2], self.units), device=device)
            unit_mass.scatter_add_(-1, unit[mask][None, None].expand(*a.shape[:2], -1), a[..., mask])
            self.data['unit_mass'][layer, :, saved] = unit_mass.cpu().numpy()
        for g, mask in enumerate((ordinary & (lag > 0), legal & evidence, ordinary & ~prompt & (lag > 0))):
            k = min(self.config.top_k, self.n)
            for prefix, weights in (("", a), ("message_", transport)):
                strength, position = weights.masked_fill(~mask[None], -1).topk(k, dim=-1)
                position = position.masked_fill(strength <= 0, -1)
                self.data["top_" + prefix + "position"][layer, :, saved, g, :k] = position.cpu().numpy()
                name = "top_message_strength" if prefix else "top_attention"
                self.data[name][layer, :, saved, g, :k] = strength.clamp_min(0).cpu().numpy()
        self.history[:, saved] = a[..., self.start - 1:].cpu().numpy()
        if self.full_archive is not None:
            self.full[:, saved] = a.cpu().numpy()

    def observe_head_output(self, layer, begin, codes):
        first, stop = max(begin, self.start - 1), min(begin + codes.shape[2], self.n)
        if stop <= first:
            return
        saved = slice(first - self.start + 1, stop - self.start + 1)
        z = codes[0, :, first - begin:stop - begin].detach().float()
        self.head_code[:, saved] = z.cpu().numpy()
        self.data["head_norm"][layer, :, saved] = torch.einsum("hqd,hde,hqe->hq", z, self.gram, z).clamp_min(0).sqrt().cpu().numpy()

    def observe_attention_write(self, layer, value):
        if self.state_archive is not None:
            write_array(self.state_archive, f"attention_{layer}", value[0, self.rows].float().cpu().numpy())

    def observe_mlp_write(self, layer, value):
        if self.state_archive is not None:
            write_array(self.state_archive, f"mlp_{layer}", value[0, self.rows].float().cpu().numpy())

    def finish(self):
        self.flush_layer()
        if not self.seen.all() or self.mass_error > .02 or self.future_leak > 1e-7:
            raise ValueError("incomplete or invalid native attention capture")
        return {**self.data, "row_position": self.rows, "token_ids": self.ids,
                "response_start": np.array(self.start), "special_mask": self.special,
                "evidence_mask": self.evidence & ~self.special, "source_unit_id": self.unit_id,
                "group_names": np.array(GROUPS), "distance_scales": np.array(SCALES),
                "top_group_names": np.array(TOP_GROUPS), "audit_schema": np.array(SCHEMA),
                "labels_used_for_capture": np.array(False),
                "row_mass_error": np.array(self.mass_error), "future_leak": np.array(self.future_leak)}


@torch.inference_mode()
def readout_accounting(model, final, ids, rows, states_path, heads, progress=None):
    """Exact frozen-denominator observed-vs-runner accounting, not truth attribution."""
    device, count = final.device, len(rows) - 1
    directions, margins, entropy, logprob, runners = [], [], [], [], []
    for begin in range(0, count, 16):
        q = rows[begin:begin + 16][:count - begin]
        raw = final[0, q].float()
        logits = model.lm_head(model.model.norm(final[0, q])).float()
        target = ids[q + 1]
        target_logits = logits.gather(1, target[:, None])[:, 0]
        lse = logits.logsumexp(-1)
        entropy.extend((lse - (logits.softmax(-1) * logits).sum(-1)).cpu().tolist())
        logprob.extend((target_logits - lse).cpu().tolist())
        logits.scatter_(1, target[:, None], -torch.inf)
        runner = logits.argmax(-1)
        runners.extend(runner.cpu().tolist())
        margins.extend((target_logits - logits.gather(1, runner[:, None])[:, 0]).cpu().tolist())
        norm = model.model.norm
        denominator = (raw.square().mean(-1) + norm.variance_epsilon).sqrt()
        d = (model.lm_head.weight[target].float() - model.lm_head.weight[runner].float())
        directions.append((d * norm.weight.float() / denominator[:, None]).cpu())
    direction = torch.cat(directions).to(device)
    layers = len(model.model.layers)
    result = {"head_margin": np.full((layers, heads, count + 1), np.nan, np.float32),
              "residual_margin": np.full((layers + 1, count + 1), np.nan, np.float32),
              "attention_margin": np.full((layers, count + 1), np.nan, np.float32),
              "mlp_margin": np.full((layers, count + 1), np.nan, np.float32)}
    with np.load(states_path, allow_pickle=False) as states:
        for l, layer in enumerate(model.model.layers):
            for field in ("residual", "attention", "mlp"):
                v = torch.as_tensor(states[f"{field}_{l}"][:count], device=device)
                result[field + "_margin"][l, :count] = (v * direction).sum(-1).cpu().numpy()
            weight = layer.self_attn.o_proj.weight.float()
            block = weight.reshape(weight.shape[0], heads, weight.shape[1] // heads)
            d = torch.einsum("qd,dhe->hqe", direction, block)
            code = torch.as_tensor(states[f"head_{l}"][:, :count], device=device)
            result["head_margin"][l, :, :count] = (d * code).sum(-1).cpu().numpy()
            if progress is not None:
                progress('readout', l+1, layers)
    result["residual_margin"][-1, :count] = (final[0, rows[:count]].float() * direction).sum(-1).cpu().numpy()
    result["rounding_margin"] = np.diff(result["residual_margin"], axis=0) - result["attention_margin"] - result["mlp_margin"]
    result['head_sum_rounding'] = result['attention_margin'] - result['head_margin'].sum(1)
    result.update(predictor_entropy=np.r_[entropy, np.nan].astype(np.float32),
                  predictor_logprob=np.r_[logprob, np.nan].astype(np.float32),
                  observed_margin=np.r_[margins, np.nan].astype(np.float32),
                  readout_runner_id=np.array(runners, np.int64))
    result['final_readout_rounding'] = result['observed_margin'] - result['residual_margin'][-1]
    return result


@torch.inference_mode()
def capture_audit(model, token_ids, start, evidence, special, unit_id, path,
                  config=AuditConfig(), progress=None):
    from contextlib import ExitStack
    from experiments.common.llama_message_intervention import (
        VALIDATED_ATTRIBUTE, forward_layers, validate_manual_forward,
    )
    model.eval()
    ids = torch.as_tensor(token_ids, device=model.get_input_embeddings().weight.device, dtype=torch.long)
    if not getattr(model, VALIDATED_ATTRIBUTE, False):
        validate_manual_forward(model, ids)
    path = Path(path)
    history_path, states_path = path.with_suffix(".history.tmp.npz"), path.with_suffix(".states.tmp.npz")
    full_path = path.with_suffix(".attention.tmp.npz")
    qk_path = path.with_suffix('.qk.tmp.npz')
    with ExitStack() as stack:
        def archive(p):
            return stack.enter_context(ZipFile(p, "w", ZIP_DEFLATED, compresslevel=1, allowZip64=True))
        observer = AuditObserver(len(model.model.layers), model.config.num_attention_heads, token_ids,
                                 start, evidence, special, unit_id, config, archive(history_path),
                                 archive(states_path), archive(full_path) if config.full_attention else None,
                                 progress)
        observer.kv_stride = model.config.num_attention_heads // model.config.num_key_value_heads
        observer.qk_archive = archive(qk_path)
        final = forward_layers(model, model.get_input_embeddings()(ids[None]), 0, observer=observer,
                               attention_query_chunk=config.query_chunk, apply_final_norm=False)
        result = observer.finish()
        if config.save_states:
            write_array(observer.state_archive, "final_residual", final[0, observer.rows].float().cpu().numpy())
    result.update(readout_accounting(model, final, ids, observer.rows, states_path, observer.heads, progress))
    result['value_head_stride'] = np.array(observer.kv_stride)
    history_path.replace(path.with_suffix(".history.npz"))
    qk_path.replace(path.with_suffix('.qk.npz'))
    if config.save_states:
        states_path.replace(path.with_suffix(".states.npz"))
    else:
        states_path.unlink()
    if config.full_attention:
        full_path.replace(path.with_suffix(".attention.npz"))
    return result


@torch.inference_mode()
def reconstruct_attention(path, layer, head, query_indices=None):
    """Model-free full-source reconstruction for any saved response-query head.

    Uses native saved Q/K, compute dtype, scaling and causal mask. Floating
    point kernels on another device may differ slightly; native history rows
    remain available as the exact stored reference. No missing edge imputation.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as trace:
        rows = trace['row_position']
    selected = np.arange(len(rows)) if query_indices is None else np.asarray(query_indices, int)
    with np.load(path.with_suffix('.qk.npz'), allow_pickle=False) as archive:
        q, k = archive[f'query_{layer}'], archive[f'key_{layer}']
        if not 0 <= head < len(q):
            raise ValueError('head outside captured axes')
        dtype = getattr(torch, str(archive[f'dtype_{layer}']))
        query = torch.as_tensor(q[head,selected], dtype=dtype)
        key = torch.as_tensor(k[head//(len(q)//len(k))], dtype=dtype)
        scores = (query @ key.T) * float(archive[f'scale_{layer}'])
    future = torch.arange(len(key))[None] > torch.as_tensor(rows[selected])[:,None]
    scores.masked_fill_(future, torch.finfo(dtype).min)
    return scores.softmax(-1,dtype=torch.float32).to(dtype).float().numpy()
