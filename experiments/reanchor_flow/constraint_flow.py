"""Aligned-condition message accounting and bounded value-path confirmation.

No hallucination labels, WAAD gate, head averaging, gradients or fitted detector.
The symmetric AV difference is exact algebra; it is not causal mediation.
Only confirm_paths performs interventions, on paths frozen before those runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from experiments.common.llama_message_intervention import (
    MessageGate, forward_layers, validate_manual_forward, VALIDATED_ATTRIBUTE,
)


@dataclass(frozen=True)
class ConstraintPair:
    """Two aligned, valid prefixes; candidates name ONE fixed next-token contrast.

    Both rows end at the same query. No token after that query is an input.
    Semantic validity is supplied by a controlled generator or external review,
    not inferred from a model response or a hallucination label.
    """
    token_ids: np.ndarray  # [2, N], condition 0 then condition 1
    response_start: int
    negative_token_id: int
    positive_token_id: int

    def check(self):
        ids = np.asarray(self.token_ids)
        if ids.ndim != 2 or ids.shape[0] != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token_ids must be aligned integer prefixes [2,N]")
        if not 0 < self.response_start <= ids.shape[1]:
            raise ValueError("response_start must leave a prompt and may have an empty shared prefix")
        if not np.array_equal(ids[0, self.response_start:], ids[1, self.response_start:]):
            raise ValueError("this protocol requires an identical, valid response prefix")
        if self.positive_token_id == self.negative_token_id:
            raise ValueError("the registered next-token candidates must differ")
        return self


def symmetric_av(a0, a1, v0, v1):
    """Return full head codes: mean(A) delta(V), delta(A) mean(V)."""
    return ((a0 + a1) * .5) @ (v1 - v0), (a1 - a0) @ ((v0 + v1) * .5)


def _numpy(x):
    return x.detach().float().cpu().numpy()


class ConstraintObserver:
    """Stream two native conditions together; keep every head and source at q.

    Full vectors are retained at all shared-prefix positions and registered
    roots. Full source rows at q and root-to-prefix coefficients allow the
    selected two-edge vectors to be reconstructed without another capture.
    """
    def __init__(self, model, pair, roots):
        self.model, self.pair = model, pair
        self.q = pair.token_ids.shape[1] - 1
        self.rows = np.arange(pair.response_start - 1, self.q + 1)
        self.roots = np.asarray(roots, dtype=int)
        self.states = np.union1d(self.rows, self.roots)
        device = model.get_input_embeddings().weight.device
        self.row_ids = torch.as_tensor(self.rows, device=device)
        self.state_ids = torch.as_tensor(self.states, device=device)
        self.root_ids = torch.as_tensor(self.roots, device=device)
        self.heads = model.config.num_attention_heads
        self.dim = model.model.layers[0].self_attn.head_dim
        self.direction = (model.lm_head.weight[pair.positive_token_id].float()
                          - model.lm_head.weight[pair.negative_token_id].float())
        self.direction *= model.model.norm.weight.float()
        l, h, r, d = len(model.model.layers), self.heads, len(self.rows), self.dim
        k, e, width = len(self.states), len(self.roots), len(self.direction)
        self.groups = model.model.layers[0].self_attn.num_key_value_groups
        kv = h // self.groups
        self.arrays = {
            "row_position": self.rows, "state_position": self.states, "root_position": self.roots,
            "residual_pair": np.empty((l + 1, 2, k, width), np.float32),
            "target_attention_pair": np.empty((l, 2, h, self.q + 1), np.float32),
            "root_attention_pair": np.empty((l, 2, h, r, e), np.float32),
            "query_to_kv": np.arange(h) // self.groups,
            "root_value_pair": np.empty((l, 2, kv, e, d), np.float32),
            "response_value_pair": np.empty((l, 2, kv, r, d), np.float32),
        }
        for name in ("content", "routing", "native"):
            self.arrays[f"head_{name}_code"] = np.empty((l, h, r, d), np.float32)
            self.arrays[f"head_{name}_projection"] = np.empty((l, h, r), np.float32)
        for name in ("content", "routing"):
            self.arrays[f"source_{name}_projection"] = np.empty((l, h, self.q + 1), np.float32)
        for name in ("attention", "post_attention", "mlp"):
            self.arrays[f"{name}_delta"] = np.empty((l, r, width), np.float32)
        self.layer = -1

    def observe_layer_input(self, layer, hidden):
        self.arrays["residual_pair"][layer] = _numpy(hidden.index_select(1, self.state_ids))
        self.input = hidden.index_select(1, self.row_ids)
        weight = self.model.model.layers[layer].self_attn.o_proj.weight
        blocks = weight.float().reshape(-1, self.heads, self.dim)
        self.beta = torch.einsum("d,dhk->hk", self.direction, blocks)

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        first, stop = max(begin, int(self.rows[0])), min(begin + probability.shape[2], self.q + 1)
        if stop <= first:
            return
        saved = slice(first - self.rows[0], stop - self.rows[0])
        a = probability[:, :, first - begin:stop - begin].float()
        v = value.float()
        if layer != self.layer:
            # GQA shares V exactly. Store each KV head once, with its explicit map.
            self.arrays["root_value_pair"][layer] = _numpy(v[:, ::self.groups].index_select(2, self.root_ids))
            self.arrays["response_value_pair"][layer] = _numpy(v[:, ::self.groups].index_select(2, self.row_ids))
            self.layer = layer
        content, routing = symmetric_av(a[0], a[1], v[0], v[1])
        for name, code in (("content", content), ("routing", routing)):
            self.arrays[f"head_{name}_code"][layer, :, saved] = _numpy(code)
            self.arrays[f"head_{name}_projection"][layer, :, saved] = _numpy(
                (code * self.beta[:, None]).sum(-1))
        self.arrays["root_attention_pair"][layer, :, :, saved] = _numpy(a.index_select(3, self.root_ids))
        if first <= self.q < stop:
            aq = a[:, :, self.q - first]
            self.arrays["target_attention_pair"][layer] = _numpy(aq)
            score = (v * self.beta[None, :, None]).sum(-1)
            self.arrays["source_content_projection"][layer] = _numpy(
                (aq[0] + aq[1]) * .5 * (score[1] - score[0]))
            self.arrays["source_routing_projection"][layer] = _numpy(
                (aq[1] - aq[0]) * (score[0] + score[1]) * .5)

    def observe_head_output(self, layer, begin, output):
        first, stop = max(begin, int(self.rows[0])), min(begin + output.shape[2], self.q + 1)
        if stop <= first:
            return
        saved = slice(first - self.rows[0], stop - self.rows[0])
        code = output[1, :, first-begin:stop-begin].float() - output[0, :, first-begin:stop-begin].float()
        self.arrays["head_native_code"][layer, :, saved] = _numpy(code)
        self.arrays["head_native_projection"][layer, :, saved] = _numpy((code * self.beta[:, None]).sum(-1))

    def observe_attention_write(self, layer, write):
        selected = write.index_select(1, self.row_ids)
        self.post = self.input + selected
        self.arrays["attention_delta"][layer] = _numpy(selected[1].float() - selected[0].float())
        self.arrays["post_attention_delta"][layer] = _numpy(self.post[1].float() - self.post[0].float())

    def observe_mlp_write(self, layer, write):
        selected = write.index_select(1, self.row_ids)
        self.arrays["mlp_delta"][layer] = _numpy(selected[1].float() - selected[0].float())

    def finish(self, raw_final):
        a = self.arrays
        a["residual_pair"][-1] = _numpy(raw_final.index_select(1, self.state_ids))
        raw = raw_final[:, self.q].double()
        inverse_rms = (raw.square().mean(-1) + self.model.model.norm.variance_epsilon).rsqrt()
        scale = float(inverse_rms.mean())
        direction = _numpy(self.direction).astype(np.float64)
        u = scale * direction
        logits = self.model.lm_head(self.model.model.norm(raw_final[:, self.q])).float()
        pair = self.pair
        margin = logits[:, pair.positive_token_id] - logits[:, pair.negative_token_id]
        a.update(inverse_rms=_numpy(inverse_rms), readout_direction=u,
                 margin_pair=_numpy(margin), target_margin_delta=np.array(float(margin[1] - margin[0])),
                 entropy_pair=_numpy(logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)))
        for key in tuple(a):
            if key.endswith("_projection"):
                a[key] *= scale
        delta = a["residual_pair"][:, 1].astype(float) - a["residual_pair"][:, 0]
        a["state_projection"] = np.einsum("lkd,d->lk", delta, u)
        for name in ("attention", "post_attention", "mlp"):
            a[f"{name}_projection"] = np.einsum("lrd,d->lr", a[f"{name}_delta"], u)
        response_slots = np.searchsorted(self.states, self.rows)
        state = a["state_projection"][:, response_slots]
        a["av_rounding"] = a["head_native_projection"] - a["head_content_projection"] - a["head_routing_projection"]
        a["head_sum_rounding"] = a["attention_projection"] - a["head_native_projection"].sum(1)
        a["residual_add_rounding"] = np.diff(state, axis=0) - a["attention_projection"] - a["mlp_projection"]
        a["source_sum_rounding"] = (a["head_native_projection"][..., -1]
                                     - a["source_content_projection"].sum(-1)
                                     - a["source_routing_projection"].sum(-1))
        normalization = float((inverse_rms[1] - inverse_rms[0]) * (raw.mean(0) * torch.as_tensor(direction, device=raw.device)).sum())
        a["normalization_term"] = np.array(normalization)
        # Each ledger entry contributes ONCE at q; carrier projections are not added again.
        terms = np.array([state[0, -1], a["source_content_projection"].sum(),
                          a["source_routing_projection"].sum(), a["mlp_projection"][:, -1].sum(), normalization])
        a["ledger_terms"] = terms
        a["ledger_names"] = np.asarray(("input", "content", "routing", "MLP", "final RMSNorm"))
        a["ledger_rounding"] = a["target_margin_delta"] - terms.sum()
        a["final_norm_rounding"] = a["target_margin_delta"] - state[-1, -1] - normalization
        return a


@torch.inference_mode()
def capture_constraint_pair(model, pair: ConstraintPair, *, query_chunk=8, roots=()):
    pair.check()
    model.eval()
    device = model.get_input_embeddings().weight.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    ids = torch.as_tensor(pair.token_ids, dtype=torch.long, device=device)
    validation = not getattr(model, VALIDATED_ATTRIBUTE, False)
    if validation:
        validate_manual_forward(model, ids[0])
    edited = np.flatnonzero(pair.token_ids[0] != pair.token_ids[1])
    observer = ConstraintObserver(model, pair, np.union1d(edited, roots))
    final = forward_layers(model, model.get_input_embeddings()(ids), 0, observer=observer,
                           attention_query_chunk=query_chunk, apply_final_norm=False)
    result = observer.finish(final)
    result.update(constraint_schema=np.array(1), token_ids=pair.token_ids,
                  response_start=np.array(pair.response_start), query=np.array(ids.shape[1]-1),
                  edited_position=edited, valid_prefix_end=np.array(ids.shape[1]-1),
                  valid_query_mask=np.ones(len(observer.rows), dtype=bool),
                  negative_token_id=np.array(pair.negative_token_id),
                  positive_token_id=np.array(pair.positive_token_id),
                  capture_seconds=np.array(perf_counter()-started),
                  peak_cuda_bytes=np.array(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                  endpoint_conditions=np.array(2), native_forward_validation_included=np.array(validation))
    return result


PATH_COLUMNS = ("source", "write_layer", "write_head", "carrier", "read_layer", "read_head", "query")


@torch.inference_mode()
def select_constraint_paths(model, trace):
    """Select one response carrier from q's signed CONTENT response, plus neighbor.

    ALL layers/heads/sources are measured. Ranking is a hypothesis screen only.
    An earlier edited-source edge is chosen by vector-change magnitude to define
    a testable two-edge witness. Its importance must be tested, not assumed.
    """
    p, q = int(trace["response_start"]), int(trace["query"])
    score = np.abs(trace["source_content_projection"]).copy()
    score[..., :p], score[..., q:] = 0, 0
    score[0] = 0  # a write must exist in a strictly earlier layer
    if not np.any(score > 0) or not len(trace["edited_position"]):
        return np.empty((0, 7), int), np.asarray([], dtype="U16")
    k, h, b = map(int, np.unravel_index(score.argmax(), score.shape))
    slot = int(np.searchsorted(trace["row_position"], b))
    best = None
    for ell in range(k):
        values = trace["root_value_pair"][ell][:, trace["query_to_kv"]]
        av = trace["root_attention_pair"][ell, :, :, slot, :, None] * values
        code = av[1] - av[0]  # [H,E,d], all roots retained
        weight = model.model.layers[ell].self_attn.o_proj.weight
        blocks = weight.float().reshape(-1, code.shape[0], code.shape[-1]).permute(1, 0, 2)
        message = torch.einsum("hDd,hed->heD", blocks, torch.as_tensor(code, device=weight.device))
        strength = _numpy(message.norm(dim=-1))
        strength[:, ~np.isin(trace["root_position"], trace["edited_position"])] = 0
        wh, root = np.unravel_index(strength.argmax(), strength.shape)
        candidate = (float(strength[wh, root]), ell, int(wh), int(root))
        if best is None or candidate[0] > best[0]:
            best = candidate
    paths, roles = [], []
    if best is not None and best[0] > 0:
        _, ell, wh, root = best
        s = int(trace["root_position"][root])
        paths.append((s, ell, wh, b, k, h, q))
        roles.append("candidate")
        neighbor = b-1 if b-1 >= p else b+1
        if neighbor < q:
            # Change ONLY the carrier coordinate, retaining both heads and root.
            paths.append((s, ell, wh, neighbor, k, h, q))
            roles.append("neighbor_control")
    return np.asarray(paths, int).reshape(-1, 7), np.asarray(roles, dtype="U16")


class _ConfirmationBaseline:
    def __init__(self, paths):
        self.paths, self.attention, self.value = paths, {}, {}

    def observe_chunk(self, layer, begin, probability, value, weight):
        for i, (_, _, _, b, k, h, q) in enumerate(self.paths):
            if k == layer and begin <= q < begin + probability.shape[2]:
                self.attention[i] = probability[0, h, q-begin, b].clone()
                self.value[i] = value[0, h, b].clone()


@torch.inference_mode()
def confirm_paths(model, trace, paths, *, query_chunk=8):
    """Exact bounded content-channel path intervention, not all-route ablation.

    (1) Replace one incoming edge message by its condition-1 value, on condition 0.
    (2) Let that change evolve up to the deeper reader layer.
    (3) Copy ONLY the induced carrier V change into the native reader edge at q;
        its A stays native. Run the native suffix and measure the SAME candidates.

    Thus tau tests this value path. K/Q-mediated paths and other relays are
    excluded by design, not declared absent. No search or reranking after tau.
    """
    if not len(paths):
        return {"path_effect": np.empty(0), "path_value_delta": np.empty((0, trace["head_native_code"].shape[-1])),
                "confirmation_forward_calls": np.array(0), "confirmation_seconds": np.array(0.)}
    model.eval()
    started = perf_counter()
    p, q = int(trace["response_start"]), int(trace["query"])
    layers, heads = len(model.model.layers), model.config.num_attention_heads
    for s, ell, wh, b, k, h, receiver in paths:
        if not (0 <= s < p <= b < receiver == q and 0 <= ell < k < layers and 0 <= h < heads and 0 <= wh < heads):
            raise ValueError("a path must have s<p<=b<q and a strictly deeper reader")
    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(trace["token_ids"][0], device=device)[None]
    embedding = model.get_input_embeddings()(ids)
    observer = _ConfirmationBaseline(paths)
    native = forward_layers(model, embedding, 0, observer=observer,
                            attention_query_chunk=query_chunk, apply_final_norm=False)
    def margin(raw):
        logits = model.lm_head(model.model.norm(raw[:, q])).float()
        return float(logits[0, int(trace["positive_token_id"])] - logits[0, int(trace["negative_token_id"])])
    baseline = margin(native)
    effect, value_delta = [], []
    for index, (s, ell, wh, b, k, h, _) in enumerate(paths):
        root = int(np.flatnonzero(trace["root_position"] == s)[0])
        slot = int(np.flatnonzero(trace["row_position"] == b)[0])
        a = trace["root_attention_pair"][ell, :, wh, slot, root]
        v = trace["root_value_pair"][ell, :, int(trace["query_to_kv"][wh]), root]
        incoming = torch.as_tensor(a[1]*v[1] - a[0]*v[0], device=device)
        gate = MessageGate(0, head_output_patch={int(ell): {int(b): {int(wh): incoming}}})
        patched = forward_layers(model, embedding, 0, gate=gate, end_layer=int(k),
                                 attention_query_chunk=query_chunk, apply_final_norm=False)
        layer = model.model.layers[int(k)]
        attention = layer.self_attn
        normalized = layer.input_layernorm(patched)
        values = attention.v_proj(normalized).reshape(1, ids.shape[1], -1, attention.head_dim)
        kv_head = int(h) // attention.num_key_value_groups
        delta = values[0, int(b), kv_head].float() - observer.value[index].float()
        outgoing = observer.attention[index].float() * delta
        gate = MessageGate(0, head_output_patch={int(k): {q: {int(h): outgoing}}})
        final = forward_layers(model, embedding, 0, gate=gate, attention_query_chunk=query_chunk,
                               apply_final_norm=False)
        effect.append(margin(final) - baseline)
        value_delta.append(_numpy(delta))
    return {"path_effect": np.asarray(effect), "path_value_delta": np.asarray(value_delta),
            "confirmation_baseline_margin": np.array(baseline),
            "confirmation_capture_margin_difference": np.array(baseline - trace["margin_pair"][0]),
            "confirmation_forward_calls": np.array(1 + 2*len(paths)),
            "confirmation_seconds": np.array(perf_counter()-started)}
