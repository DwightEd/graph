"""Single-example frozen Llama replay for explicitly defined graph operations.

This is the measurement backend, not an evidence reader or an automatic search.
Input IDs are the actual model inputs (exclude the final prediction target).
Full-shape O projection is preserved for exact sham comparisons in bf16.
"""

import hashlib
import math
from dataclasses import dataclass

import torch

from route_graph.causal_contrast import message_gate_delta


@dataclass(frozen=True)
class CapturedValues:
    """Donor identity from this model, branch, layer and ordered key set."""

    values: torch.Tensor
    input_ids: tuple[int, ...]
    layer: int
    keys: tuple[int, ...]
    model_identity: int
    input_scale: tuple[tuple[int, ...], float] | None
    values_sha256: str


@dataclass(frozen=True)
class NativeGate:
    layer: int
    queries: tuple[int, ...]
    kind: str
    strength: float = 1.0
    keys: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()
    destination: tuple[int, ...] = ()
    # Exact branch-matched donor values, [key, KV head, head dimension].
    donor: CapturedValues | None = None
    expected_input_scale: tuple[tuple[int, ...], float] | None = None


def _values_digest(values):
    raw = values.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _positions(positions, length, name, allow_empty=False):
    if (
        (not positions and not allow_empty)
        or len(set(positions)) != len(positions)
        or any(type(p) is not int or not 0 <= p < length for p in positions)
    ):
        raise ValueError(f"invalid {name} positions")


@torch.no_grad()
def native_forward(model, input_ids, gates=(), input_scale=None, capture=()):
    """Replay gates, optional input embedding scaling, and requested V capture.

    input_scale is (positions, strength); capture contains (layer, keys) pairs.
    E/R use kind='route' and explicit key domains; paired R sets destination.
    kind='content' scales selected A*v, 'donor' replaces values on those edges,
    'mlp' scales the actual MLP branch. One attention gate plus one MLP gate per
    layer is allowed, enabling real joint interventions. All downstream layers
    recompute; no original future activations are restored.

    Returns native logits on their original device/dtype, CPU donor captures,
    and operator diagnostics. Callers own the logits lifetime and budget.
    Captured values and diagnostics describe the current receiving computation.
    """
    ids = tuple(input_ids)
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("expected nonempty token IDs")
    if (
        model.training
        or model.config.model_type != "llama"
        or model.config._attn_implementation != "eager"
        or model.dtype not in {torch.float16, torch.bfloat16, torch.float32}
    ):
        raise ValueError(
            "native audit requires eval-mode eager Llama in fp16/bf16/fp32"
        )
    layers = model.model.layers
    gates = tuple(gates)
    slots = [(g.layer, g.kind == "mlp") for g in gates]
    if len(set(slots)) != len(slots):
        raise ValueError("at most one attention and one MLP gate per layer")
    for gate in gates:
        if type(gate.layer) is not int or not 0 <= gate.layer < len(layers):
            raise ValueError("invalid gate layer")
        _positions(gate.queries, len(ids), "query")
        if (
            gate.kind not in {"route", "content", "donor", "mlp"}
            or not math.isfinite(gate.strength)
            or not 0 <= gate.strength <= 1
        ):
            raise ValueError("invalid gate kind or strength")
        if gate.kind == "mlp":
            if (
                gate.keys
                or gate.selected
                or gate.destination
                or gate.donor is not None
                or gate.expected_input_scale is not None
            ):
                raise ValueError("MLP gate cannot specify attention keys or donor")
            continue
        _positions(gate.keys, len(ids), "key")
        _positions(gate.selected, len(ids), "selected", allow_empty=True)
        _positions(gate.destination, len(ids), "destination", allow_empty=True)
        if not set(gate.selected + gate.destination) <= set(gate.keys):
            raise ValueError("selected/destination positions must belong to key domain")
        if gate.selected and min(gate.selected) > max(gate.queries):
            raise ValueError("no selected key is visible at any query")
        if gate.destination and (
            gate.kind != "route" or set(gate.selected) & set(gate.destination)
        ):
            raise ValueError("only route gates support disjoint destinations")
        if (gate.kind == "donor") != (gate.donor is not None):
            raise ValueError("only donor gates require donor values")
        if gate.kind != "donor" and gate.expected_input_scale is not None:
            raise ValueError("only donor gates declare expected input scaling")
        if gate.kind == "donor" and gate.strength != 1:
            raise ValueError("donor strength belongs to input scaling, not recipient")
        if gate.kind == "donor":
            donor = gate.donor
            if (
                not isinstance(donor, CapturedValues)
                or donor.input_ids != ids
                or donor.layer != gate.layer
                or donor.keys != gate.keys
                or donor.model_identity != id(model)
                or donor.input_scale != gate.expected_input_scale
                or not isinstance(donor.values, torch.Tensor)
                or not donor.values.is_floating_point()
                or donor.values.dtype != model.dtype
                or donor.values.shape
                != (
                    len(gate.keys),
                    model.config.num_key_value_heads,
                    layers[gate.layer].self_attn.head_dim,
                )
                or not torch.isfinite(donor.values).all()
                or donor.values_sha256 != _values_digest(donor.values)
            ):
                raise ValueError("donor provenance, dtype or values mismatch")

    capture = tuple((index, tuple(keys)) for index, keys in capture)
    if len({i for i, _ in capture}) != len(capture):
        raise ValueError("one V capture specification per layer")
    for index, keys in capture:
        if type(index) is not int or not 0 <= index < len(layers):
            raise ValueError("invalid capture layer")
        _positions(keys, len(ids), "capture")
    inputs = torch.tensor([ids], device=model.device)
    kwargs = {"input_ids": inputs}
    scaling_record = None
    if input_scale is not None:
        positions, strength = input_scale
        _positions(positions, len(ids), "input scaling")
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("invalid input scaling strength")
        embeddings = model.model.embed_tokens(inputs).clone()
        embeddings[0, list(positions)] *= strength
        kwargs = {"inputs_embeds": embeddings}
        scaling_record = (tuple(positions), float(strength))

    handles, captured, diagnostics = [], {}, []
    kv_heads = model.config.num_key_value_heads
    try:
        for index, keys in capture:

            def capture_v(module, args, output, index=index, keys=keys):
                values = (
                    output[0, list(keys)].reshape(len(keys), kv_heads, -1).cpu().clone()
                )
                captured[index] = CapturedValues(
                    values=values,
                    input_ids=ids,
                    layer=index,
                    keys=keys,
                    model_identity=id(model),
                    input_scale=scaling_record,
                    values_sha256=_values_digest(values),
                )

            handles.append(
                layers[index].self_attn.v_proj.register_forward_hook(capture_v)
            )
        for gate in gates:
            queries = list(gate.queries)
            if gate.kind == "mlp":

                def scale_mlp(module, args, output, gate=gate, queries=queries):
                    changed = output.clone()
                    changed[0, queries] *= gate.strength
                    return changed

                handles.append(layers[gate.layer].mlp.register_forward_hook(scale_mlp))
                continue
            attention = layers[gate.layer].self_attn
            cache = {}

            def current_values(module, args, output, gate=gate, cache=cache):
                cache["values"] = output[0, list(gate.keys)].reshape(
                    len(gate.keys), kv_heads, -1
                )

            def current_context(module, args, cache=cache):
                cache["context"] = args[0]

            def modify_attention(
                module, args, output, gate=gate, queries=queries, cache=cache
            ):
                weights = output[1][0, :, queries][:, :, list(gate.keys)]
                future = (
                    torch.tensor(gate.keys, device=weights.device)[None, :]
                    > torch.tensor(queries, device=weights.device)[:, None]
                )
                if torch.any(weights[:, future] != 0):
                    raise ValueError("native attention contains future-to-past edges")
                values = cache["values"]
                selected = [k in gate.selected for k in gate.keys]
                if gate.kind == "donor":
                    donor = gate.donor.values.to(values.device)
                    if donor.shape != values.shape or not torch.isfinite(donor).all():
                        raise ValueError("invalid branch-matched donor values")
                    # Negative of the selected content gives selected A*(donor-V).
                    delta, stats = message_gate_delta(
                        weights, values.float() - donor.float(), selected, 0, "content"
                    )
                else:
                    destination = (
                        [k in gate.destination for k in gate.keys]
                        if gate.destination
                        else None
                    )
                    delta, stats = message_gate_delta(
                        weights, values, selected, gate.strength, gate.kind, destination
                    )
                context = cache["context"].clone()
                context[0, queries] = (
                    context[0, queries].float() + delta.reshape(len(queries), -1)
                ).to(context.dtype)
                projected = module.o_proj(context)
                diagnostics.append({"layer": gate.layer, "kind": gate.kind, **stats})
                return projected, output[1]

            handles.extend(
                [
                    attention.v_proj.register_forward_hook(current_values),
                    attention.o_proj.register_forward_pre_hook(current_context),
                    attention.register_forward_hook(modify_attention),
                ]
            )
        output = model(**kwargs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return output.logits[0], captured, diagnostics
