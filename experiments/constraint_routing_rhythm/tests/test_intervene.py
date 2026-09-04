from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.constraint_routing_rhythm import intervene
from experiments.constraint_routing_rhythm.intervene import (
    RelayGate,
    baseline_forward,
    gated_eager_attention,
    rerun_gate,
    validate_attention_backend,
)


def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config).eval()


def empty_gate(tokens: int, split_layer: int = 1) -> RelayGate:
    return RelayGate(
        upstream_edges=torch.zeros(tokens, tokens, dtype=torch.bool),
        downstream_edges=torch.zeros(tokens, tokens, dtype=torch.bool),
        split_layer=split_layer,
        cut_evidence=False,
        cut_upstream=False,
        cut_downstream=False,
        evidence_mask=torch.zeros(tokens, dtype=torch.bool),
    )


class TinyAttention:
    num_key_value_groups = 1
    training = False

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.o_proj = torch.nn.Linear(2, 2, bias=False)


def attention_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
    mask = torch.tensor([[[[0.0, -torch.inf], [0.0, 0.0]]]])
    return query, key, value, mask


def test_identity_gate_matches_baseline() -> None:
    model = tiny_model()
    cache = baseline_forward(model, [1, 2, 3, 4, 5, 6, 7], response_start=4)

    delta = rerun_gate(model, cache, empty_gate(tokens=6))

    torch.testing.assert_close(delta, torch.zeros_like(cache.full_margin))


def test_baseline_keeps_only_requested_rerun_checkpoints() -> None:
    model = tiny_model()
    cache = baseline_forward(
        model,
        [1, 2, 3, 4, 5, 6, 7],
        response_start=4,
        checkpoint_layers=(0, 1),
    )

    assert cache.layer_count == 3
    assert set(cache.layer_input) == {0, 1}
    assert all(value.device.type == "cpu" for value in cache.layer_input.values())


def test_native_eager_matches_an_executed_noop_custom_gate(monkeypatch) -> None:
    model = tiny_model()
    observed_gates = []
    native_backend = intervene.gated_eager_attention

    def recording_backend(*args, relay_gate=None, **kwargs):
        observed_gates.append(relay_gate)
        return native_backend(*args, relay_gate=relay_gate, **kwargs)

    monkeypatch.setattr(intervene, "gated_eager_attention", recording_backend)

    error = validate_attention_backend(model, torch.arange(1, 13))

    assert error <= 1e-6
    assert len(observed_gates) == model.config.num_hidden_layers
    assert all(gate is not None for gate in observed_gates)
    assert all(
        gate.cut_evidence and gate.cut_upstream and gate.cut_downstream
        for gate in observed_gates
    )
    assert getattr(model, intervene.VALIDATED_ATTRIBUTE)
    assert model.config._attn_implementation == intervene.ATTENTION_BACKEND


def test_baseline_validates_attention_backend_once_per_model(monkeypatch) -> None:
    model = tiny_model()
    calls = 0
    native_validator = intervene.validate_attention_backend

    def recording_validator(model, token_ids):
        nonlocal calls
        calls += 1
        return native_validator(model, token_ids)

    monkeypatch.setattr(intervene, "validate_attention_backend", recording_validator)

    baseline_forward(model, [1, 2, 3, 4, 5, 6, 7], response_start=4)
    baseline_forward(model, [1, 2, 3, 4, 5, 6, 7], response_start=4)

    assert calls == 1


def test_source_cut_deletes_mass_without_renormalizing() -> None:
    query, key, value, mask = attention_inputs()
    gate = empty_gate(tokens=2)
    gate = RelayGate(
        **{
            **gate.__dict__,
            "cut_evidence": True,
            "evidence_mask": torch.tensor([True, False]),
        }
    )

    output, _ = gated_eager_attention(
        TinyAttention(layer_idx=0),
        query,
        key,
        value,
        mask,
        scaling=1.0,
        relay_gate=gate,
    )

    native_probability = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(
        output[0, 1, 0], torch.tensor([0.0, native_probability[1] * 4.0])
    )


def test_direct_response_cut_preserves_prompt_queries() -> None:
    query, key, value, mask = attention_inputs()
    gate = RelayGate(
        upstream_edges=torch.zeros(2, 2, dtype=torch.bool),
        downstream_edges=torch.zeros(2, 2, dtype=torch.bool),
        split_layer=1,
        cut_evidence=True,
        cut_upstream=False,
        cut_downstream=False,
        evidence_mask=torch.tensor([True, False]),
        evidence_targets=torch.tensor([False, True]),
    )

    output, _ = gated_eager_attention(
        TinyAttention(layer_idx=0),
        query,
        key,
        value,
        mask,
        scaling=1.0,
        relay_gate=gate,
    )

    probability = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(
        output[0, 1, 0], torch.tensor([0.0, probability[1] * 4.0])
    )


def test_upstream_and_downstream_cuts_respect_split_layer() -> None:
    query, key, value, mask = attention_inputs()
    upstream = torch.zeros(2, 2, dtype=torch.bool)
    upstream[1, 0] = True
    downstream = torch.zeros(2, 2, dtype=torch.bool)
    downstream[1, 1] = True
    gate = RelayGate(
        upstream_edges=upstream,
        downstream_edges=downstream,
        split_layer=1,
        cut_evidence=False,
        cut_upstream=True,
        cut_downstream=True,
        evidence_mask=torch.zeros(2, dtype=torch.bool),
    )

    before, _ = gated_eager_attention(
        TinyAttention(layer_idx=0),
        query,
        key,
        value,
        mask,
        scaling=1.0,
        relay_gate=gate,
    )
    after, _ = gated_eager_attention(
        TinyAttention(layer_idx=1),
        query,
        key,
        value,
        mask,
        scaling=1.0,
        relay_gate=gate,
    )

    probability = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(
        before[0, 1, 0], torch.tensor([0.0, probability[1] * 4.0])
    )
    torch.testing.assert_close(
        after[0, 1, 0], torch.tensor([probability[0] * 2.0, 0.0])
    )


def test_early_message_cut_changes_later_attention() -> None:
    model = tiny_model()
    baseline_layer_one: list[Tensor] = []
    changed_layer_one: list[Tensor] = []

    def baseline_observer(
        layer: int, probability: Tensor, _value: Tensor, _weight: Tensor
    ) -> None:
        if layer == 1:
            baseline_layer_one.append(probability[:, :, 4, :].detach().clone())

    cache = baseline_forward(
        model,
        [1, 2, 3, 4, 5, 6, 7],
        response_start=4,
        observer=baseline_observer,
    )
    upstream = torch.zeros(6, 6, dtype=torch.bool)
    upstream[4, 0] = True
    gate = RelayGate(
        upstream_edges=upstream,
        downstream_edges=torch.zeros_like(upstream),
        split_layer=1,
        cut_evidence=False,
        cut_upstream=True,
        cut_downstream=False,
        evidence_mask=torch.zeros(6, dtype=torch.bool),
    )

    def changed_observer(
        layer: int, probability: Tensor, _value: Tensor, _weight: Tensor
    ) -> None:
        if layer == 1:
            changed_layer_one.append(probability[:, :, 4, :].detach().clone())

    rerun_gate(model, cache, gate, observer=changed_observer)

    assert not torch.allclose(baseline_layer_one[0], changed_layer_one[0])


def test_response_tokens_use_previous_query_position() -> None:
    model = tiny_model()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])

    cache = baseline_forward(model, tokens, response_start=4)

    torch.testing.assert_close(cache.query, torch.tensor([3, 4, 5]))
    torch.testing.assert_close(cache.target, tokens[cache.query + 1])
    assert cache.full_margin.shape == cache.query.shape


def test_runner_uses_native_logits_and_margin_uses_fp32_weight_difference() -> None:
    model = tiny_model().to(torch.bfloat16)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(model, tokens, response_start=4)

    with torch.inference_mode():
        hidden = intervene.forward_layers(
            model,
            model.model.embed_tokens(tokens[:-1][None]),
            0,
        )[0].index_select(0, cache.query)
        logits = F.linear(hidden, model.lm_head.weight)
        log_probability = logits.float().log_softmax(dim=1)
        target_logprob = log_probability.gather(1, cache.target[:, None])[:, 0]
        entropy = -(log_probability.exp() * log_probability).sum(dim=1)
        logits.scatter_(1, cache.target[:, None], -torch.inf)
        runner = logits.argmax(dim=1)
        direction = model.lm_head.weight.index_select(0, cache.target).float()
        direction -= model.lm_head.weight.index_select(0, runner).float()
        margin = torch.einsum("td,td->t", hidden.float(), direction)

    assert cache.readout_direction.dtype == torch.float32
    assert cache.baseline_target_logprob.dtype == torch.float32
    assert cache.baseline_entropy.dtype == torch.float32
    torch.testing.assert_close(cache.runner, runner)
    torch.testing.assert_close(cache.readout_direction, direction)
    torch.testing.assert_close(cache.full_margin, margin)
    torch.testing.assert_close(cache.baseline_target_logprob, target_logprob)
    torch.testing.assert_close(cache.baseline_entropy, entropy)


def test_cached_suffix_matches_full_gated_forward() -> None:
    model = tiny_model()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(model, tokens, response_start=4, checkpoint_layers=(0, 1))
    downstream = torch.zeros(6, 6, dtype=torch.bool)
    downstream[4, 0] = True
    downstream[5, 2] = True
    gate = RelayGate(
        upstream_edges=torch.zeros_like(downstream),
        downstream_edges=downstream,
        split_layer=1,
        cut_evidence=False,
        cut_upstream=False,
        cut_downstream=True,
        evidence_mask=torch.zeros(6, dtype=torch.bool),
    )
    suffix_delta = rerun_gate(model, cache, gate)

    device = model.device
    source_ids = tokens[:-1].to(device)[None]
    with torch.inference_mode():
        hidden = model.model.embed_tokens(source_ids)
        hidden = intervene.forward_layers(
            model,
            hidden,
            0,
            gate=intervene.gate_to(gate, device),
        )
        response_hidden = hidden.index_select(1, cache.query.to(device))
        full_margin = torch.einsum(
            "btd,td->bt",
            response_hidden.float(),
            cache.readout_direction.to(device).float(),
        )[0]
    full_delta = full_margin.cpu() - cache.full_margin

    torch.testing.assert_close(suffix_delta, full_delta, atol=1e-6, rtol=1e-5)


def test_uncached_gate_split_falls_back_to_layer_zero_exactly() -> None:
    model = tiny_model()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(model, tokens, response_start=4, checkpoint_layers=(0, 1))
    downstream = torch.zeros(6, 6, dtype=torch.bool)
    downstream[5, 2] = True
    gate = RelayGate(
        upstream_edges=torch.zeros_like(downstream),
        downstream_edges=downstream,
        split_layer=2,
        cut_evidence=False,
        cut_upstream=False,
        cut_downstream=True,
        evidence_mask=torch.zeros(6, dtype=torch.bool),
    )

    fallback_delta = rerun_gate(model, cache, gate)
    device = model.device
    with torch.inference_mode():
        hidden = intervene.forward_layers(
            model,
            model.model.embed_tokens(tokens[:-1].to(device)[None]),
            0,
            gate=intervene.gate_to(gate, device),
        )
        hidden = hidden.index_select(1, cache.query.to(device))
        margin = torch.einsum(
            "btd,td->bt",
            hidden.float(),
            cache.readout_direction.to(device),
        )[0]

    torch.testing.assert_close(
        fallback_delta,
        margin.cpu() - cache.full_margin,
        atol=1e-6,
        rtol=1e-5,
    )
