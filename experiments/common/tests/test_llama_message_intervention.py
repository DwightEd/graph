from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.common import llama_message_intervention as intervention
from experiments.common.llama_message_intervention import (
    MessageGate,
    baseline_forward,
    forward_layers,
    gated_attention,
    rerun_gate,
    validate_manual_forward,
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
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


class TinyAttention:
    num_key_value_groups = 1
    attention_dropout = 0.0
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


def test_manual_decoder_matches_native_eager() -> None:
    model = tiny_model()
    error = validate_manual_forward(model, torch.arange(1, 9))
    assert error <= 1e-5
    assert getattr(model, intervention.VALIDATED_ATTRIBUTE)


def test_source_cut_deletes_mass_without_renormalizing() -> None:
    query, key, value, mask = attention_inputs()
    gate = MessageGate(split_layer=1, source_mask=torch.tensor([True, False]))
    output = gated_attention(
        TinyAttention(0), query, key, value, mask, 1.0, gate=gate
    )
    probability = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(
        output[0, 1, 0], torch.tensor([0.0, probability[1] * 4.0])
    )


def test_source_target_mask_preserves_other_queries() -> None:
    query, key, value, mask = attention_inputs()
    gate = MessageGate(
        split_layer=1,
        source_mask=torch.tensor([True, False]),
        source_targets=torch.tensor([False, True]),
    )
    output = gated_attention(
        TinyAttention(0), query, key, value, mask, 1.0, gate=gate
    )
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([2.0, 0.0]))


def test_early_and_late_edge_masks_respect_split() -> None:
    query, key, value, mask = attention_inputs()
    early = torch.zeros(2, 2, dtype=torch.bool)
    early[1, 0] = True
    late = torch.zeros(2, 2, dtype=torch.bool)
    late[1, 1] = True
    gate = MessageGate(split_layer=1, early_edges=early, late_edges=late)

    before = gated_attention(TinyAttention(0), query, key, value, mask, 1.0, gate=gate)
    after = gated_attention(TinyAttention(1), query, key, value, mask, 1.0, gate=gate)
    probability = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    torch.testing.assert_close(
        before[0, 1, 0], torch.tensor([0.0, probability[1] * 4.0])
    )
    torch.testing.assert_close(
        after[0, 1, 0], torch.tensor([probability[0] * 2.0, 0.0])
    )


def test_early_cut_recomputes_later_attention() -> None:
    model = tiny_model()
    baseline_layer_one: list[Tensor] = []
    changed_layer_one: list[Tensor] = []

    def baseline_observer(
        layer: int,
        probability: Tensor,
        _value: Tensor,
        _weight: Tensor,
    ) -> None:
        if layer == 1:
            baseline_layer_one.append(probability[:, :, 4, :].detach().clone())

    cache = baseline_forward(
        model,
        [1, 2, 3, 4, 5, 6, 7],
        response_start=4,
        observer=baseline_observer,
    )
    early = torch.zeros(6, 6, dtype=torch.bool)
    early[4, 0] = True

    def changed_observer(
        layer: int,
        probability: Tensor,
        _value: Tensor,
        _weight: Tensor,
    ) -> None:
        if layer == 1:
            changed_layer_one.append(probability[:, :, 4, :].detach().clone())

    rerun_gate(
        model,
        cache,
        MessageGate(split_layer=1, early_edges=early),
        observer=changed_observer,
    )
    assert not torch.allclose(baseline_layer_one[0], changed_layer_one[0])


def test_prediction_positions_use_the_previous_query() -> None:
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(tiny_model(), tokens, response_start=4)
    torch.testing.assert_close(cache.query, torch.tensor([3, 4, 5]))
    torch.testing.assert_close(cache.target, tokens[cache.query + 1])


def test_fixed_readout_matches_native_logits() -> None:
    model = tiny_model()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(model, tokens, response_start=4)
    with torch.inference_mode():
        hidden = forward_layers(
            model,
            model.model.embed_tokens(tokens[:-1][None]),
            0,
        )[0].index_select(0, cache.query)
        logits = F.linear(hidden, model.lm_head.weight, model.lm_head.bias)
        log_probability = logits.float().log_softmax(dim=1)
        logits.scatter_(1, cache.target[:, None], -torch.inf)
        runner = logits.argmax(dim=1)
        direction = model.lm_head.weight.index_select(0, cache.target).float()
        direction -= model.lm_head.weight.index_select(0, runner).float()
        margin = torch.einsum("td,td->t", hidden.float(), direction)
    torch.testing.assert_close(cache.runner, runner)
    torch.testing.assert_close(cache.full_margin, margin)
    torch.testing.assert_close(
        cache.baseline_target_logprob,
        log_probability.gather(1, cache.target[:, None])[:, 0],
    )


def test_cached_suffix_matches_full_manual_rerun() -> None:
    model = tiny_model()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    cache = baseline_forward(
        model,
        tokens,
        response_start=4,
        checkpoint_layers=(0, 1),
    )
    late = torch.zeros(6, 6, dtype=torch.bool)
    late[4, 0] = True
    late[5, 2] = True
    gate = MessageGate(split_layer=1, late_edges=late)
    suffix_delta = rerun_gate(model, cache, gate)

    with torch.inference_mode():
        hidden = forward_layers(
            model,
            model.model.embed_tokens(tokens[:-1][None]),
            0,
            gate=gate,
        ).index_select(1, cache.query)
        full_margin = torch.einsum(
            "btd,td->bt", hidden.float(), cache.readout_direction
        )[0]
        full_margin += cache.readout_bias
    torch.testing.assert_close(
        suffix_delta,
        full_margin - cache.full_margin,
        atol=1e-6,
        rtol=1e-5,
    )
