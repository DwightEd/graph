"""Native accounting gates: causal rows, cancellation, normalization and precision."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.common.llama_message_intervention import forward_layers
from experiments.reanchor_flow.native_trace import (
    FrozenReadout,
    NativeTraceObserver,
    capture_native_trace,
    residual_projection,
)


@pytest.fixture
def model():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    result = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        result.model.norm.weight.copy_(torch.linspace(0.5, 1.8, 16))
    return result


def test_shared_projection_is_orthonormal_reproducible_and_linear():
    basis = residual_projection(12, 5, 2026)
    np.testing.assert_array_equal(basis, residual_projection(12, 5, 2026))
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-7)
    assert not np.allclose(basis, residual_projection(12, 5, 2027))
    first, second = np.random.default_rng(3).normal(size=(2, 4, 12))
    np.testing.assert_allclose((first + second) @ basis, first @ basis + second @ basis)


def test_trace_aligns_q_to_observed_q_plus_one_and_uses_raw_final_rms(model):
    ids = torch.tensor([2, 4, 7, 3, 11, 9, 12])
    trace = capture_native_trace(model, ids, 3, sketch_dim=16, query_chunk=2)
    rows = torch.arange(2, 6)
    np.testing.assert_array_equal(trace["row_position"], rows.numpy())
    np.testing.assert_array_equal(trace["observed_token_ids"], ids[3:].numpy())
    assert np.all(trace["runner_token_ids"] != trace["observed_token_ids"])
    with torch.inference_mode():
        raw = forward_layers(
            model, model.model.embed_tokens(ids[:-1][None]), 0,
            attention_query_chunk=2, apply_final_norm=False,
        )[0, rows]
        expected_rms = (raw.float().square().mean(-1) + model.model.norm.variance_epsilon).sqrt()
        logits = model(ids[:-1][None], use_cache=False).logits[0, rows].float()
        observed = logits.gather(1, ids[3:, None]).flatten()
        logits.scatter_(1, ids[3:, None], -torch.inf)
        expected_margin = observed - logits.max(-1).values
    np.testing.assert_allclose(trace["final_rms_denominator"], expected_rms.numpy(), rtol=1e-6)
    np.testing.assert_allclose(trace["final_margin"], expected_margin.numpy(), atol=1e-6)
    np.testing.assert_allclose(trace["forward_repeat_max_abs_error"], 0, atol=1e-8)
    assert trace["head_sketch"].shape == (2, 4, 4, 16)
    np.testing.assert_allclose(
        np.linalg.norm(trace["head_sketch"], axis=-1), trace["head_norm"], atol=1e-7,
    )
    valid = trace["edge_source_position"] >= 0
    assert np.all(trace["edge_source_position"][valid] <= np.broadcast_to(
        rows.numpy()[None, None, :, None], trace["edge_source_position"].shape
    )[valid])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_signed_and_vector_accounting_close_with_explicit_rounding(model, dtype):
    trace = capture_native_trace(
        model.to(dtype), torch.tensor([2, 4, 7, 3, 11, 9]), 2,
        sketch_dim=8, query_chunk=2,
    )
    for suffix in ("sketch", "margin"):
        residual = trace["residual_sketch" if suffix == "sketch" else "stage_margin"]
        attention = trace[f"head_{suffix}"].sum(1) + trace[f"head_remainder_{suffix}"]
        post = residual[:-1] + attention + trace[f"attention_add_remainder_{suffix}"]
        after = post + trace[f"mlp_{suffix}"] + trace[f"mlp_add_remainder_{suffix}"]
        np.testing.assert_allclose(attention, trace[f"attention_{suffix}"], atol=1e-6)
        np.testing.assert_allclose(post, trace[f"post_attention_{suffix}"], atol=1e-6)
        np.testing.assert_allclose(after, residual[1:], atol=1e-6)
    np.testing.assert_allclose(
        trace["stage_margin"][-1] + trace["readout_bias"] + trace["readout_remainder"],
        trace["final_margin"], atol=1e-7,
    )
    np.testing.assert_allclose(
        trace["edge_sum_margin"] + trace["edge_rounding_remainder"], trace["head_margin"],
        atol=1e-7,
    )
    if dtype == torch.bfloat16:
        assert np.abs(trace["attention_add_remainder_margin"]).max() > 0


def test_raw_gqa_writes_close_without_remainder_corrections(model):
    """A wrong W_O/head mapping cannot be hidden in defined remainder arrays."""

    for layer_index, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.bias = torch.nn.Parameter(
            torch.linspace(-0.003, 0.005, 16) * (layer_index + 1)
        )
    model.lm_head.bias = torch.nn.Parameter(torch.linspace(-0.07, 0.08, 41))
    trace = capture_native_trace(
        model, torch.tensor([2, 4, 7, 3, 11, 9]), 2, sketch_dim=16, query_chunk=2,
    )
    bias = np.stack([
        layer.self_attn.o_proj.bias.detach().numpy() for layer in model.model.layers
    ])
    direction = (
        model.lm_head.weight[trace["observed_token_ids"]]
        - model.lm_head.weight[trace["runner_token_ids"]]
    ).detach().numpy() * model.model.norm.weight.detach().numpy()[None]
    direction /= trace["final_rms_denominator"][:, None]
    bias_sketch = (bias @ trace["sketch_projection"])[:, None]
    bias_margin = np.einsum("ld,qd->lq", bias, direction)
    np.testing.assert_allclose(
        trace["head_sketch"].sum(1) + bias_sketch, trace["attention_sketch"], atol=1e-7,
    )
    np.testing.assert_allclose(
        trace["head_margin"].sum(1) + bias_margin, trace["attention_margin"], atol=1e-6,
    )
    np.testing.assert_allclose(trace["edge_sum_margin"], trace["head_margin"], atol=1e-6)
    for suffix in ("sketch", "margin"):
        residual = trace["residual_sketch" if suffix == "sketch" else "stage_margin"]
        np.testing.assert_allclose(
            residual[:-1] + trace[f"attention_{suffix}"],
            trace[f"post_attention_{suffix}"], atol=1e-6,
        )
        np.testing.assert_allclose(
            trace[f"post_attention_{suffix}"] + trace[f"mlp_{suffix}"],
            residual[1:], atol=1e-6,
        )
    np.testing.assert_allclose(
        trace["stage_margin"][0]
        + (trace["attention_margin"] + trace["mlp_margin"]).sum(0)
        + trace["readout_bias"],
        trace["final_margin"], atol=1e-6,
    )


def test_query_chunks_display_budget_and_future_tokens_do_not_change_net_writes(model):
    ids = torch.tensor([2, 4, 7, 3, 11, 9, 12])
    one = capture_native_trace(model, ids, 3, sketch_dim=7, query_chunk=1, display_edges=0)
    three = capture_native_trace(model, ids, 3, sketch_dim=7, query_chunk=3, display_edges=4)
    prefix = capture_native_trace(model, ids[:5], 3, sketch_dim=7, query_chunk=2)
    for key in ("head_sketch", "head_margin", "head_norm", "edge_sum_margin", "edge_total_absolute_margin"):
        np.testing.assert_allclose(one[key], three[key], atol=1e-6)
        np.testing.assert_allclose(one[key][:, :, :2], prefix[key], atol=1e-6)
    for key in ("residual_sketch", "mlp_sketch", "stage_margin", "mlp_margin"):
        np.testing.assert_allclose(one[key], three[key], atol=1e-6)
        np.testing.assert_allclose(one[key][:, :2], prefix[key], atol=1e-6)
    np.testing.assert_array_equal(one["runner_token_ids"][:2], prefix["runner_token_ids"])
    assert one["edge_source_position"].shape[-1] == 0


def test_opposing_messages_remain_signed_and_omission_does_not_hide_cancellation():
    attention = SimpleNamespace(
        head_dim=1, q_proj=SimpleNamespace(out_features=2),
        o_proj=SimpleNamespace(weight=torch.tensor([[1.0, -1.0], [0.0, 0.0]])),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    observer = NativeTraceObserver(
        model, torch.tensor([1]), torch.tensor([[1.0, 0.0]]), torch.eye(2), 0, True,
    )
    observer.observe_layer_input(0, torch.zeros(1, 2, 2))
    probability = torch.tensor([[[[1., 0.], [.5, .5]], [[1., 0.], [.5, .5]]]])
    value = torch.tensor([[[[2.], [-2.]], [[2.], [2.]]]])
    code = probability @ value
    observer.observe_chunk(0, 0, probability, value, attention.o_proj.weight)
    observer.observe_head_output(0, 0, code)
    write = code.transpose(1, 2).reshape(1, 2, 2) @ attention.o_proj.weight.T
    observer.observe_attention_write(0, write)
    observer.observe_mlp_write(0, torch.zeros_like(write))
    arrays = observer.finish(FrozenReadout(
        torch.tensor([[1., 0.]]), {"final_margin": np.array([-2.]), "readout_bias": np.zeros(1)},
    ))
    np.testing.assert_allclose(arrays["head_margin"][0, :, 0], [0, -2])
    np.testing.assert_allclose(arrays["head_norm"][0, :, 0], [0, 2])
    # Head 0 carries +1 and -1 edges: zero signed remainder does not mean no omitted activity.
    assert arrays["omitted_margin"][0, 0, 0] == 0
    assert arrays["edge_omitted_absolute_margin"][0, 0, 0] == 2
    np.testing.assert_allclose(arrays["head_code"][0, :, 0, 0], [0, 2])
