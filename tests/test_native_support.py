"""Algorithm invariants and actual tiny-model execution, not natural-data results."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from state_audit.attribution import _projection_gram
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend, prefill_cache
from state_audit.native_forward import iter_forward_traces
from state_audit.native_ledger import finish_ledger, readout_direction
from state_audit.native_trace import layer_messages, native_hooks, source_masks
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=40, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return ModelAdapter(LlamaForCausalLM(config))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_forward_matches_actual_logits_without_backward(model, dtype):
    model.native.to(dtype)
    tokens = list(range(1, 16))
    with patch("torch.autograd.grad", side_effect=AssertionError("backward forbidden")):
        arrays = next(iter_forward_traces(model, tokens, 7, [2], [1], prefill_chunk_size=3))
    with torch.no_grad():
        hidden = model.forward(tokens[:9])[-1]
        logits = model.native.lm_head(hidden).float().numpy()
    candidates = [tokens[9], int(arrays["competitor_id"])]
    np.testing.assert_allclose(arrays["candidate_logits"], logits[candidates], atol=2e-3)
    assert abs(float(arrays["ledger_error"])) < 3e-6
    assert arrays["attention"].shape == (3, 4, 9)
    assert arrays["edge_value_energy"].shape == (3, 4, 9)
    assert np.isfinite(arrays["edge_value_energy"]).all()
    assert "edge_margin_sensitivity" not in arrays
    assert arrays["observed_id"].item() != arrays["competitor_id"].item()
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_queries_reuse_cache_and_saved_rows_exclude_future_keys(model):
    tokens = list(range(1, 18))
    calls = []
    original = model.native.model.forward

    def observe(*args, **kwargs):
        calls.extend(kwargs["input_ids"][0].tolist())
        return original(*args, **kwargs)

    with patch.object(model.native.model, "forward", side_effect=observe):
        before = list(iter_forward_traces(model, tokens, 7, [0, 1, 2], [1], prefill_chunk_size=3))
    assert calls == tokens[:9]
    altered = tokens[:10] + [30] * 7
    after = list(iter_forward_traces(model, altered, 7, [0, 1, 2], [1]))
    for left, right in zip(before, after):
        np.testing.assert_allclose(left["edge_logit_write"], right["edge_logit_write"], atol=1e-6)


def test_block_uses_fewer_forwards_and_cannot_read_future_within_block(model):
    tokens = list(range(1, 16))
    calls = []
    original = model.native.model.forward

    def observe(*args, **kwargs):
        calls.append((model.native.config._attn_implementation, kwargs["input_ids"].shape[-1]))
        return original(*args, **kwargs)

    with patch.object(model.native.model, "forward", side_effect=observe):
        before = list(iter_forward_traces(model, tokens, 7, range(8), [1],
                                         prefill_chunk_size=3, query_chunk_size=3))
    assert calls == [("sdpa", 3), ("sdpa", 3), ("eager", 3), ("eager", 3), ("eager", 2)]
    altered = tokens[:8] + [30] * 7
    after = list(iter_forward_traces(model, altered, 7, range(8), [1], query_chunk_size=3))
    for name in ("candidate_logits", "edge_logit_write", "edge_value_energy", "mlp_score"):
        np.testing.assert_allclose(before[0][name], after[0][name], atol=1e-6)
    sparse = list(iter_forward_traces(model, tokens, 7, [0, 1, 4, 5, 6], [1], query_chunk_size=3))
    for arrays in sparse:
        np.testing.assert_allclose(arrays["edge_value_energy"], before[int(arrays["target"])]["edge_value_energy"], atol=1e-6)


def test_hook_failure_restores_model_without_parameter_edits(model):
    with (
        patch.object(model.native.lm_head, "forward", side_effect=RuntimeError("readout")),
        pytest.raises(RuntimeError, match="readout"),
    ):
        next(iter_forward_traces(model, list(range(1, 16)), 7, [0], [1]))
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.requires_grad for parameter in model.native.parameters())
