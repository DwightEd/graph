"""Physical message reconstruction and independent causal replay checks."""
from contextlib import contextmanager

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from state_audit.model.adapter import ModelAdapter
from .grounded_projection import (capture_projection_inputs, content_candidates,
    history_transport, observed_attention, projected_carriers, replay_queries)
from .grounded_projection_cycle import cycle_transport
from .grounded_projection_match import corrected_scores
from .grounded_projection_run import freeze


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    adapter = ModelAdapter(model)
    tokens = torch.tensor([[1, 2, 7, 9, 3, 7, 5, 8]])
    with capture_projection_inputs(model, 4) as records:
        hidden = model.model(tokens).last_hidden_state[0]
    positions = torch.arange(tokens.shape[1])
    cosine, sine = model.model.rotary_emb(model.model.embed_tokens(tokens), positions[None])
    return adapter, tokens, records, hidden, cosine[0], sine[0]


@contextmanager
def finite_patch(model, index, position, delta):
    def change(module, inputs):
        value = inputs[0].clone()
        value[0, position] += delta.flatten()
        return (value,)
    handle = model.model.layers[index].self_attn.o_proj.register_forward_pre_hook(change)
    try:
        yield
    finally:
        handle.remove()


def test_attention_reconstruction_and_identity_all_cuts():
    adapter, tokens, records, hidden, cosine, sine = fixture()
    positions = torch.arange(3, 8)
    for layer in range(3):
        attention, values = observed_attention(adapter, layer, records[layer], positions, cosine, sine)
        message = torch.einsum('thk,khd->thd', attention, values).flatten(1)
        assert torch.allclose(message, records[layer]['head'], atol=1e-7)
        replay = replay_queries(adapter, records, layer, torch.arange(5), positions,
                                torch.zeros(5, 4, 8), cosine, sine)
        assert torch.allclose(replay, hidden[positions], atol=2e-6)


def test_batched_worlds_equal_individual_native_finite_patches():
    adapter, tokens, records, hidden, cosine, sine = fixture()
    positions = torch.tensor([3, 5, 6])
    delta = torch.randn(3, 4, 8) * .05
    for layer in (0, 1, 2):
        replay = replay_queries(adapter, records, layer, positions - 3, positions, delta, cosine, sine)
        for row, position in enumerate(positions):
            with finite_patch(adapter.native, layer, position, delta[row]):
                changed = adapter.native.model(tokens).last_hidden_state[0, position]
            assert torch.allclose(replay[row], changed, atol=3e-6)
        assert (replay - hidden[positions]).abs().max() > 1e-3


def test_projection_and_history_never_read_future_carriers():
    adapter, tokens, records, hidden, cosine, sine = fixture()
    positions = torch.arange(3, 8)
    attention, values = observed_attention(adapter, 1, records[1], positions, cosine, sine)
    source, kernel, similarity = content_candidates(adapter.native, tokens[0].tolist(), 4,
                                                    [False, True, True, False])
    graph, flat, projected = projected_carriers(attention, values, source, kernel, 4)
    assert torch.equal(kernel[1] > 0, torch.tensor([False, True]))
    assert torch.allclose(projected.norm(dim=-1), values[4:].norm(dim=-1), atol=1e-7)
    original = history_transport(attention, graph, 4)
    changed = graph.clone()
    changed[2:] += 100
    for rewired in (False, True):
        before = history_transport(attention, graph, 4, rewired)
        after = history_transport(attention, changed, 4, rewired)
        assert torch.equal(before[:3], after[:3])
        assert torch.equal(before[0], torch.zeros_like(before[0]))
    assert torch.isfinite(original).all() and torch.isfinite(flat).all()


def test_cycle_batch_invariance_and_causal_carriers():
    adapter, tokens, records, hidden, cosine, sine = fixture()
    attention, values = observed_attention(adapter, 1, records[1], torch.arange(3, 8), cosine, sine)
    source, kernel, similarity = content_candidates(adapter.native, tokens[0].tolist(), 4,
                                                    [False, True, True, False])
    two = cycle_transport(attention, values, source, kernel, 4, batch=2)
    one = cycle_transport(attention, values, source, kernel, 4, batch=1)
    changed = values.clone()
    changed[6:] += 100
    future = cycle_transport(attention, changed, source, kernel, 4, batch=2)
    for kind in two:
        assert torch.allclose(two[kind], one[kind], atol=1e-7)
        assert torch.equal(two[kind][:3], future[kind][:3])
        assert torch.equal(two[kind][0], torch.zeros_like(two[kind][0]))


def test_zero_matched_risk_cancels_layout_bias_preserves_edge_comparison():
    baseline = torch.tensor([-1., -2.])
    identity = baseline[:, None] + torch.tensor([[1e-4] * 4, [-1e-4] * 4])
    zero_effect = (baseline[:, None] - identity).numpy()
    previous = {name + '_layers': zero_effect.copy() for name in ('graph', 'flat', 'rewired')}
    scores = corrected_scores(previous, identity, baseline)
    assert all((scores[name] == 0).all() for name in ('graph', 'flat', 'rewired'))
    previous['graph_layers'] += .1
    changed = corrected_scores(previous, identity, baseline)
    assert torch.allclose(torch.tensor(changed['edge_increment']), torch.full((2,), .1))


def test_resume_rejects_changed_frozen_inputs_without_overwriting_protocol(tmp_path):
    inputs = tmp_path / 'inputs.json'
    inputs.write_text('{}')
    freeze(tmp_path, [])
    original = (tmp_path / 'protocol.json').read_bytes()
    freeze(tmp_path, [])
    inputs.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Frozen code/input changed'):
        freeze(tmp_path, [])
    assert (tmp_path / 'protocol.json').read_bytes() == original
