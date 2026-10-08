"""Scientific contracts: causal rows, native source messages, fit isolation, order."""
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from state_audit.model.adapter import ModelAdapter

from .observe import collect_control
from .readout import OrderedCompatibility, WideCurrentCompatibility, fit_statistics, row_permutations, transform_view


def tiny_adapter():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'sdpa'
    return ModelAdapter(LlamaForCausalLM(config).eval())


def test_native_source_reconstruction_and_causal_overlap():
    adapter = tiny_adapter()
    control = dict(token_ids=list(range(3, 23)), source_mask=[True] * 7 + [False] * 13,
                   candidate_ids=[30, 31])
    states, heads, mass, native, errors = collect_control(adapter, control)
    assert states.shape == (8, 2, 4, 32)
    assert errors['state_equation'] == 0
    assert errors['head_reconstruction'] < 1e-6
    np.testing.assert_allclose(states[:, 0, :3].sum(axis=1), states[:, 1, 0], atol=1e-7)
    extended = dict(control, token_ids=control['token_ids'] + [40, 41],
                    source_mask=control['source_mask'] + [False, False])
    next_states, next_heads, _, _, _ = collect_control(adapter, extended)
    np.testing.assert_allclose(states[2:], next_states[:-2], atol=1e-7)
    np.testing.assert_allclose(heads[2:], next_heads[:-2], atol=1e-7)
    reordered = dict(control, candidate_ids=[31, 30])
    other_states, _, _, other_native, _ = collect_control(adapter, reordered)
    np.testing.assert_array_equal(states, other_states)
    np.testing.assert_allclose(native['pair_logits'], other_native['pair_logits'][::-1], atol=0)
    assert np.all((mass >= 0) & (mass <= 1))


def test_fit_statistics_cannot_read_dev():
    generator = np.random.default_rng(42)
    states = generator.normal(size=(5, 4, 2, 4, 8)).astype(np.float32)
    first_mean, first_scale = fit_statistics(states, [0, 1, 2])
    states[3:] += 1e6
    mean, scale = fit_statistics(states, [0, 1, 2])
    torch.testing.assert_close(mean, first_mean, rtol=0, atol=0)
    torch.testing.assert_close(scale, first_scale, rtol=0, atol=0)


def test_ordered_interactions_and_matched_views():
    torch.manual_seed(42)
    states = torch.randn(2, 4, 2, 4, 8)
    candidates = torch.randn(2, 2, 8)
    model = OrderedCompatibility(window=4, layers=2, sites=4, width=8)
    with torch.no_grad():
        model.level.normal_()
        model.transition.normal_()
    permutation = torch.tensor([[2, 0, 1, 3], [1, 2, 0, 3]])
    shuffled = transform_view(states, 'shuffled', permutation)
    torch.testing.assert_close(states[:, -1], shuffled[:, -1])
    torch.testing.assert_close(states.mean(1), shuffled.mean(1))
    assert not torch.allclose(model(states, candidates), model(shuffled, candidates))
    torch.testing.assert_close(transform_view(states, 'single'), transform_view(shuffled, 'single'))
    torch.testing.assert_close(transform_view(states, 'mean'), transform_view(shuffled, 'mean'))
    assert np.all(row_permutations(12, 4).numpy()[:, -1] == 3)


def test_wide_current_baseline_cannot_read_previous_rows():
    torch.manual_seed(73)
    states = torch.randn(2, 4, 2, 4, 8)
    candidates = torch.randn(2, 2, 8)
    model = WideCurrentCompatibility(layers=2, sites=4, width=8)
    with torch.no_grad():
        model.level.normal_()
        model.square.normal_()
    changed = states.clone()
    changed[:, :-1] += 100
    torch.testing.assert_close(model(states, candidates), model(changed, candidates), rtol=0, atol=0)
    changed[:, -1] += 1
    assert not torch.allclose(model(states, candidates), model(changed, candidates))
