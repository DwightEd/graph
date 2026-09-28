"""Numerical invariants needed for interpreting native measurements."""
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.internal_flow.capture import capture, reroute_write
from experiments.path_conflict.operators import grouped_values


def test_rerouting_preserves_mass_and_matches_explicit_attention():
    torch.manual_seed(42)
    attention = torch.softmax(torch.randn(1, 2, 5, 5), -1)
    values = torch.randn(1, 2, 5, 3)
    weight = torch.randn(6, 6)
    source, target = [0, 1, 2], [1]
    delta, mass, moved = reroute_write(attention, values, weight, 4, 1, source, target, .25)
    changed = attention.clone()
    changed[0, 1, 4, source] *= .75
    changed[0, 1, 4, 1] += .25 * mass
    torch.testing.assert_close(changed.sum(-1), attention.sum(-1))
    original_write = (attention[0, 1, 4] @ values[0, 1]) @ weight[:, 3:].T
    new_write = (changed[0, 1, 4] @ values[0, 1]) @ weight[:, 3:].T
    torch.testing.assert_close(delta, new_write - original_write, atol=1e-6, rtol=1e-5)
    assert moved > 0


def test_native_hooks_do_not_change_forward_and_null_patch_is_identity():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    ids = list(range(1, 10))
    groups = dict(source=[1, 2, 3], evidence=[2], other_source=[1, 3])
    result, run = capture(model, ids, groups, [10, 11])
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1].float()
    assert abs(result['margin'] - float(logits[10] - logits[11])) < 1e-6
    assert result['reconstruction_max'] < 1e-5
    assert len(run.states) == 6
    patch = dict(layer=0, head=1, target='evidence', dose=0., kind='reroute')
    sham, _ = capture(model, ids, groups, [10, 11], patch=patch, collect=False)
    assert sham['margin'] == result['margin']
    assert all(np.isfinite(value).all() for value in run.states.values())


def test_complete_candidate_scoring_and_initial_query_null_patch():
    from experiments.internal_flow.capture import sequence_logp
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    prefix = list(range(1, 10))
    probe = dict(prefix_ids=prefix, candidate_ids=[10, 11],
                 variants=dict(correct=prefix+[10, 12], wrong=prefix+[11, 12]),
                 groups=dict(source=[1, 2, 3], evidence=[2], other_source=[1, 3]))
    actual = sequence_logp(model, probe, 'correct')
    with torch.no_grad():
        logits = model(torch.tensor([prefix+[10]])).logits[0, -2:].double()
        expected = logits.log_softmax(-1)[torch.arange(2), torch.tensor([10, 12])].sum()
    assert abs(actual-float(expected)) < 1e-6
    patch = dict(layer=0, head=1, target='evidence', dose=0., kind='reroute')
    assert sequence_logp(model, probe, 'correct', patch) == actual
