"""Native operator invariants with an actual small causal GQA transformer.

This exercises real attention/MLP forwards, not a mock returning fake scores.
It is not a test of the user's 8B weights or installed Transformers version.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.path_conflict.native import (Intervention, NativeRun, grouped_values,
    source_write, equal_norm_change, forward, evaluate)
from experiments.path_conflict.main import experiments, run_side, save_run
from experiments.path_conflict.report import compare, interactions, paired_effects, report
from experiments.path_conflict.data import inventory, quote_tokens, validate_text_cases


class Attention(nn.Module):
    def __init__(self, hidden=16, heads=4, kv_heads=2):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden // heads * kv_heads, bias=False)
        self.v_proj = nn.Linear(hidden, hidden // heads * kv_heads, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, states):
        batch, count, width = states.shape
        queries = self.q_proj(states).view(batch, count, self.heads, width // self.heads).transpose(1, 2)
        keys = grouped_values(self.k_proj(states), self.heads, self.kv_heads)
        values = grouped_values(self.v_proj(states), self.heads, self.kv_heads)
        raw = queries @ keys.transpose(-1, -2) / (width // self.heads) ** .5
        mask = torch.ones(count, count, device=states.device, dtype=torch.bool).triu(1)
        weights = raw.masked_fill(mask, -float('inf')).softmax(-1)
        heads = (weights @ values).transpose(1, 2).reshape(batch, count, width)
        return self.o_proj(heads), weights


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(16)
        self.self_attn = Attention()
        self.post_attention_layernorm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 16))

    def forward(self, states):
        output, weights = self.self_attn(self.input_layernorm(states))
        states = states + output
        return (states + self.mlp(self.post_attention_layernorm(states)), weights)


class SmallLlama(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = nn.Embedding(40, 16)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer() for _ in range(3)])
        self.model.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 40, bias=False)
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, model_type='llama')

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_attentions=True, return_dict=True):
        state = self.embedding(input_ids)
        attentions = []
        for layer in self.model.layers:
            state, attention = layer(state)
            attentions.append(attention)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(state)), attentions=attentions)


def fixture():
    model = SmallLlama().eval()
    probe = dict(prefix_ids=[2, 3, 4, 5, 6], prompt_length=3, candidates=[[7, 8], [9, 10, 11]],
        groups=dict(evidence=np.array([0]), value_source=np.array([1]),
                    other_prompt=np.array([2]), history=np.array([3, 4])),
        trace='00000.npz', seed=0, response_step=2, sampled_next=7)
    with torch.no_grad():
        last = model(torch.tensor([probe['prefix_ids']])).logits[0, -1]
    top = last.topk(5)
    probe.update(top_ids=top.indices.numpy(), top_logits=top.values.numpy(), log_normalizer=float(last.logsumexp(-1)))
    return model, probe


def test_original_forward_and_full_source_decomposition():
    model, probe = fixture()
    logits, run = forward(model, probe['prefix_ids'], probe)
    expected = model(torch.tensor([probe['prefix_ids']])).logits[0]
    torch.testing.assert_close(logits, expected)
    assert max(run.reconstruction) < 1e-6
    assert len(run.writes) == 3 * 4 * 4
    assert len(run.trajectory) == 6


def test_gqa_values_repeat_in_correct_head_order():
    values = torch.arange(12.).reshape(1, 3, 4)
    repeated = grouped_values(values, 4, 2)
    torch.testing.assert_close(repeated[:, 0], repeated[:, 1])
    torch.testing.assert_close(repeated[:, 2], repeated[:, 3])
    assert not torch.equal(repeated[:, 0], repeated[:, 2])


def test_source_partition_exact_sum_and_future_exclusion():
    model, probe = fixture()
    layer = model.model.layers[0]
    states = model.embedding(torch.tensor([probe['prefix_ids'] + [20, 21]]))
    normalized = layer.input_layernorm(states)
    output, attention = layer.self_attn(normalized)
    values = grouped_values(layer.self_attn.v_proj(normalized), 4, 2)
    parts = [source_write(attention, values, layer.self_attn.o_proj.weight, [4], list(indices), range(4))[1]
             for indices in probe['groups'].values()]
    torch.testing.assert_close(sum(parts)[0] + layer.self_attn.o_proj.bias, output[0, 4])
    future = source_write(attention, values, layer.self_attn.o_proj.weight, [4], [5, 6], range(4))[1]
    assert not future.any()


def test_cut_every_source_keeps_output_bias_and_residual_path():
    model, probe = fixture()
    changed = []
    with torch.inference_mode(), NativeRun(model, 5, probe['groups'], [7, 9], (Intervention(0, tuple(probe['groups'])),)):
        handle = model.model.layers[0].self_attn.register_forward_hook(lambda module, inputs, output: changed.append(output[0].detach()))
        model(torch.tensor([probe['prefix_ids']]))
        handle.remove()
    torch.testing.assert_close(changed[0][0, 4], model.model.layers[0].self_attn.o_proj.bias, atol=1e-7, rtol=1e-5)


def test_query_cut_does_not_change_earlier_outputs():
    model, probe = fixture()
    base, _ = forward(model, probe['prefix_ids'], probe)
    cut, run = forward(model, probe['prefix_ids'], probe, (Intervention(0, ('evidence',)),))
    torch.testing.assert_close(base[:-1], cut[:-1])
    assert not torch.equal(base[-1], cut[-1])
    assert run.changes[0]['query_change_norm'] > 0


def test_empty_history_is_zero_not_special_fallback():
    model, probe = fixture()
    probe['groups']['history'] = np.array([], dtype=int)
    base, _ = forward(model, probe['prefix_ids'], probe)
    changed, _ = forward(model, probe['prefix_ids'], probe, (Intervention(0, ('history',)),))
    torch.testing.assert_close(base, changed)


def test_single_head_cut_keeps_other_head_writes():
    model, probe = fixture()
    layer = model.model.layers[0]
    normalized = layer.input_layernorm(model.embedding(torch.tensor([probe['prefix_ids']])))
    _, attention = layer.self_attn(normalized)
    values = grouped_values(layer.self_attn.v_proj(normalized), 4, 2)
    components, _, _ = source_write(attention, values, layer.self_attn.o_proj.weight, [4], [0], [2])
    assert components[2].abs().sum() > 0
    assert not components[[0, 1, 3]].any()


def test_prefix_intervention_never_uses_candidate_suffix():
    model, probe = fixture()
    first, _ = forward(model, probe['prefix_ids'] + [7, 8], probe, (Intervention(0, ('evidence',), 'prefix'),))
    second, _ = forward(model, probe['prefix_ids'] + [9, 10, 11], probe, (Intervention(0, ('evidence',), 'prefix'),))
    torch.testing.assert_close(first[:5], second[:5])


def test_residual_random_control_matches_every_query_norm():
    write = torch.randn(10, 16)
    changed = equal_norm_change(write, 3)
    torch.testing.assert_close(changed.norm(dim=-1), write.norm(dim=-1))
    assert not torch.equal(changed, write)


def test_restore_original_downstream_heads_is_identity_without_cut():
    model, probe = fixture()
    base, run = forward(model, probe['prefix_ids'], probe)
    restore = dict(layer=2, heads=(1, 3), states=run.baseline_heads[2])
    restored, _ = forward(model, probe['prefix_ids'], probe, restore=restore)
    torch.testing.assert_close(restored, base)


def test_no_hooks_survive_exception():
    model, probe = fixture()
    with pytest.raises(RuntimeError):
        with NativeRun(model, 5, probe['groups'], [7, 9]):
            raise RuntimeError('deliberate test')
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_full_candidate_logp_matches_manual_teacher_forcing():
    model, probe = fixture()
    score, _, _, _ = evaluate(model, probe)
    for name, candidate in zip(['correct', 'wrong'], probe['candidates']):
        logits = model(torch.tensor([probe['prefix_ids'] + candidate])).logits[0]
        logs = logits[4:4 + len(candidate)].log_softmax(-1)
        expected = sum(float(logs[i, token].detach()) for i, token in enumerate(candidate))
        assert score[name + '_logp'] == pytest.approx(expected, abs=1e-6)
        assert score[name + '_tokens'] == len(candidate)


def test_inventory_does_not_inherit_old_annotations(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'settings.json').write_text(json.dumps(dict(model='model')))
    records = [dict(source_id='1', seed=0, trace='0.npz', tokens=3, response='good'),
               dict(source_id='1', seed=1, trace='1.npz', tokens=2, response='bad')]
    (source / 'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (source / 'prompts.jsonl').write_text(json.dumps(dict(source_id='1', prompt='evidence value')))
    case = dict(case_id='one', source_id='1', supported=dict(seed=0, target='good'),
                unsupported=dict(seed=1, target='bad'), evidence=['evidence'], value_source=['value'])
    output = tmp_path / 'out'
    output.mkdir()
    inventory(source, output, [case])
    table = pd.read_csv(output / 'samples.csv')
    assert set(table.whole_answer_label) == {'not_annotated'}
    assert not table.trace_available.any()
    assert len(pd.read_csv(output / 'same_question_pairs.csv')) == 1


def test_interaction_signs_and_normal_side_retained():
    records = []
    for side in ['supported', 'unsupported']:
        for name, margin in [('full', 1.), ('cut_evidence', 0.), ('cut_value_source', 2.), ('cut_history', 1.5),
                             ('cut_evidence_value_source', .5), ('cut_evidence_history', .25)]:
            records.append(dict(case_id='test', side=side, layer=-1 if name == 'full' else 0,
                scope='none' if name == 'full' else 'query', variant=name, next_margin=margin,
                sequence_margin=margin, mean_margin=margin, correct_logp=margin, wrong_logp=0.,
                correct_first_logp=margin, wrong_first_logp=0.))
    result = compare(pd.DataFrame(records))
    assert result[result.variant == 'cut_evidence'].delta_next_margin.eq(-1).all()
    interaction = interactions(result)
    assert interaction[interaction.rival == 'value_source'].evidence_support_next_margin.eq(1).all()
    assert paired_effects(result).delta_next_margin_difference.eq(0).all()


def test_actual_native_pipeline_resume_and_figures(tmp_path):
    model, probe = fixture()
    output = tmp_path / 'out'
    (output / 'runs').mkdir(parents=True)
    args = SimpleNamespace(replay_atol=1e-5, message_rtol=1e-5, restore_layer=2, restore_heads=[1])
    schedule = experiments([0], ['query'], [], 1, 7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for side in ('supported', 'unsupported'):
        run_side(args, model, dict(case_id='test', source_id='1'), side, probe, schedule, output)
    capture = next((output / 'runs').glob('*.npz'))
    stamp = capture.stat().st_mtime_ns
    run_side(args, model, dict(case_id='test', source_id='1'), 'supported', probe, schedule, output)
    assert capture.stat().st_mtime_ns == stamp
    report(output)
    assert (output / 'path_conflict_review.tar.gz').is_file()
    assert list((output / 'figures').glob('*.svg'))
    assert len(pd.read_csv(output / 'same_question_effects.csv')) > 0
    for name, values in model.state_dict().items():
        torch.testing.assert_close(values, before[name])


def test_mlp_ablation_is_target_local_and_preserves_attention():
    model, probe = fixture()
    base, _ = forward(model, probe['prefix_ids'], probe)
    changed, run = forward(model, probe['prefix_ids'], probe, (Intervention(1, ('mlp',)),))
    torch.testing.assert_close(base[:4], changed[:4])
    assert not torch.equal(base[4], changed[4])
    assert run.changes[0]['operation'] == 'cut_mlp'


def test_first_sample_seed_is_not_a_model_input():
    model, probe = fixture()
    same_prefix = dict(probe, seed=99, sampled_next=9)
    first, _, _, _ = evaluate(model, probe)
    second, _, _, _ = evaluate(model, same_prefix)
    assert first == second


def test_source_group_write_matches_independent_dense_projection():
    model, probe = fixture()
    attention = model.model.layers[0].self_attn
    raw = model.embedding(torch.tensor([probe['prefix_ids']]))
    output, weights = attention(raw)
    values = grouped_values(attention.v_proj(raw), 4, 2)
    actual = source_write(weights, values, attention.o_proj.weight, [2, 4], [0, 1], [1, 3])[1]
    joined = torch.zeros(2, 16)
    for row, query in enumerate([2, 4]):
        for head in [1, 3]:
            for source in [0, 1]:
                joined[row, head*4:(head+1)*4] += weights[0, head, query, source] * values[0, head, source]
    expected = joined @ attention.o_proj.weight.T
    torch.testing.assert_close(actual, expected)


class WordTokenizer:
    """Whitespace-preserving tokenizer used only to test saved-ID case assembly."""
    def __init__(self):
        self.pieces = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        matches = list(re.finditer(r' ?[A-Za-z0-9]+|[^A-Za-z0-9 ]| +', text))
        ids = []
        for match in matches:
            piece = match.group()
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(self.pieces.index(piece))
        return dict(input_ids=ids, offset_mapping=[match.span() for match in matches])

    def encode(self, text, add_special_tokens=False):
        return self(text)['input_ids']

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


def test_compile_keeps_original_prompt_and_stops_before_target(tmp_path):
    from experiments.path_conflict.data import compile_cases
    tokenizer = WordTokenizer()
    prompt = 'Evidence caps. Value headdress. Output:'
    prompt_ids = tokenizer.encode(prompt)
    samples = []
    for seed, response in [(0, 'They wear caps'), (1, 'They instead wear headdress')]:
        ids = prompt_ids + tokenizer.encode(response)
        filename = f'{seed}.npz'
        tokens = np.array([tokenizer.decode([token]) for token in ids])
        count = len(ids) - len(prompt_ids)
        np.savez(tmp_path / filename, token_ids=ids, token_text=tokens, prompt_length=len(prompt_ids),
                 top_ids=np.zeros((count, 5), int), top_logits=np.zeros((count, 5)), log_normalizer=np.zeros(count))
        samples.append(dict(source_id='one', seed=seed, trace=filename, response=response))
    case = dict(case_id='test', source_id='one', supported=dict(seed=0, target='caps'),
        unsupported=dict(seed=1, target='headdress'), candidates=[' caps', ' headdress'],
        evidence=['Evidence caps.'], value_source=['Value headdress.'])
    result = compile_cases(tmp_path, tokenizer, samples, {'one': prompt}, [case], tmp_path)[0]
    assert not result['identical_prefix']
    for name, side in result['sides'].items():
        decoded = tokenizer.decode(side['prefix_ids'])
        assert decoded.endswith('wear')
        assert side['prefix_ids'][:len(prompt_ids)] == prompt_ids
        assert len(side['groups']['history']) == side['response_step']


def test_duplicate_or_missing_claim_text_rejected_before_model_load():
    records = [dict(source_id='a', seed=0, response='same same'), dict(source_id='a', seed=1, response='bad')]
    case = dict(case_id='a', source_id='a', supported=dict(seed=0, target='same'),
        unsupported=dict(seed=1, target='bad'), evidence=['source'], value_source=['value'])
    with pytest.raises(AssertionError, match='target not unique'):
        validate_text_cases(records, {'a': 'source value'}, [case])


def test_replay_mismatch_stops_native_interventions():
    from experiments.path_conflict.main import verify_replay
    model, probe = fixture()
    probe['top_logits'] = probe['top_logits'] + 1
    with pytest.raises(AssertionError, match='do not replay'):
        verify_replay(model, probe, 1e-5, 1e-5)


def test_huggingface_llama_eager_operator_when_installed():
    transformers = pytest.importorskip('transformers')
    config = transformers.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = 'eager'
    model = transformers.LlamaForCausalLM(config).eval()
    _, probe = fixture()
    logits, capture = forward(model, probe['prefix_ids'], probe)
    original = model(torch.tensor([probe['prefix_ids']]), use_cache=False).logits[0]
    torch.testing.assert_close(logits, original, atol=1e-6, rtol=1e-5)
    assert max(capture.reconstruction) < 1e-5
    changed, _ = forward(model, probe['prefix_ids'], probe, (Intervention(0, ('evidence',)),))
    torch.testing.assert_close(changed[:4], original[:4], atol=1e-6, rtol=1e-5)
