"""Small native models and exact identities; no natural-label performance claims."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

from state_audit.functional_capture import capture_candidate, log_probability, rms_readout
from state_audit.functional_hooks import functional_hooks
from state_audit.functional_readout import contrast_weights, distribution_change, entropy_partition
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.storage import read_json, write_json
from experiments.native_support.functional_bank import answer_units, compile_bank
from experiments.native_support.functional_run import main


@pytest.fixture
def model():
    torch.manual_seed(37)
    torch.set_num_threads(1)
    return ModelAdapter(LlamaForCausalLM(LlamaConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)))


def test_meaning_expression_partition_and_changes():
    groups = np.array([0, 0, 1, 1])
    before = np.log([.25, .25, .25, .25])
    wording = np.log([.4, .1, .4, .1])
    meaning = np.log([.4, .4, .1, .1])
    assert entropy_partition(before, groups)["entropy_chain_error"] == pytest.approx(0, abs=1e-15)
    assert distribution_change(before, wording, groups)["meaning_kl"] == pytest.approx(0, abs=1e-15)
    assert distribution_change(before, meaning, groups)["expression_kl"] == pytest.approx(0, abs=1e-15)
    assert distribution_change(before, wording, groups)["expression_kl"] > .1


def test_contrast_derivative_is_group_log_odds():
    groups = np.array([0, 0, 1, 1])
    scores = np.array([-2., -3., -4., -1.])
    direction = np.array([.2, -.4, .1, .5])
    read = lambda x: logsumexp(x[:2]) - logsumexp(x[2:])
    measured = (read(scores + 1e-5 * direction) - read(scores - 1e-5 * direction)) / 2e-5
    assert contrast_weights(scores, groups, 0, 1) @ direction == pytest.approx(measured, abs=1e-10)


def test_rms_scaling_can_suppress_margin_without_direct_write(model):
    with torch.no_grad():
        model.native.lm_head.weight[:, -1] = 0
        before = torch.randn(2, 24)
        before[:, -1] = 0
        message = torch.zeros_like(before)
        message[:, -1] = 5
        logits = model.native.lm_head(model.native.model.norm(before + message))
        target = logits.argmax(-1)
        rows = rms_readout(model, before, message, target, logits)
    np.testing.assert_allclose(rows["rms_direct_margin"], 0, atol=1e-7)
    assert np.all(rows["rms_rescale_margin"] < 0)
    np.testing.assert_allclose(rows["rms_identity_error"], 0, atol=1e-7)


def test_capture_matches_native_sequence_and_root_directional_derivative(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13, 15]
    captured = capture_candidate(model, prefix, phrase, [0, 0, 1, 1], 2)
    ids = prefix + phrase[:-1]
    with attention_backend(model, "sdpa"):
        embeddings = model.native.model.embed_tokens(model.input_ids(ids)).detach()
        def objective(values):
            hidden = model.native.model(inputs_embeds=values, use_cache=False).last_hidden_state
            logits = model.native.lm_head(hidden[0, len(prefix) - 1:])
            return log_probability(logits, torch.tensor(phrase)).sum()
        with torch.no_grad():
            expected = float(objective(embeddings))
            direction = torch.zeros_like(embeddings)
            direction[0, 1] = embeddings[0, 1]
            finite = float((objective(embeddings + .01 * direction) - objective(embeddings - .01 * direction)) / .02)
    assert captured["log_probability"].sum() == pytest.approx(expected, abs=2e-6)
    assert captured["prefix_root_sensitivity"][1] == pytest.approx(finite, abs=2e-4)
    np.testing.assert_allclose(captured["head_total"], captured["head_residual"] + captured["head_ffn_mediated"], atol=1e-7)
    assert captured["head_total"].shape == (3, 2, 4, 3)
    assert captured["prefix_root_sensitivity"].shape == (4,)
    assert captured["branch_root_sensitivity"].shape == (2,)
    np.testing.assert_allclose(captured["head_attention"].sum(-1), 1, atol=2e-7)


def test_ffn_channel_matches_native_local_jacobian(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13]
    row = capture_candidate(model, prefix, phrase, [0, 0, 1, 1], 2)
    with attention_backend(model, "sdpa"), functional_hooks(model) as records:
        output = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]), use_cache=False)
        logits = model.native.lm_head(output.last_hidden_state[0, 3:])
        objective = log_probability(logits, torch.tensor(phrase)).sum()
        mid, write = records[0]["mid"], records[0]["ffn"]
        upstream, downstream = torch.autograd.grad(objective, (mid, write), retain_graph=True)
        via_ffn, = torch.autograd.grad(write, mid, grad_outputs=downstream, retain_graph=True)
        torch.testing.assert_close(upstream, downstream + via_ffn)
        head = records[0]["head"][0, 3].view(4, 6)
        direction = (via_ffn[0, 3] @ model.layers[0].self_attn.o_proj.weight).view(4, 6)
        expected = (head * direction).sum(-1).detach().numpy()
    np.testing.assert_allclose(row["head_ffn_mediated"][0, 0].sum(-1), expected, atol=1e-6)


def test_native_hook_resources_restore_on_failure(model):
    original = model.layers[0].mlp.forward
    with pytest.raises(RuntimeError), functional_hooks(model):
        raise RuntimeError("stop")
    assert model.layers[0].mlp.forward == original
    assert not model.layers[0].mlp._forward_hooks
    capture_candidate(model, [1, 5, 7], [9], [0, 0, 1], 2)
    assert all(parameter.requires_grad for parameter in model.native.parameters())
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_sliding_window_rows_match_native_attention():
    model = ModelAdapter(MistralForCausalLM(MistralConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, sliding_window=3)))
    prefix, phrase = [1, 5, 7, 9, 11], [13, 15]
    groups = list(range(len(prefix)))
    captured = capture_candidate(model, prefix, phrase, groups, len(prefix))
    with torch.no_grad(), attention_backend(model, "eager"):
        native = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]),
                                    use_cache=False, output_attentions=True)
    for layer, attention in enumerate(native.attentions):
        np.testing.assert_allclose(captured["head_attention"][:, layer],
            attention[0, :, 4:6].transpose(0, 1).numpy(), atol=2e-7)


class TinyTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join({1: 'Q ', 5: 'P ', 7: 'a ', 9: 'b.'}[value] for value in ids)

    def encode(self, text, **kwargs):
        return {'c.': [11], 'd.': [13], 'e.': [15]}[text]


def test_bank_preserves_observed_tokenization_and_positions():
    item = dict(id="a", prompt_length=2, token_ids=[1, 5, 7, 9], token_text=['Q ', 'P ', 'a ', 'b.'])
    assert list(answer_units(item)) == [(0, 2)]
    bank = compile_bank(item, TinyTokenizer(), 0, 2,
                        dict(paraphrase='c.', polarity=['d.', 'e.'], binding=[]))
    assert bank["candidates"][0]["token_ids"] == [7, 9]
    assert bank["prefix_ids"] == [1, 5]


def test_run_resume_report_and_pack_without_label_reads(tmp_path, model):
    source, destination = tmp_path / 'input', tmp_path / 'output'
    item = dict(id="a", source_id="s", prompt_length=2, token_ids=[1, 5, 7, 9],
                token_text=['Q ', 'P ', 'a ', 'b.'])
    settings = dict(model="tiny", responses=[item])
    write_json(source / 'settings.json', settings)
    write_json(source / 'value_transport/capture_settings.json', settings)
    write_json(source / 'value_transport/capture/0000/sources.json',
               dict(blocks=[dict(id='p')], group_ids=[0, 0, 1, 1]))
    # A malformed annotation file must never be opened by this audit.
    (source / 'annotations.json').write_text('NOT JSON')
    bank = compile_bank(item, TinyTokenizer(), 0, 2,
                        dict(paraphrase='c.', polarity=['d.', 'e.'], binding=[]))
    bank.update(valid=True, reason='fixture')
    def prepare(*args):
        write_json(args[-2] / 'bank.json', bank)
        return bank
    args = ['--input', str(source), '--output', str(destination), '--device', 'cpu', '--dtype', 'float32']
    with patch('state_audit.model.load_model', return_value=(model, TinyTokenizer())), \
         patch('experiments.native_support.functional_run.prepare_bank', side_effect=prepare):
        main(args)
    summary = read_json(destination / 'summary.json')
    assert summary['completed_units'] == 1 and summary['new_auroc'] is None
    assert summary['observed_tokens'] == 2
    assert (destination / 'head_contrasts.csv').is_file()
    assert (tmp_path / 'output_review.zip').is_file()
    assert (destination / 'annotations.json').read_text() == 'NOT JSON'
    with patch('state_audit.model.load_model', side_effect=AssertionError('loaded model')):
        main(args + ['--resume'])
        main(['--stage', 'report', '--output', str(destination)])
    assert read_json(destination / 'summary.json') == summary


def test_boundary_only_capture_preserves_shared_contrast_and_full_ffn(model):
    arguments = (model, [1, 5, 7], [9, 11, 13], [0, 0, 1], 2)
    full = capture_candidate(*arguments)
    compact = capture_candidate(*arguments, all_head_queries=False)
    for name in ('head_total', 'head_residual', 'head_ffn_mediated', 'head_attention'):
        np.testing.assert_allclose(compact[name], full[name][:1], atol=1e-7)
    for name in ('prefix_root_sensitivity', 'log_probability', 'ffn_write_sensitivity'):
        np.testing.assert_allclose(compact[name], full[name], atol=1e-7)
    np.testing.assert_array_equal(compact['head_query'], [2])
    np.testing.assert_array_equal(compact['query'], [2, 3, 4])
