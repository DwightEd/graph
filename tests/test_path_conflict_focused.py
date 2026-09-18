"""Real small-GQA interventions: same-prefix metrics, disjoint sources and mediation."""

from dataclasses import asdict
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_path_conflict import fixture, WordTokenizer
from experiments.path_conflict.native import Intervention, forward
from experiments.path_conflict.operators import vocabulary_logits, equal_norm_change
from experiments.path_conflict.scoring import evaluate_candidates, read_prefix
from experiments.path_conflict.focused_plan import Trial, focused_trials, SOURCE_ROLES
from experiments.path_conflict.focused_inputs import partition_sources, candidate_panels
from experiments.path_conflict.focused import run_panel, match_cut_norms
from experiments.path_conflict.focused_report import report_focused, derived_contrasts


def focused_fixture():
    model, probe = fixture()
    roles = dict(scope=np.array([0]), supported_value=np.array([1]), value_source=np.array([2]))
    probe = dict(probe, groups=partition_sources(3, 5, roles, 1), panel='natural',
                 forced_common_tokens=0, candidate_texts=[' correct', ' wrong'],
                 trace_heads=((0, 1), (1, 2), (2, 1)), trace_window=3)
    return model, probe


def test_history_groups_are_disjoint_from_query_and_cover_all_keys():
    roles = dict(scope=np.array([0]), supported_value=np.array([1]))
    for prompt, count in [(3, 3), (3, 5), (3, 30)]:
        groups = partition_sources(prompt, count, roles, 10)
        joined = np.concatenate(list(groups.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(count))
        assert groups['query_self'].tolist() == [count - 1]
        assert count - 1 not in groups['recent_history']
        assert len(groups['recent_history']) <= 10


def test_wording_control_puts_intervention_at_first_differing_token():
    tokenizer = WordTokenizer()
    case_id = '14315_headwear_scope'
    prompt = ' | '.join(q for quotes in SOURCE_ROLES[case_id].values() for q in quotes)
    prompt_ids = tokenizer.encode(prompt)
    native = dict(prefix_ids=prompt_ids + tokenizer.encode(' They wore'), prompt_length=len(prompt_ids))
    case = dict(case_id=case_id, candidates=[' caps with cloth', ' a headdress with feathers'])
    natural, parallel = candidate_panels(tokenizer, native, case)
    assert natural['prefix_ids'] == native['prefix_ids']
    assert parallel['forced_common_tokens'] > 0
    assert tokenizer.decode(parallel['prefix_ids']).endswith(' wore a')
    assert tokenizer.decode(parallel['candidates'][0]).startswith(' cap')
    assert tokenizer.decode(parallel['candidates'][1]).startswith(' headdress')
    assert parallel['candidates'][0][0] != parallel['candidates'][1][0]
    assert 'Only the Inca could wear' == tokenizer.decode(parallel['prefix_ids'][i] for i in parallel['groups']['scope']).strip()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_first_probabilities_and_margin_are_one_exact_readout(dtype):
    model, probe = focused_fixture()
    model.to(dtype)
    score, _ = evaluate_candidates(model, probe)
    assert score['next_margin'] == score['correct_first_logp'] - score['wrong_first_logp']
    assert score['correct_token_logps'][0] == score['correct_first_logp']
    assert score['wrong_token_logps'][0] == score['wrong_first_logp']
    assert score['sequence_margin'] == pytest.approx(score['next_margin'] + score['tail_margin'])


def test_vocab_chunked_projection_matches_full_fp32_multiply():
    model, probe = focused_fixture()
    _, run = forward(model, probe['prefix_ids'], probe)
    values = run.final_normalized
    with torch.inference_mode():
        expected = values.float() @ model.lm_head.weight.float().T
        actual = vocabulary_logits(model, values, 7)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)


def test_multiple_same_layer_actions_are_not_silently_overwritten():
    model, probe = focused_fixture()
    a = Intervention(0, ('scope',), heads=(1,))
    b = Intervention(0, ('supported_value',), heads=(1,))
    joined = Intervention(0, ('scope', 'supported_value'), heads=(1,))
    individual, run = forward(model, probe['prefix_ids'], probe, (a, b))
    combined, _ = forward(model, probe['prefix_ids'], probe, (joined,))
    torch.testing.assert_close(individual, combined, atol=1e-6, rtol=1e-6)
    assert len(run.changes) == 2


def test_zero_dose_and_self_source_have_the_declared_scope():
    model, probe = focused_fixture()
    base, _ = forward(model, probe['prefix_ids'], probe)
    null, _ = forward(model, probe['prefix_ids'], probe, (Intervention(0, ('scope',), dose=0),))
    torch.testing.assert_close(null, base, rtol=0, atol=0)
    changed, _ = forward(model, probe['prefix_ids'], probe, (Intervention(0, ('query_self',)),))
    torch.testing.assert_close(changed[:-1], base[:-1])
    assert not torch.equal(changed[-1], base[-1])


def test_mlp_restoration_identity_and_response_to_an_upstream_cut():
    model, probe = focused_fixture()
    base, baseline = forward(model, probe['prefix_ids'], probe)
    restore = dict(layer=2, site='mlp', states=baseline.baseline_mlps[2])
    identity, _ = forward(model, probe['prefix_ids'], probe, restore=restore)
    torch.testing.assert_close(identity, base, rtol=0, atol=0)
    cut = (Intervention(1, ('query_self',)),)
    natural, _ = forward(model, probe['prefix_ids'], probe, cut)
    held, _ = forward(model, probe['prefix_ids'], probe, cut, restore)
    assert not torch.equal(natural[-1], held[-1])
    torch.testing.assert_close(natural[:-1], held[:-1])


def test_final_layer_cut_does_not_create_later_token_recovery():
    model, probe = focused_fixture()
    base, _ = evaluate_candidates(model, probe)
    cut, _ = evaluate_candidates(model, probe, (Intervention(2, ('query_self',)),))
    for name in ('correct', 'wrong'):
        np.testing.assert_allclose(cut[name + '_token_logps'][1:], base[name + '_token_logps'][1:], atol=1e-7)
    assert cut['tail_margin'] == pytest.approx(base['tail_margin'], abs=1e-7)
    assert abs(cut['next_margin'] - base['next_margin']) > 1e-6


def test_early_cut_can_change_later_candidate_conditionals():
    model, probe = focused_fixture()
    base, _ = evaluate_candidates(model, probe)
    cut, _ = evaluate_candidates(model, probe, (Intervention(0, ('query_self',)),))
    assert abs(cut['tail_margin'] - base['tail_margin']) > 1e-6


def test_baseline_routes_include_identity_value_energy_and_local_write_effect():
    model, probe = focused_fixture()
    _, run = forward(model, probe['prefix_ids'], probe)
    for layer, head in probe['trace_heads']:
        key = f'L{layer}H{head}'
        for row, query in enumerate(run.routes[key + '_queries']):
            assert not run.routes[key + '_attention'][row, query + 1:].any()
            assert run.routes[key + '_attention'][row].sum() == pytest.approx(1.0, abs=1e-6)
        assert run.routes[key + '_value_norm'].shape == (5,)
        assert run.routes[key + '_source_write_norm'].shape == (5,)
        assert run.routes[key + '_source_lens_support'].shape == (5,)


def test_random_layer_dose_matches_the_corresponding_actual_cut(tmp_path):
    path = tmp_path / 'parent.npz'
    changes = [dict(layer=0, query_change_norm=2.0), dict(layer=1, query_change_norm=3.0)]
    np.savez(path, changes=json.dumps(changes))
    trial = Trial('random', (Intervention(0, ('scope',)), Intervention(1, ('scope',))), kind='random')
    matched = match_cut_norms(trial, path)
    assert [a.reference_norm for a in matched.actions] == [2.0, 3.0]
    for action in matched.actions:
        changed = equal_norm_change(torch.ones(1, 16) * .02, 0, action.reference_norm)
        assert float(changed.norm()) == pytest.approx(action.reference_norm, abs=5e-7)


def test_plan_has_controls_single_joint_dose_and_ordered_restoration():
    plan = focused_trials()
    assert len(plan) == len({trial.name for trial in plan})
    seen = set()
    for trial in plan:
        assert all(parent in seen for parent in trial.parents)
        if trial.restore_layer is not None:
            assert all(item.layer <= trial.restore_layer for item in trial.actions)
        seen.add(trial.name)
    assert {'early_22', 'early_23', 'early_joint', 'early_joint_half', 'history_hold_mlp31'} <= seen


def test_exact_joint_and_restore_differences_do_not_claim_additivity():
    rows = []
    for name, kind, parents, value in [('full', 'baseline', [], 3), ('a', 'cut', [], 2),
        ('b', 'cut', [], 4), ('ab', 'joint', ['a', 'b'], 5), ('restore', 'restore', ['a'], 3)]:
        rows.append(dict(case_id='x', side='supported', panel='natural', variant=name, kind=kind,
            parents=parents, **{m: value for m in ['next_margin', 'correct_first_logp', 'wrong_first_logp', 'sequence_margin', 'tail_margin']}))
    measured = derived_contrasts(pd.DataFrame(rows)).set_index('variant')
    assert measured.loc['ab', 'interaction_next_margin'] == 2
    assert measured.loc['restore', 'minus_parent_next_margin'] == 1


def test_real_focused_execution_resume_and_report(tmp_path):
    model, probe = focused_fixture()
    output = tmp_path
    (output / 'runs').mkdir()
    (output / 'routes').mkdir()
    schedule = [Trial('full', kind='baseline'), Trial('early_joint', (Intervention(0, ('scope',)),)),
                Trial('late_history', (Intervention(2, ('query_self',)),)),
                Trial('early_restore', (Intervention(0, ('scope',)),), 'restore', 2, 'mlp', parents=('early_joint',))]
    class DisplayTokenizer:
        def decode(self, tokens, **kwargs):
            return ' '.join(str(t) for t in tokens)
    identities, sources = [], []
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = SimpleNamespace()
    for side in ['supported', 'unsupported']:
        identity, rows = run_panel(args, model, DisplayTokenizer(), dict(case_id='test', source_id='1'),
                                   side, probe, schedule, output)
        identities.append(identity)
        sources.extend(rows)
    capture = next((output / 'runs').glob('*full.npz'))
    stamp = capture.stat().st_mtime_ns
    run_panel(args, model, DisplayTokenizer(), dict(case_id='test', source_id='1'), 'supported', probe, schedule, output)
    assert capture.stat().st_mtime_ns == stamp
    (output / 'focused_config.json').write_text(json.dumps(dict(cases=[dict(case_id='test')], schedule=[asdict(t) for t in schedule])))
    (output / 'panels.json').write_text(json.dumps(identities))
    pd.DataFrame(sources).to_csv(output / 'source_tokens.csv.gz', index=False)
    report_focused(output)
    assert json.loads((output / 'status.json').read_text())['complete']
    assert (output / 'source_routes.html').is_file()
    assert (output / 'path_conflict_review.tar.gz').is_file()
    assert (output / 'figures/test_natural_effects.png').is_file()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_half_dose_halves_the_actual_attention_output_change():
    model, probe = focused_fixture()
    outputs = []
    for dose in (0.0, 0.5, 1.0):
        snapshots = []
        from experiments.path_conflict.native import NativeRun
        action = Intervention(0, ('scope',), heads=(1,), dose=dose)
        with torch.inference_mode(), NativeRun(model, 5, probe['groups'], [7, 9], (action,)):
            handle = model.model.layers[0].self_attn.register_forward_hook(
                lambda module, inputs, output: snapshots.append(output[0].clone()))
            try:
                model(torch.tensor([probe['prefix_ids']]))
            finally:
                handle.remove()
        outputs.append(snapshots[0])
    torch.testing.assert_close(outputs[0] - outputs[1], .5 * (outputs[0] - outputs[2]), atol=1e-7, rtol=1e-5)
