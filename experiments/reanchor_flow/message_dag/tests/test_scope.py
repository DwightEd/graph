"""Interrupted captures must expose usable scopes before any model access."""
import json

import numpy as np
import pytest

from experiments.reanchor_flow.attention_audit_run import analysis_manifest
from experiments.reanchor_flow.message_dag import run as entry


def interrupted_cache(directory, *, ready=True, test_planned=True):
    samples = [dict(split='train', task_type='QA', sample_id='10', source_id='s10',
                    path='train/QA/10.npz', response_tokens=3, resumed=False)]
    if test_planned:
        samples.append(dict(split='test', task_type='QA', sample_id='20', source_id='s20',
                            path='test/QA/20.npz', response_tokens=3, resumed=False))
    if ready:
        path = directory / samples[0]['path']
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, token_ids=np.arange(5), row_position=np.arange(1,5),
                            special_mask=np.zeros(5,bool), response_start=np.array(2))
        for suffix in ('.history.npz', '.qk.npz', '.states.npz'):
            np.savez_compressed(path.with_suffix(suffix), sentinel=np.array(1))
    original = dict(audit_schema=3, settings=dict(save_states=True, full_attention=True,
                    model=str(directory/'absent-model')), samples=samples)
    (directory/'index.json').write_text(json.dumps(original))
    return original


@pytest.mark.parametrize('test_planned', [True, False])
def test_empty_requested_scope_reports_available_and_all_split_uses_train(tmp_path, test_planned):
    original = interrupted_cache(tmp_path, test_planned=test_planned)
    before = (tmp_path/'index.json').read_bytes()
    argv = ['--audit',str(tmp_path),'--completed-only','--task','QA']
    with pytest.raises(ValueError, match='split=test, task=QA') as error:
        entry.run(entry.parser().parse_args([*argv,'--split','test']))
    assert 'train/QA=1' in str(error.value)
    assert '--split all --task QA' in str(error.value)
    selected = entry.plan(entry.parser().parse_args([*argv,'--split','all']), original)
    assert [(e['split'],e['sample_id']) for e in selected['samples']] == [('train','10')]
    assert selected['samples'][0]['targets'] == [2,3,4]
    assert (tmp_path/'index.json').read_bytes() == before
    assert not (tmp_path/'message_dag_v2').exists()


@pytest.mark.parametrize('ready', [True, False])
def test_list_available_needs_no_model_npz_reads_or_output(tmp_path, monkeypatch, capsys, ready):
    interrupted_cache(tmp_path, ready=ready)
    before = (tmp_path/'index.json').read_bytes()
    def forbidden(*a, **kw): pytest.fail('listing must only inspect file existence')
    monkeypatch.setattr(np, 'load', forbidden)
    result = entry.run(entry.parser().parse_args(['--audit',str(tmp_path),'--list-available']))
    assert result['analysis_coverage']['completed_samples'] == int(ready)
    assert 'test/QA' in capsys.readouterr().out
    assert (tmp_path/'index.json').read_bytes() == before
    assert not (tmp_path/'message_dag_v2').exists()


def test_required_states_and_scope_local_strictness(tmp_path):
    original = interrupted_cache(tmp_path)
    # Optional full attention is not read by the DAG. Unfinished test samples
    # must not block an explicitly selected, complete train scope.
    args = entry.parser().parse_args(['--audit',str(tmp_path),'--split','train','--task','QA'])
    assert len(entry.plan(args, original)['samples']) == 1
    args.split = 'all'
    with pytest.raises(ValueError, match='add --completed-only'):
        entry.plan(args, original)
    (tmp_path/'train/QA/10.states.npz').unlink()
    args.completed_only = True
    with pytest.raises(ValueError, match='Available completed scopes: none'):
        entry.plan(args, original)
    # Existing audit callers retain their empty-cache failure by default.
    with pytest.raises(ValueError, match='no completed samples'):
        analysis_manifest(tmp_path, original, completed_only=True)
