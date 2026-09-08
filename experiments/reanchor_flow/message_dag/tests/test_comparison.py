"""Control provenance, content coverage, and frozen comparisons under interruption."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow.attention_audit import special_token_mask
from experiments.reanchor_flow.tests.test_message_lineage import capture_fixture
from experiments.reanchor_flow.message_dag.cache import NativeCache, repair_mask
from experiments.reanchor_flow.message_dag.graph import Tape, prepare, trace_target
from experiments.reanchor_flow.message_dag.selection import content_positions


def test_tokenizer_decoder_and_legacy_control_mask_preserve_ordinary_inputs():
    tokenizer = SimpleNamespace(all_special_ids=[1], added_tokens_decoder={
        2: SimpleNamespace(special=True), 3: SimpleNamespace(special=False)})
    np.testing.assert_array_equal(special_token_mask(tokenizer, [1, 2, 3]), [True, True, False])
    trace = dict(token_ids=np.arange(7), special_mask=np.array([True, False, False, False, False, False, False]),
                 token_text=np.array(['BOS', '<|start_header_id|>', 'assistant', '<|end_header_id|>',
                                      '.', 'quote <|start_header_id|>', '<|reserved_special_token_8|>']))
    fixed = repair_mask(trace)
    np.testing.assert_array_equal(fixed['added_special_positions'], [1, 3, 6])
    np.testing.assert_array_equal(fixed['token_ids'], trace['token_ids'])
    np.testing.assert_array_equal(repair_mask(fixed)['added_special_positions'], [1, 3, 6])
    assert trace['special_mask'].sum() == 1
    assert not fixed['special_mask'][[2, 4, 5]].any()


def test_content_targets_use_word_starts_not_subwords_spaces_or_citations():
    pieces = ['prompt', 'Based', ' on', ' the', ' provided', ' passages', ',', ' But', 'cher', ' Shop',
              ':', ' (', '510', ')', ' 889', '-', '869', '0', '.', '\n', '1', '.', ' In', ' steel',
              ' nail', ' files', ' (', 'Pass', 'age', ' ', '1', ')', '\n', 'Unable', ' to', ' answer',
              ' based', ' on', ' given', ' passages', '.']
    trace = dict(token_ids=np.arange(len(pieces)), token_text=np.array(pieces), response_start=np.array(1),
                 row_position=np.arange(len(pieces)), special_mask=np.zeros(len(pieces), bool))
    targets = content_positions(trace)
    assert targets.tolist() == [7, 9, 12, 14, 16, 23, 24, 25]


def test_header_reclassification_changes_full_source_propagation_not_only_display(tmp_path):
    path, trace, weights = capture_fixture(tmp_path)
    trace['token_text'] = trace['token_text'].astype('<U40')
    trace['token_text'][2] = '<|start_header_id|>'
    np.savez_compressed(path, **trace)
    before = path.read_bytes()
    results = []
    for repaired in (False, True):
        with NativeCache(path, weights) as cache, TemporaryDirectory() as directory:
            if not repaired:  # Reproduce the legacy classification using the same native operands.
                cache.trace['special_mask'] = cache.trace['capture_special_mask']
            tape = Tape(cache, directory)
            try:
                prepare(cache, tape)
                results.append(trace_target(cache, tape, 7, edge_budget=50))
            finally:
                tape.close()
    old, new = results
    names = old['source_names'].tolist()
    other, special = names.index('other_prompt'), names.index('special')
    shift = old['root_position_credit'][2]
    assert abs(shift) > 1e-7
    np.testing.assert_allclose(new['source_output'][other], old['source_output'][other] - shift, atol=3e-7)
    np.testing.assert_allclose(new['source_output'][special], old['source_output'][special] + shift, atol=3e-7)
    for key in ('source_output', 'node_input', 'head_relay'):
        np.testing.assert_allclose(new[key][[other, special]].sum(0), old[key][[other, special]].sum(0), atol=3e-7)
        np.testing.assert_allclose(new[key][0], old[key][0], atol=3e-7)  # Material roots were unchanged.
    assert new['balance_error'].max() < 2e-5
    assert not np.isin(new['edge_index'][:, 2:], [2]).any()
    assert path.read_bytes() == before


def comparison_fixture(tmp_path, classes):
    path, trace, weights = capture_fixture(tmp_path)
    trace['token_text'] = np.array(['BOS', ' material', '<|start_header_id|>', ' source', ' boundary',
                                    ' red', ' Paris', ' Berlin', ' ', ' London', ' Rome', ' blue', ' Oslo'])
    np.savez_compressed(path, **trace)
    native = tmp_path / 'native'
    native.mkdir()
    entries = []
    for i, cls in enumerate(classes):
        relative = Path('train/QA') / f'{cls}{i}.npz'
        dest = native / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.npz', '.history.npz', '.qk.npz', '.states.npz'):
            shutil.copyfile(path.with_suffix(suffix), dest.with_suffix(suffix))
        labels = np.zeros(8, int)
        if cls == 'H': labels[2] = 1
        np.savez_compressed(dest.with_suffix('.labels.npz'), labels=labels)
        entries.append(dict(split='train', task_type='QA', sample_id=f'{cls}{i}', source_id=f'source{i}',
                            path=relative.as_posix(), response_tokens=8))
    original = dict(audit_schema=3, settings={'save_states':True, 'model':str(weights.directory)}, samples=entries)
    (native / 'index.json').write_text(json.dumps(original))
    return native, original


def test_all_normal_scope_fails_preflight_before_weights_or_outputs(tmp_path):
    from experiments.reanchor_flow.message_dag import run as entry
    native, original = comparison_fixture(tmp_path, ['N', 'N'])
    shutil.rmtree(Path(original['settings']['model']))
    argv = ['--audit', str(native), '--selection', 'paired', '--samples-per-group', '2', '--targets-per-sample', '3']
    with pytest.raises(ValueError, match='No supported N/H content comparison'):
        entry.run(entry.parser().parse_args(argv))
    assert not (native / 'message_dag_v2').exists()


def test_paired_run_resume_and_partial_report_keep_frozen_controls(tmp_path, monkeypatch):
    from experiments.reanchor_flow.message_dag import run as entry, selection
    native, original = comparison_fixture(tmp_path, ['N', 'H', 'N', 'H'])
    before = (native / 'index.json').read_bytes()
    argv = ['--audit', str(native), '--selection', 'paired', '--samples-per-group', '4',
            '--targets-per-sample', '3', '--device', 'cpu', '--edge-budget', '3', '--bootstrap', '0']
    result = entry.run(entry.parser().parse_args(argv))
    output = native / 'message_dag_v2'
    c = result['cohorts']['train/QA']
    assert (c['normal'], c['hallucinated'], c['matched']) == (10, 2, 2)
    assert c['normal_content'] == 10 and c['hallucinated_content'] == 2
    assert c['corrected_control_positions'] == 4 and result['comparison_ready']
    assert result['labels_used_for_selection'] and not result['labels_used_for_graph']
    assert result['detection_diagnostics'] == {} and result['detection_scope'] == 'not_run_for_label_selected_cases'
    assert (native / 'index.json').read_bytes() == before
    manifest = json.loads((output / 'index.json').read_text())
    hall = next(e for e in manifest['samples'] if e['comparison_class'] == 'H')
    folder = output / hall['folder']
    assert hall['audit_pairs'] == [[7, 6, 10]]
    assert all((folder / f'target_{t}.html').exists() for t in hall['targets'])
    assert 'src="target_7.html"' in (folder / 'index.html').read_text()
    frozen = (output / 'index.json').read_bytes()
    with pytest.raises(ValueError, match='DAG settings changed'):
        entry.run(entry.parser().parse_args([*argv, '--mlp-rule', 'up']))
    assert (output / 'index.json').read_bytes() == frozen
    def forbidden(*a, **kw): pytest.fail('resume must neither resample nor replay completed source propagation')
    monkeypatch.setattr(selection, 'inventory', forbidden)
    monkeypatch.setattr(entry, 'prepare', forbidden)
    resumed = entry.run(entry.parser().parse_args(argv))
    assert resumed['completed_targets'] == 12
    # Missing the fixed right control cannot cause a different normal target to be selected.
    (folder / 'target_10.npz').unlink()
    partial = entry.run(entry.parser().parse_args(['--phase', 'evaluate', '--output', str(output), '--bootstrap', '0']))
    assert partial['completed_targets'] == 11
    assert partial['cohorts']['train/QA']['matched'] == 1
    assert partial['cohorts']['train/QA']['missing_pairs'] == 1
