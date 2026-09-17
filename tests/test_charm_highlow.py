"""Score tails are diagnostic, not labels; optional node x never loads edges."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_charm_structure_audit import fixture_files
from experiments.charm_structure_audit.score_groups import (
    score_tails, prepare_table, summarize_groups, fixed_pair_tokens,
    read_node_values, within_interval_contrasts, summarize_channels,
)
from experiments.charm_structure_audit.compare_fixed_scores import compare_fixed_scores
from experiments.charm_structure_audit.main import main


def test_tails_never_use_gold():
    frame = pd.DataFrame(dict(id=['a']*10, score=np.arange(10)/10, gold=[0]*5+[1]*5))
    first = score_tails(frame)
    frame['gold'] = 1-frame.gold
    second = score_tails(frame)
    assert first.score_tail.tolist() == second.score_tail.tolist()
    assert first.score_tail.tolist().count('high') == 2
    assert first.score_tail.tolist().count('low') == 2


def test_constant_scores_do_not_invent_extremes():
    result = score_tails(pd.DataFrame(dict(id=['a']*8, score=[.5]*8)))
    assert set(result.score_tail) == {'tied'}
    with pytest.raises(ValueError):
        score_tails(result, .5)


def test_quantile_ties_stay_together():
    result = score_tails(pd.DataFrame(dict(id=['a']*10, score=[0]*4+[1]*2+[2]*4)))
    assert list(result.score_tail[:4]) == ['low']*4
    assert list(result.score_tail[-4:]) == ['high']*4


def scored_fixture(tmp_path):
    root, prepared = fixture_files(tmp_path)
    path = root/'charm_in/test/tokens.csv'
    frame = pd.read_csv(path, keep_default_na=False)
    frame['score'] = .2
    frame.loc[frame.token % 2 == 0, 'score'] = .9
    frame['predicted'] = (frame.score > .5).astype(int)
    frame.to_csv(path, index=False)
    pairs = json.loads((root/'charm_in/test/cluster_audit/pairs.json').read_text())
    return root, prepared, pairs


def test_fixed_partner_is_not_selected_for_its_score(tmp_path):
    root, _, pairs = scored_fixture(tmp_path)
    table, _ = prepare_table(root/'charm_in', .2)
    paired = fixed_pair_tokens(table, pairs, 'cluster')
    assert paired.normal_token.tolist() == [8, 9, 10, 11]
    table.loc[table.token.between(8, 11), 'score'] = 1.
    changed = fixed_pair_tokens(table, pairs, 'cluster')
    assert paired.normal_token.tolist() == changed.normal_token.tolist()
    with pytest.raises(ValueError, match='reuses'):
        fixed_pair_tokens(table, pairs*2, 'cluster')


def test_node_reader_does_not_need_any_graph_edges(tmp_path):
    root, prepared, _ = scored_fixture(tmp_path)
    graph = prepared/'graphs/test/test.npz'
    with np.load(graph) as archive:
        values = {key: archive[key] for key in archive.files if key not in ['edge_attr', 'edge_index', 'edge_mark']}
    np.savez(graph, **values)
    table, _ = prepare_table(root/'charm_in', .2)
    attributes, geometry = read_node_values(prepared, table)
    assert geometry == (2, 2)
    assert attributes.shape == (32, 4)


def test_within_interval_controls_and_empty_coverage(tmp_path):
    root, prepared, _ = scored_fixture(tmp_path)
    table, _ = prepare_table(root/'charm_in', .2)
    values, shape = read_node_values(prepared, table)
    rows = within_interval_contrasts(values, table)
    assert rows
    frame = summarize_channels(rows, shape, 0)
    assert frame.low.isna().all() and frame.high.isna().all()
    table['score_tail'] = 'high'
    assert within_interval_contrasts(values, table) == []


def test_interval_and_source_weighting_not_token_weighting():
    rows = [dict(source='a', unit='x', contrast='c', observations=1, delta=np.array([1.]))]*10
    rows += [dict(source='b', unit='y', contrast='c', observations=100, delta=np.array([0.]))]
    result = summarize_channels(rows, (1, 1), 0)
    assert result.source_mean_delta.iloc[0] == .5
    assert result.intervals.iloc[0] == 2 and result.sources.iloc[0] == 2


def test_real_cli_all_counts_preserved_and_no_torch(tmp_path):
    root, prepared, _ = scored_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path/'out'
    code = "import runpy,sys;sys.argv=['audit']+sys.argv[1:];runpy.run_module('experiments.charm_structure_audit.main',run_name='__main__');assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code, '--mode', 'highlow', '--root', str(root),
        '--prepared', str(prepared), '--models', 'charm_in', '--node-values', '--bootstrap', '0',
        '--output', str(output)], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    groups = pd.read_csv(output/'charm_in/score_groups.csv')
    assert groups.tokens.sum() == 32
    assert (output/'charm_in/node_value_contrasts.csv.gz').exists()
    assert (output/'charm_in/figures/four_score_groups.png').exists()
    for path, content in before.items():
        assert path.read_bytes() == content


def fixed_fixture(tmp_path):
    from experiments.charm_structure_audit.compare_fixed_scores import METHODS
    root, prepared, _ = scored_fixture(tmp_path)
    table, _ = prepare_table(root/'charm_in', .2)
    fixed = tmp_path/'fixed'
    (fixed/'predictions').mkdir(parents=True)
    with np.load(prepared/'graphs/test/test.npz') as graph:
        ids = graph['token_ids']
        prompt = int(graph['prompt_length'])
    record = dict(id='test', source_id='testsource', file='000000.npz')
    scores = np.linspace(0, 1, len(table))
    scores[0] = np.nan
    np.savez(fixed/'predictions/000000.npz', record_json=np.array(json.dumps(record)),
        token_ids=ids, prompt_length=prompt, coverage=np.isfinite(scores), **{k:scores for k in METHODS})
    (fixed/'predictions/freeze.json').write_text(json.dumps(dict(complete=True, records=[record])))
    return fixed, prepared, table


def test_fixed_comparison_keeps_common_coverage_and_directions(tmp_path):
    fixed, prepared, table = fixed_fixture(tmp_path)
    output = tmp_path/'compare'
    output.mkdir()
    compare_fixed_scores(fixed, prepared, table, output, .2)
    report = json.loads((output/'fixed_comparison_protocol.json').read_text())
    assert report['all_tokens'] == 32 and report['common_tokens'] == 31
    frame = pd.read_csv(output/'fixed_score_comparison.csv')
    assert len(frame) == 7
    assert 'NOT removed' in report['target_alignment']


def test_fixed_comparison_refuses_different_token_ids(tmp_path):
    fixed, prepared, table = fixed_fixture(tmp_path)
    path = fixed/'predictions/000000.npz'
    with np.load(path) as archive:
        values = {k:archive[k] for k in archive.files}
    values['token_ids'][0] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match='token IDs'):
        compare_fixed_scores(fixed, prepared, table, tmp_path, .2)
