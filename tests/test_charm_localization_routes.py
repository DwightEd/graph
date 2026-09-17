"""Synthetic tests only: sentence offsets, counting denominators, channel axes."""

from pathlib import Path
from types import SimpleNamespace
import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_charm_structure_audit import fixture_files, graph_sample
from experiments.charm_structure_audit.sentences import sentence_intervals, map_offsets, locate_sentences
from experiments.charm_structure_audit.data import load_predictions, read_tables
from experiments.charm_structure_audit.positions import annotate
from experiments.charm_structure_audit.localization import diagnostic_intervals, interval_tokens, one_interval, summarize_intervals
from experiments.charm_structure_audit.routes import conditional_excess, history_js, query_index, measure_pair, METRICS, PHASES, summarize_routes
from experiments.charm_structure_audit.head_audit import channel_units, mask_channels, compare_unit
from experiments.charm_structure_audit.visualize import escaped_sentence
from experiments.charm_structure_audit.main import main


def test_sentences_do_not_split_decimal_or_common_abbreviation():
    text = 'Dr. Lee measured 15.3 inches. Is it true? Yes!'
    actual = [text[a:b] for a, b in sentence_intervals(text)]
    assert actual == ['Dr. Lee measured 15.3 inches.', 'Is it true?', 'Yes!']


def test_initials_quotes_paragraphs_and_cjk():
    text = 'The U.S. team said "yes."\n\nFirst paragraph\n\n第二句。 第三句！'
    actual = [text[a:b] for a, b in sentence_intervals(text)]
    assert actual == ['The U.S. team said "yes."', 'First paragraph', '第二句。', '第三句！']


def test_sentence_split_independent_of_scores_and_labels():
    text = 'One. Two.'
    assert sentence_intervals(text) == [(0, 4), (5, 9)]
    offsets = np.asarray([[0, 3], [3, 4], [5, 8], [8, 9], [0, 0]])
    ids, cross = map_offsets(text, offsets, sentence_intervals(text))
    assert ids.tolist() == [0, 0, 1, 1, -1]
    assert not cross.any()


def test_overlapping_offsets_share_sentence_and_crossing_is_disclosed():
    text = 'One. Two.'
    offsets = np.array([[0, 3], [0, 3], [3, 8], [4, 5]])
    ids, cross = map_offsets(text, offsets, sentence_intervals(text))
    assert ids.tolist() == [0, 0, 1, -1]
    assert cross.tolist() == [False, False, True, False]


def test_render_does_not_duplicate_characters_with_overlapping_offsets():
    import html
    import re
    text = 'abc.'
    group = pd.DataFrame(dict(char_start=[0, 0, 3], char_end=[3, 3, 4], token=[0, 1, 2],
        outcome=['TP', 'FN', 'TN'], score=[.9, .2, .1], offset=[0, 1, -1], sentence_offset=[0, 1, 2]))
    rendered = escaped_sentence(text, group, 0, len(text))
    assert html.unescape(re.sub(r'<[^>]*>', '', rendered)) == text
    assert 'mixed' in rendered


def localization_fixture(tmp_path):
    root, prepared = fixture_files(tmp_path)
    samples = load_predictions(root / 'charm_in/test')
    table, spans = read_tables(root / 'charm_in/test')
    table.loc[:, 'score'] = .1
    table.loc[(table.token >= 12) & (table.token < 16), 'score'] = [.1, .9, .9, .9]
    table.loc[table.token == 9, 'score'] = .9
    table['predicted'] = (table.score > .5).astype(int)
    table = annotate(table, spans)
    table, sentences = locate_sentences(table, samples)
    pair = dict(id='test', source_id='testsource', tier='cluster', error_start=12, normal_start=8, length=4)
    return table, samples, pair


def test_span_success_counts_do_not_expand_labels(tmp_path):
    table, samples, pair = localization_fixture(tmp_path)
    intervals = diagnostic_intervals(table, [pair], 'cluster')
    tokens = interval_tokens(table, intervals)
    spans = pd.DataFrame([one_interval(g, str(samples[0]['response']))
                         for _, g in tokens.groupby(['population', 'id', 'start'])])
    summary = summarize_intervals(spans).set_index('population')
    assert summary.loc['gold_error', 'tokens'] == 4
    assert summary.loc['gold_error', 'alarm_tokens'] == 3
    assert summary.loc['gold_error', 'any_alarm'] == 1
    assert summary.loc['gold_error', 'alarm80'] == 0
    assert summary.loc['gold_error', 'all_alarm'] == 0
    assert summary.loc['matched_normal', 'success_count'] == 0
    assert summary.loc['matched_normal', 'token_fpr'] == .25
    assert summary.loc['normal_run', 'spans'] == 2
    assert summary.loc['normal_run', 'success_count'] == 1


def test_population_counts_do_not_double_count_normal_match(tmp_path):
    table, _, pair = localization_fixture(tmp_path)
    duplicate = [pair, dict(pair, error_start=12)]
    intervals = diagnostic_intervals(table, duplicate, 'cluster')
    with pytest.raises(ValueError, match='Overlapping'):
        interval_tokens(table, intervals)


def test_sentence_locations_reject_wrong_original_text(tmp_path):
    root, _ = fixture_files(tmp_path)
    samples = load_predictions(root / 'charm_in/test')
    table, _ = read_tables(root / 'charm_in/test')
    table.loc[0, 'text'] = 'invented'
    with pytest.raises(ValueError, match='CSV text'):
        locate_sentences(table, samples)


def test_sentence_locations_reject_different_source(tmp_path):
    root, _ = fixture_files(tmp_path)
    samples = load_predictions(root / 'charm_in/test')
    table, _ = read_tables(root / 'charm_in/test')
    table['source_id'] = 'wrong'
    with pytest.raises(ValueError, match='source ID'):
        locate_sentences(table, samples)


def test_pure_lag1_cannot_fake_excess_history_reuse():
    ids = np.arange(11)
    excess, no_copy = conditional_excess(np.array([9]), np.array([[.8, .4]]), 10, 7, 11, 2, ids)
    np.testing.assert_allclose(excess, 0)
    np.testing.assert_allclose(no_copy, 0)


def test_excess_matches_fixed_lag_exact_expectation():
    ids = np.arange(11)
    excess, no_copy = conditional_excess(np.array([7, 8, 9]), np.array([[.6], [.2], [.1]]), 10, 7, 11, 2, ids)
    np.testing.assert_allclose(excess, [.3])
    np.testing.assert_allclose(no_copy, [.3])


def test_current_token_copy_is_controlled_not_counted_as_noncopy_lockin():
    ids = np.arange(11)
    ids[7] = ids[10]
    excess, no_copy = conditional_excess(np.array([7]), np.array([[.6]]), 10, 7, 11, 2, ids)
    np.testing.assert_allclose(excess, 0)
    np.testing.assert_allclose(no_copy, 0)


def js_graph():
    return dict(prompt_length=np.array(1), x=np.zeros((6, 2)), layers=np.array(1), heads=np.array(2),
        edge_index=np.array([[1, 2, 1, 2, 3], [3, 3, 4, 4, 4]]),
        edge_attr=np.array([[.2, .0], [.2, .0], [.1, .0], [.1, .0], [.8, .9]]))


def test_js_excludes_new_key_and_preserves_missing_head():
    graph = js_graph()
    result = history_js(graph, query_index(graph), 4)
    assert result[0] == pytest.approx(0.)
    assert np.isnan(result[1])


def test_js_detects_different_common_history_distribution():
    graph = js_graph()
    graph['edge_attr'][2:4, 0] = [.2, .0]
    assert history_js(graph, query_index(graph), 4)[0] > 0


def test_route_axes_and_common_phase_coverage():
    graph, sample = graph_sample()
    pair = dict(error_start=12, normal_start=8, length=4)
    values, counts = measure_pair(graph, query_index(graph), sample, pair, 5)
    assert values.shape == (2, len(PHASES), len(METRICS), 2, 2)
    np.testing.assert_array_equal(counts[0], counts[1])
    # The normal window's post context overlaps the error, so no post data.
    assert not counts[:, PHASES.index('post')].any()
    assert np.isnan(values[:, PHASES.index('post')]).all()
    np.testing.assert_allclose(values[:, -1], values[:, 1] - values[:, 0], equal_nan=True)


def test_channel_mask_keeps_separate_original_layers_and_heads():
    graph, _ = graph_sample()
    graph['x'][:] = np.arange(4)[None] + 1
    before = copy.deepcopy(graph)
    view, counts = mask_channels(graph, [3], 'node')
    assert np.all(view['x'][:, 3] == 0)
    np.testing.assert_array_equal(view['x'][:, :3], graph['x'][:, :3])
    np.testing.assert_array_equal(view['edge_attr'], graph['edge_attr'])
    assert counts['changed_node_cells'] == len(graph['x'])
    for key in graph:
        np.testing.assert_array_equal(graph[key], before[key])


def test_edge_mask_does_not_recompute_edges_or_touch_target_attributes():
    graph, _ = graph_sample()
    view, counts = mask_channels(graph, [0, 1], 'edge')
    assert not view['edge_attr'][:, :2].any()
    assert counts['changed_edge_cells'] > 0
    np.testing.assert_array_equal(view['x'], graph['x'])
    assert view['edge_index'] is graph['edge_index']


def test_whole_layer_vs_individual_channel_selection():
    args = SimpleNamespace(llm_layers=[1], channels=None, channel_unit='layer', channel_sites=['node', 'edge'], channel_operations=['zero'])
    units = channel_units(2, 3, args)
    assert units[0]['columns'] == [3, 4, 5] and len(units) == 2
    args.channel_unit = 'head'
    assert len(channel_units(2, 3, args)) == 6
    args.channels = ['0:2']
    assert channel_units(2, 3, args)[0]['columns'] == [2]
    args.channels = ['2:0']
    with pytest.raises(ValueError, match='outside'):
        channel_units(2, 3, args)


def test_masked_scores_both_shift_is_not_ranking_gain(tmp_path):
    table, _, pair = localization_fixture(tmp_path)
    unit = dict(name='node_L0', site='node', operation='zero', llm_layer=0, llm_head=-1)
    result, pairs, roles = compare_unit(table, table.score.to_numpy() - .05, [pair], .5, unit, 0)
    assert result['auroc_delta'] == 0
    assert result['matched_auc_delta'] == 0
    assert result['matched_margin_delta'] == pytest.approx(0.)
    assert sum(r['tokens'] for r in roles) == len(table)


def test_single_source_route_interval_not_false_certainty(tmp_path):
    values = np.zeros((2, len(PHASES), len(METRICS), 2, 2))
    entries = [dict(id='a', source_id='s', error_start=2, values=values)]
    frame = summarize_routes(entries, tmp_path, 3)
    assert frame.low.isna().all() and frame.high.isna().all()


def test_source_balanced_route_means_not_token_replicates(tmp_path):
    zero = np.zeros((2, len(PHASES), len(METRICS), 2, 2))
    one = zero.copy()
    one[0] = 1.
    entries = [dict(id=str(i), source_id='s1', error_start=2, values=one) for i in range(4)]
    entries += [dict(id='other', source_id='s2', error_start=2, values=zero)]
    frame = summarize_routes(entries, tmp_path, 0)
    np.testing.assert_allclose(frame.source_mean_delta, .5)
    assert set(frame.pairs) == {5} and set(frame.sources) == {2}


def test_actual_new_modes_and_unchanged_inputs(tmp_path):
    root, prepared = fixture_files(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    common = ['--root', str(root), '--prepared', str(prepared), '--device', 'cpu', '--bootstrap', '0', '--models', 'charm_in']
    for mode in ('locate', 'routes', 'heads'):
        main(common + ['--mode', mode, '--output', str(tmp_path / mode), '--llm-layers', '0'])
    for path, state in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state
    assert (tmp_path / 'locate/charm_in/span_counts.csv').exists()
    assert (tmp_path / 'locate/charm_in/figures/span_counts.svg').exists()
    assert (tmp_path / 'routes/layer_head_routes.csv.gz').exists()
    assert (tmp_path / 'heads/channel_effects.csv').exists()
    assert (tmp_path / 'heads/scores/test.npz').exists()
    main(common + ['--mode', 'heads', '--output', str(tmp_path / 'heads'), '--llm-layers', '0'])


def test_locate_cli_does_not_load_torch_or_embeddings(tmp_path):
    root, prepared = fixture_files(tmp_path)
    code = "import runpy,sys;sys.argv=['main']+sys.argv[1:];runpy.run_module('experiments.charm_structure_audit.main',run_name='__main__');assert 'torch' not in sys.modules"
    done = subprocess.run([sys.executable, '-c', code, '--mode', 'locate', '--root', str(root),
        '--models', 'charm_in', '--output', str(tmp_path / 'out'), '--bootstrap', '0'], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_routes_refuse_empty_requested_tier(tmp_path):
    root, prepared = fixture_files(tmp_path)
    with pytest.raises(ValueError, match='no matched pairs'):
        main(['--mode', 'routes', '--root', str(root), '--prepared', str(prepared),
              '--output', str(tmp_path / 'empty'), '--pair-tier', 'cluster_heads'])


def test_same_pair_both_sides_success_criteria(tmp_path):
    from experiments.charm_structure_audit.localization import paired_detection_counts
    table, samples, pair = localization_fixture(tmp_path)
    intervals = diagnostic_intervals(table, [pair], 'cluster')
    tokens = interval_tokens(table, intervals)
    spans = pd.DataFrame([one_interval(g, str(samples[0]['response']))
                         for _, g in tokens.groupby(['population', 'id', 'start'])])
    result = paired_detection_counts(spans)
    assert result['pairs'] == 1 and result['error_any_hit'] == 1
    assert result['normal_completely_clear'] == 0
    assert result['both_error_any_and_normal_clear'] == 0


def test_no_matched_controls_has_zero_denominator_not_substitute(tmp_path):
    from experiments.charm_structure_audit.localization import paired_detection_counts
    table, samples, _ = localization_fixture(tmp_path)
    intervals = diagnostic_intervals(table, [], 'cluster')
    tokens = interval_tokens(table, intervals)
    spans = pd.DataFrame([one_interval(g, str(samples[0]['response']))
                         for _, g in tokens.groupby(['population', 'id', 'start'])])
    assert paired_detection_counts(spans)['pairs'] == 0


def test_future_route_edge_rejected_at_input_boundary():
    graph, _ = graph_sample()
    graph['edge_index'][0, 0] = graph['edge_index'][1, 0] + 1
    with pytest.raises(ValueError, match='past-to-response'):
        query_index(graph)


def test_head_geometry_not_silently_inferred_from_width():
    graph, _ = graph_sample()
    graph['layers'] = np.array(3)
    with pytest.raises(ValueError, match='channel dimension'):
        query_index(graph)


def test_chinese_sentences_without_spaces():
    text = '这是第一句。第二句！第三句？'
    assert [text[a:b] for a, b in sentence_intervals(text)] == ['这是第一句。', '第二句！', '第三句？']


def test_scoped_head_permutation_keeps_other_layers_and_marginals():
    from experiments.charm_structure_audit.head_audit import alter_channels
    graph, _ = graph_sample()
    graph['edge_index'] = np.array([[2, 3, 4, 5], [10, 10, 10, 10]])
    graph['edge_attr'] = np.array([[.1, .2, .3, .4], [.2, .3, .4, .5], [.3, .4, .5, .6], [.4, .5, .6, .7]])
    for operation in ('coupled', 'independent'):
        unit = dict(operation=operation, columns=[0, 1], site='edge')
        view, info = alter_channels(graph, unit, 42)
        np.testing.assert_array_equal(view['edge_attr'][:, 2:], graph['edge_attr'][:, 2:])
        np.testing.assert_allclose(view['edge_attr'].sum(axis=0), graph['edge_attr'].sum(axis=0))
        np.testing.assert_array_equal(view['edge_index'], graph['edge_index'])
        assert info['changed_node_cells'] == 0


def test_one_channel_independent_and_coupled_are_same_control():
    from experiments.charm_structure_audit.head_audit import alter_channels
    graph, _ = graph_sample()
    a, _ = alter_channels(graph, dict(operation='coupled', columns=[1], site='edge'), 42)
    b, _ = alter_channels(graph, dict(operation='independent', columns=[1], site='edge'), 42)
    np.testing.assert_array_equal(a['edge_attr'], b['edge_attr'])


def test_actual_coupled_vs_independent_scoped_cli(tmp_path):
    root, prepared = fixture_files(tmp_path)
    output = tmp_path / 'relations'
    main(['--mode', 'heads', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu',
          '--output', str(output), '--llm-layers', '0', '--channel-sites', 'edge',
          '--channel-operations', 'coupled', 'independent', '--bootstrap', '0'])
    table = pd.read_csv(output / 'independent_minus_coupled.csv')
    assert set(table.region) >= {'all', 'first', 'interior'}
    assert set(table.llm_layer) == {0}
