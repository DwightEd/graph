"""Validate interventions against real model algebra, not a fabricated lock-in label."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_charm_structure_audit import graph_sample, model, fixture_files
from experiments.charm_structure_audit.lockin_forward import (
    factual_states, interval_view, span_forward, aggregate_groups,
    pair_forward, random_internal_mask, last_swap)
from experiments.charm_structure_audit.lockin_report import components, report
from experiments.charm_structure_audit.lockin import pair_metadata, validate_pair_set
from experiments.charm_structure_audit.main import main
from experiments.charm_structure_audit.model import degree


def full_graph_cut(net, graph, start, end, kind, pulse=None):
    """Independent whole-graph intervention to test pruning, not the production loop."""
    state = net.in_proj(torch.as_tensor(graph['x'])).relu()
    source, target = graph['edge_index']
    inside_target = (target >= start) & (target < end)
    inside_source = source >= start
    mask = inside_target & (inside_source if kind == 'internal' else ~inside_source)
    if kind == 'all':
        mask = inside_target
    states = [state]
    for step, layer in enumerate(net.mp_layers):
        inputs = torch.cat((state[source], torch.as_tensor(graph['edge_attr']),
                            torch.as_tensor(graph['edge_mark'])), dim=1)
        message = layer.msg_mlp(inputs)
        if pulse is None or pulse == step:
            message = message * torch.as_tensor(~mask)[:, None]
        total = torch.zeros_like(state).index_add(0, torch.as_tensor(target), message)
        state = layer.update(state, total / torch.as_tensor(degree(graph))[:, None])
        states.append(state)
    return net.pred(state).view(-1), states


@pytest.mark.parametrize('kind', ['internal', 'external', 'all'])
@pytest.mark.parametrize('pulse', [None, 0, 1])
def test_pruned_intervention_equals_whole_graph(kind, pulse):
    graph, _ = graph_sample()
    net = model()
    with torch.no_grad():
        factual = factual_states(net, graph)
        view = interval_view(graph, 14, 18)
        mask = view['internal'] if kind == 'internal' else ~view['internal']
        if kind == 'all':
            mask = np.ones(len(mask), bool)
        got = span_forward(net, factual, view, mask, pulse=pulse)
        expected, _ = full_graph_cut(net, graph, 14, 18, kind, pulse)
    torch.testing.assert_close(got['logits'], expected[14:18], atol=2e-6, rtol=1e-5)


def test_factual_and_local_replay_exact():
    graph, _ = graph_sample()
    net = model()
    with torch.no_grad():
        states = factual_states(net, graph)
        torch.testing.assert_close(net.pred(states[-1]).view(-1), net(graph))
        result = span_forward(net, states, interval_view(graph, 14, 18))
        torch.testing.assert_close(result['logits'], net(graph)[14:18])


def test_cut_all_is_full_models_own_mlp_not_separately_trained_node_model():
    graph, _ = graph_sample()
    net = model()
    with torch.no_grad():
        view = interval_view(graph, 14, 18)
        got = span_forward(net, factual_states(net, graph), view, np.ones(len(view['internal']), bool))
        own = net(graph, ablation='no_graph')
    torch.testing.assert_close(got['logits'], own[14:18])


def test_bias_is_muted_with_message_not_with_edge_attribute():
    graph, _ = graph_sample()
    net = model()
    with torch.no_grad():
        net.mp_layers[0].msg_mlp[-1].bias.fill_(3)
        state = factual_states(net, graph)[0]
        view = interval_view(graph, 14, 18)
        disabled = np.ones(len(view['internal']), bool)
        total, inside, outside = aggregate_groups(net.mp_layers[0], state, view, 3, disabled)
    assert not total.any() and not inside.any() and not outside.any()
    np.testing.assert_array_equal(view['divisor'], degree(graph)[14:18])


def test_onset_has_no_internal_incoming_path():
    graph, sample = graph_sample()
    net = model()
    with torch.no_grad():
        base = factual_states(net, graph)
        view = interval_view(graph, 14, 18)
        full = span_forward(net, base, view)
        cut = span_forward(net, base, view, view['internal'])
        no_reuse = span_forward(net, base, view, reference=cut)
    torch.testing.assert_close(full['logits'][0], cut['logits'][0])
    torch.testing.assert_close(full['logits'][0], no_reuse['logits'][0])


def test_no_reuse_one_round_equals_full():
    graph, _ = graph_sample()
    net = model(1)
    with torch.no_grad():
        base = factual_states(net, graph)
        view = interval_view(graph, 14, 18)
        full = span_forward(net, base, view)
        cut = span_forward(net, base, view, view['internal'])
        no_reuse = span_forward(net, base, view, reference=cut)
    torch.testing.assert_close(no_reuse['logits'], full['logits'])


def test_no_reuse_removes_two_hop_internal_inheritance_in_positive_toy():
    graph, _ = graph_sample()
    net = model(2)
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            parameter.fill_(.08 if name.endswith('weight') else 0)
        base = factual_states(net, graph)
        view = interval_view(graph, 14, 18)
        full = span_forward(net, base, view)
        cut = span_forward(net, base, view, view['internal'])
        no_reuse = span_forward(net, base, view, reference=cut)
    torch.testing.assert_close(no_reuse['logits'][:2], full['logits'][:2])
    assert torch.all(full['logits'][2:] > no_reuse['logits'][2:])


def test_no_internal_edges_means_no_internal_effect():
    graph, _ = graph_sample()
    keep = graph['edge_index'][0] < 2
    graph['edge_index'] = graph['edge_index'][:, keep]
    graph['edge_attr'] = graph['edge_attr'][keep]
    graph['edge_mark'] = graph['edge_mark'][keep]
    net = model()
    with torch.no_grad():
        base = factual_states(net, graph)
        view = interval_view(graph, 14, 18)
        full = span_forward(net, base, view)
        cut = span_forward(net, base, view, view['internal'])
        no_reuse = span_forward(net, base, view, reference=cut)
    torch.testing.assert_close(full['logits'], no_reuse['logits'])


def test_two_by_two_and_readout_are_exact_signed_decompositions():
    graph, sample = graph_sample()
    pair = dict(id='a', source_id='asource', error_start=12, normal_start=8, length=4)
    with torch.no_grad():
        net = model()
        saved, _ = pair_forward(net, graph, sample, factual_states(net, graph), pair, 2, 42)
    parts = components(saved['logits'], saved['names'])
    np.testing.assert_allclose(parts['own'] + parts['internal'] + parts['external'], saved['logits'][:, 0], atol=1e-6)
    np.testing.assert_allclose(saved['contributions'].sum(-1) + saved['readout_bias'], saved['logits'], atol=1e-6)


def test_swapping_identical_activation_is_noop():
    graph, _ = graph_sample()
    net = model()
    with torch.no_grad():
        full = span_forward(net, factual_states(net, graph), interval_view(graph, 14, 18))
        for branch in ('own', 'internal', 'external'):
            result = last_swap(net, full, full, branch)
            torch.testing.assert_close(result['logits'], full['logits'])
            torch.testing.assert_close(result['gate'], full['gate'])


def test_random_controls_do_not_pretend_pure_lag1_is_exchangeable():
    graph, sample = graph_sample()
    view = interval_view(graph, 14, 18)
    mask, eligible = random_internal_mask(view, sample['token_ids'], 0)
    np.testing.assert_array_equal(mask, view['internal'])
    assert eligible.sum() == 0


def test_random_mask_preserves_each_target_lag_copy_count():
    graph, sample = graph_sample()
    source = np.arange(2, 12)
    graph['edge_index'] = np.stack((source, np.full(len(source), 12)))
    graph['edge_attr'] = np.ones((len(source), 4), dtype=np.float32)
    graph['edge_mark'] = np.tile([[0, 1]], (len(source), 1))
    view = interval_view(graph, 6, 13)
    ids = np.arange(34)
    mask, eligible = random_internal_mask(view, ids, 2)
    distance = np.ceil(np.log2(12 - source)).astype(int)
    for group in np.unique(distance):
        selected = distance == group
        assert mask[selected].sum() == view['internal'][selected].sum()
    assert eligible.sum() > 0


def test_pair_inputs_not_modified_and_roles_separate():
    graph, sample = graph_sample()
    original = copy.deepcopy(graph)
    pair = dict(id='a', source_id='asource', error_start=12, normal_start=8, length=4)
    with torch.no_grad():
        net = model()
        saved, _ = pair_forward(net, graph, sample, factual_states(net, graph), pair, 1, 0)
    for key in graph:
        np.testing.assert_array_equal(graph[key], original[key])
    assert saved['logits'].shape[:2] == (2, 11)


def test_pair_validation_rejects_reused_controls():
    pair = dict(id='a', error_start=12, normal_start=8, length=4)
    with pytest.raises(ValueError, match='duplicate'):
        validate_pair_set([pair, pair])


def test_actual_cli_resume_report_and_no_original_overwrites(tmp_path):
    root, prepared = fixture_files(tmp_path)
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    out = tmp_path / 'audit'
    args = ['--mode', 'lockin', '--root', str(root), '--prepared', str(prepared),
            '--device', 'cpu', '--output', str(out), '--bootstrap', '0', '--lockin-random', '1']
    main(args)
    capture = out / 'captures/000000.npz'
    timestamp = capture.stat().st_mtime_ns
    main(args)
    assert timestamp == capture.stat().st_mtime_ns
    for path, value in original.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == value
    summary = pd.read_csv(out / 'summary.csv')
    assert set(summary.experiment) >= {'full', 'cut_internal', 'no_reuse', 'last_swap_external'}
    random = pd.read_csv(out / 'random_adjusted.csv')
    assert random.difference.isna().all()
    assert (out / 'figures/specific_effect.png').exists()
    assert (out / 'lockin_review.tar.gz').stat().st_size > 0
    code = "import runpy,sys;sys.argv=['audit']+sys.argv[1:];runpy.run_module('experiments.charm_structure_audit.main',run_name='__main__');assert 'torch' not in sys.modules"
    done = subprocess.run([sys.executable, '-c', code, *args, '--lockin-stage', 'report'], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_wrong_checkpoint_replay_stops_not_fixed_by_refitting(tmp_path):
    root, prepared = fixture_files(tmp_path)
    checkpoint = root / 'charm_in/checkpoint.pt'
    saved = torch.load(checkpoint, weights_only=True)
    saved['model_state']['pred.3.bias'] += 1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='replay failed'):
        main(['--mode', 'lockin', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu',
              '--output', str(tmp_path / 'failed'), '--bootstrap', '0'])


def test_no_match_not_fabricated(tmp_path):
    root, prepared = fixture_files(tmp_path)
    with pytest.raises(ValueError, match='No fixed pairs'):
        main(['--mode', 'lockin', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu',
              '--output', str(tmp_path / 'empty'), '--pair-tier', 'cluster_heads'])


def test_depth_summary_drops_missing_scalar_observations(tmp_path):
    from experiments.charm_structure_audit.routes import depth_contrasts, PHASES, METRICS
    observed = np.zeros((2, len(PHASES), len(METRICS), 3, 2))
    observed[0] = 1
    missing = np.full_like(observed, np.nan)
    rows = [dict(id='a', source_id='s1', error_start=2, values=observed),
            dict(id='b', source_id='s2', error_start=2, values=missing)]
    depth_contrasts(rows, tmp_path, 0)
    result = pd.read_csv(tmp_path / 'depth_summary.csv')
    assert result['mean'].eq(1).all()
    assert result.sources.eq(1).all()
