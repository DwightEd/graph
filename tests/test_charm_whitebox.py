"""Scientific checks for the actual frozen node MLP, including failure cases."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from test_charm_structure_audit import fixture_files, graph_sample, model
from experiments.charm_structure_audit.model import CHARM
from experiments.charm_structure_audit.data import read_json, load_predictions, save_scores, token_frame
from experiments.charm_structure_audit.whitebox_math import (
    linearize, evaluate_nodes, integrate_difference, hybrid_logits, matched_random_masks)
from experiments.charm_structure_audit.whitebox import node_input
from experiments.charm_structure_audit.main import main


@pytest.mark.parametrize('residual', [False, True])
def test_affine_weights_and_intercepts_equal_actual_node_function(residual):
    net = model().double()
    for layer in net.mp_layers:
        layer.residual = residual
    graph, _ = graph_sample()
    rng = np.random.default_rng(13)
    x = torch.tensor(rng.normal(size=(34, 4)), dtype=torch.float64, requires_grad=True)
    reference_graph = dict(graph, x=x.detach().numpy())
    net.double()
    # Parent forward forces float32 inputs; compare independent original layer algebra.
    state = net.in_proj(x).relu()
    for layer in net.mp_layers:
        state = layer.update(state, torch.zeros_like(state))
    logits = net.pred(state).flatten()
    gradient = torch.autograd.grad(logits.sum(), x)[0]
    decoded = linearize(net, x)
    torch.testing.assert_close(decoded['logits'], logits)
    torch.testing.assert_close(decoded['beta'], gradient, atol=1e-10, rtol=1e-10)
    rebuilt = (decoded['beta'] * x).sum(-1) + decoded['intercept']
    torch.testing.assert_close(rebuilt, logits, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(decoded['contributions'].sum(-1) + net.pred[3].bias[0], logits)


def test_no_message_network_is_ever_called():
    net = model()
    def forbidden(*args, **kwargs):
        raise AssertionError('Message execution in pointwise whitebox')
    for layer in net.mp_layers:
        layer.msg_mlp.forward = forbidden
    evaluate_nodes(net, torch.rand(5, 4))


def hinge_network(cancellation=False):
    net = CHARM(4, 4, dict(hidden_dim=4, gnn_layers=0, residual_mp=True)).double().eval()
    with torch.no_grad():
        net.in_proj.weight.copy_(torch.eye(4))
        net.in_proj.bias.fill_(-.317)
        net.pred[0].weight.zero_()
        net.pred[0].weight[0, 0] = 1.
        net.pred[0].weight[1, 1] = 1.
        net.pred[0].bias.zero_()
        net.pred[3].weight[:] = torch.tensor([[1., -1. if cancellation else 0.]])
        net.pred[3].bias.zero_()
    net.requires_grad_(False)
    return net


def test_gate_crossings_require_path_not_endpoint_gradient_times_difference():
    net = hinge_network()
    normal = torch.zeros(2, 4, dtype=torch.float64)
    error = torch.ones_like(normal)
    result = integrate_difference(net, normal, error, 512)
    torch.testing.assert_close(result['attribution'][:, 0], torch.full((2,), .683, dtype=torch.float64), atol=.004, rtol=0)
    assert (result['completeness_residual'].abs() < .004).all()
    assert not linearize(net, normal)['gates'].equal(linearize(net, error)['gates'])
    assert (linearize(net, error)['beta'][:, 0] == 1).all()
    assert (linearize(net, normal)['beta'][:, 0] == 0).all()


def test_completeness_cancellation_does_not_fake_allocation_convergence():
    net = hinge_network(cancellation=True)
    normal = torch.zeros(1, 4, dtype=torch.float64)
    error = torch.ones_like(normal)
    result = integrate_difference(net, normal, error, 32, atol=1e-12, rtol=1e-12)
    assert result['completeness_residual'].abs().max() < 1e-12
    assert result['allocation_change'].min() > 1e-4
    assert not result['converged'].any()


def test_identical_endpoints_converge_to_zero_attribution():
    net = model().double()
    values = torch.ones(3, 4, dtype=torch.float64)
    result = integrate_difference(net, values, values, 32)
    assert result['converged'].all()
    assert not result['attribution'].any()


def test_full_exchange_and_no_exchange_have_exact_limits():
    net = model().double()
    normal, error = torch.rand(3, 4).double(), torch.rand(3, 4).double()
    base_n, base_e = evaluate_nodes(net, normal)['logits'], evaluate_nodes(net, error)['logits']
    full = torch.ones_like(error, dtype=torch.bool)
    removed, added = hybrid_logits(net, error, normal, full)
    torch.testing.assert_close(removed, base_n)
    torch.testing.assert_close(added, base_e)
    removed, added = hybrid_logits(net, error, normal, ~full)
    torch.testing.assert_close(removed, base_e)
    torch.testing.assert_close(added, base_n)


def test_changing_trained_readout_reverses_contributions():
    net = model().double()
    values = torch.rand(3, 4).double()
    before = linearize(net, values)
    with torch.no_grad():
        net.pred[3].weight.mul_(-1)
        net.pred[3].bias.mul_(-1)
    after = linearize(net, values)
    torch.testing.assert_close(after['beta'], -before['beta'])
    torch.testing.assert_close(after['contributions'], -before['contributions'])


def test_random_channels_preserve_layer_and_delta_magnitude_strata():
    rng = np.random.default_rng(3)
    values = torch.tensor(rng.normal(size=(3, 32)))
    difference = torch.tensor(rng.normal(size=(3, 32)))
    chosen, controls, exchangeable = matched_random_masks(values, difference, 8, 6, 3, 42)
    assert (chosen.sum(-1) == 6).all()
    assert (controls.sum(-1) == 6).all()
    for row in range(3):
        for start in range(0, 32, 8):
            magnitude = abs(difference[row, start:start+8].numpy())
            band = np.searchsorted(np.quantile(magnitude, [.25, .5, .75]), magnitude, side='right')
            for value in np.unique(band):
                columns = np.arange(start, start+8)[band == value]
                assert (controls[:, row, columns].sum(-1) == chosen[row, columns].sum()).all()
    assert (exchangeable >= 0).all()


def test_nonexchangeable_random_sets_are_identical_and_disclosed():
    values = torch.ones(2, 4)
    chosen, controls, exchangeable = matched_random_masks(values, values, 2, 4, 3, 0)
    assert chosen.all() and controls.all()
    assert not exchangeable.any()


def whitebox_fixture(tmp_path):
    root, prepared = fixture_files(tmp_path)
    shutil.copytree(root / 'charm_in', root / 'node_only')
    graph_path = prepared / 'graphs/test/test.npz'
    with np.load(graph_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays['x'] = np.random.default_rng(2).uniform(0, 1, arrays['x'].shape).astype(np.float32)
    save_scores(graph_path, **arrays)
    net = model()
    with torch.no_grad():
        score = net(arrays, 'no_graph')[2:].sigmoid().numpy()
    path = root / 'node_only/test/samples/test.npz'
    with np.load(path) as saved:
        result = {key: saved[key] for key in saved.files}
    result['score'] = score
    save_scores(path, **result)
    sample = dict(json.loads(str(result['record_json'])), **result)
    token_frame(sample, score, .5).to_csv(root / 'node_only/test/tokens.csv', index=False)
    return root, prepared


def test_loader_does_not_read_edge_members(tmp_path, monkeypatch):
    root, prepared = whitebox_fixture(tmp_path)
    record = load_predictions(root / 'node_only/test')[0]
    original = np.load
    accessed = []
    class Guard:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): self.value.close()
        def __getitem__(self, key):
            accessed.append(key)
            assert key not in ('edge_index', 'edge_attr', 'edge_mark')
            return self.value[key]
    monkeypatch.setattr(np, 'load', lambda *a, **kw: Guard(original(*a, **kw)))
    values, geometry = node_input(record, prepared)
    assert geometry == (2, 2) and values.shape == (32, 4)
    assert 'x' in accessed


def test_source_identity_rejected_at_boundary(tmp_path):
    root, prepared = whitebox_fixture(tmp_path)
    record = load_predictions(root / 'node_only/test')[0]
    record['source_id'] = 'other'
    with pytest.raises(ValueError, match='source'):
        node_input(record, prepared)


def test_real_whitebox_cli_resume_report_and_originals_unchanged(tmp_path):
    root, prepared = whitebox_fixture(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    args = ['--mode', 'whitebox', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu',
            '--wb-max-points', '32', '--wb-budget', '2', '--bootstrap', '0']
    main(args)
    output = root / 'audit_whitebox'
    assert read_json(output / 'status.json')['completed_pairs'] == 1
    assert (output / 'figures/layer_contrast.png').exists()
    assert (output / 'whitebox_review.tar.gz').exists()
    table = pd.read_csv(output / 'tokens.csv.gz')
    np.testing.assert_allclose(table.logit_gap, table.readout_gap_sum, atol=1e-8)
    assert set(pd.read_csv(output / 'intervention_summary.csv').family) >= {'layer'}
    capture = next((output / 'captures').glob('*.npz'))
    stamp = capture.stat().st_mtime_ns
    main(args)
    assert capture.stat().st_mtime_ns == stamp
    code = "import runpy,sys;sys.argv=['main']+sys.argv[1:];runpy.run_module('experiments.charm_structure_audit.main',run_name='__main__');assert 'torch' not in sys.modules"
    done = subprocess.run([sys.executable, '-c', code, *args, '--wb-stage', 'report'], capture_output=True, text=True, timeout=40)
    assert done.returncode == 0, done.stderr
    for path, content in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == content


def test_replay_failure_stops_before_attribution(tmp_path):
    root, prepared = whitebox_fixture(tmp_path)
    path = root / 'node_only/test/samples/test.npz'
    with np.load(path) as saved:
        values = {key: saved[key] for key in saved.files}
    values['score'][:] = 0
    save_scores(path, **values)
    with pytest.raises(ValueError, match='replay failed'):
        main(['--mode', 'whitebox', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu'])
    assert not (root / 'audit_whitebox/captures').exists()


def test_source_mean_not_token_count_weighting():
    from experiments.charm_structure_audit.whitebox_report import source_contrast
    table = pd.DataFrame(dict(source_id=['A']*100+['B'], id=['1']*100+['2'], pair_start=[1]*101))
    values = np.r_[np.ones(100), 0.][:, None]
    assert source_contrast(values, table).mean() == .5


def test_joint_head_pattern_can_separate_identical_per_head_marginals():
    net = hinge_network()
    with torch.no_grad():
        net.in_proj.weight.zero_()
        net.in_proj.weight[0, :2] = torch.tensor([1., -1.])
        net.in_proj.weight[1, :2] = torch.tensor([-1., 1.])
        net.in_proj.bias.zero_()
        net.pred[0].weight[0, :2] = 1.
    normal = torch.tensor([[.1,.1,0,0], [.9,.9,0,0]], dtype=torch.float64)
    error = torch.tensor([[.1,.9,0,0], [.9,.1,0,0]], dtype=torch.float64)
    torch.testing.assert_close(error.mean(0), normal.mean(0))
    torch.testing.assert_close(evaluate_nodes(net, error)['logits'], torch.tensor([.8,.8], dtype=torch.float64))
    assert not evaluate_nodes(net, normal)['logits'].any()
    result = integrate_difference(net, normal, error, 32)
    assert result['converged'].all()
    torch.testing.assert_close(result['attribution'].sum(-1), torch.tensor([.8,.8], dtype=torch.float64))


def test_positive_relu_rescaling_keeps_function_and_coefficients():
    net = hinge_network()
    values = torch.rand(5, 4).double()
    before = linearize(net, values)
    with torch.no_grad():
        net.in_proj.weight.mul_(7)
        net.in_proj.bias.mul_(7)
        net.pred[0].weight.div_(7)
    after = linearize(net, values)
    torch.testing.assert_close(after['logits'], before['logits'])
    torch.testing.assert_close(after['beta'], before['beta'])


def test_original_public_forward_is_same_node_only_function():
    graph, _ = graph_sample()
    graph['x'] = np.random.default_rng(14).normal(size=graph['x'].shape).astype(np.float32)
    net = model()
    expected = net(graph, 'no_graph')
    actual = evaluate_nodes(net, torch.as_tensor(graph['x']))['logits']
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
