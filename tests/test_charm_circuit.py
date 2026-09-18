"""Core algebra and faithful execution; synthetic tests never claim natural performance."""

import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.charm_structure_audit.model import CHARM
from experiments.charm_structure_audit.circuit_math import (
    tail, project, dominant_direction, axis_experiments, paired_accounting,
    relu_secant, local_gradients)
from experiments.charm_structure_audit.circuit import token_weights, run_circuit


def network(residual=True):
    torch.manual_seed(5)
    return CHARM(4, 4, dict(hidden_dim=8, gnn_layers=2, residual_mp=residual)).double().eval()


@pytest.mark.parametrize('residual', [True, False])
def test_finite_difference_conservation(residual):
    model = network(residual)
    error, normal = torch.randn(10, 4).double(), torch.randn(10, 4).double()
    decoded = paired_accounting(model, error, normal)
    gap = tail(model, project(model, error)) - tail(model, project(model, normal))
    torch.testing.assert_close(decoded['head_contribution'].sum(1), gap, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(decoded['unit_contribution'].sum(1), gap, atol=1e-12, rtol=1e-10)


def test_original_node_formula():
    model = network()
    values = torch.rand(6, 4).double()
    state = model.in_proj(values).relu()
    for layer in model.mp_layers:
        joined = torch.cat((state, torch.zeros_like(state)), dim=-1)
        update = layer.up_mlp(joined)
        state = (state + update).relu()
    torch.testing.assert_close(tail(model, project(model, values)), model.pred(state).flatten())


def test_zero_delta_has_zero_contribution():
    model = network()
    values = torch.rand(6, 4).double()
    assert not paired_accounting(model, values, values)['head_contribution'].any()


def test_secant_across_zero():
    left = torch.tensor([-1., 2., 0., 1.])
    right = torch.tensor([2., -1., 0., 1.])
    scale = relu_secant(left, right)
    torch.testing.assert_close(scale*(left-right), left.relu()-right.relu())


def test_direction_is_weight_only_and_reconstructs_rank_one():
    weight = torch.arange(1, 9).double()[:, None] @ torch.tensor([[.1, -.5, .2, .3]]).double()
    direction, loading, singular = dominant_direction(weight)
    torch.testing.assert_close(loading[:, None]*direction[None], weight)
    assert direction[direction.abs().argmax()] > 0
    assert singular[1] < 1e-10


def test_exact_rank_one_network_is_preserved():
    model = network()
    with torch.no_grad():
        model.in_proj.weight.copy_(torch.randn(8, 1).double() @ torch.randn(1, 4).double())
        scores, *_ = axis_experiments(model, torch.rand(4, 4).double(), torch.rand(4, 4).double(), repeats=2)
    torch.testing.assert_close(scores['full'], scores['keep_axis'], atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(scores['remove_axis'][0], scores['remove_axis'][1], atol=1e-10, rtol=1e-8)


def test_orthogonal_control_exact_pre_activation_norm():
    model = network()
    direction, loading, _ = dominant_direction(model.in_proj.weight)
    difference = torch.randn(7, 4).double()
    delta = (difference@direction)[:, None]*loading
    axis = loading/loading.norm()
    vector = torch.randn(8).double()
    vector -= (vector@axis)*axis
    vector /= vector.norm()
    control = (delta@axis)[:, None]*vector
    torch.testing.assert_close(delta.norm(dim=1), control.norm(dim=1))
    torch.testing.assert_close(control@axis, torch.zeros(7).double(), atol=1e-12, rtol=0)


def test_no_message_execution():
    model = network()
    def forbidden(*args, **kwargs):
        raise AssertionError('Message MLP was called')
    for layer in model.mp_layers:
        layer.msg_mlp.forward = forbidden
    values = torch.randn(5, 4).double()
    tail(model, project(model, values))
    paired_accounting(model, values, values/2)
    local_gradients(model, values)


def test_equal_source_weights_not_equal_token_weights():
    table = pd.DataFrame(dict(source_id=['a']*5+['b'], id=['x']*5+['y'], start=[1]*6))
    weights = token_weights(table)
    assert weights[:5].sum() == pytest.approx(.5)
    assert weights[-1] == .5


def test_real_export_format_and_cli(tmp_path):
    model = network()
    error, normal = torch.rand(4, 4).double(), torch.rand(4, 4).double()
    logits = torch.stack((tail(model, project(model, error)), tail(model, project(model, normal))))
    content = io.BytesIO()
    np.savez_compressed(content, error_x=error.numpy(), normal_x=normal.numpy(),
        baseline_score=logits.detach().sigmoid().numpy(), threshold=.5,
        text=np.array([['a','b','c','d'],['e','f','g','h']]))
    weights = io.BytesIO()
    torch.save(dict(model_state=model.state_dict(), hp=dict(hidden_dim=8,gnn_layers=2,residual_mp=True)),weights)
    pair=dict(id='a',source_id='s',error_start=4,normal_start=0,length=4,file='000000.npz')
    members={'manifest.json':json.dumps(dict(pairs=[pair])).encode(),
             'geometry.json':json.dumps(dict(layers=2,heads=2)).encode(),
             'node_only/checkpoint.pt':weights.getvalue(),'inputs/000000.npz':content.getvalue()}
    path=tmp_path/'input.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        for name,data in members.items():
            info=tarfile.TarInfo(name)
            info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    before=path.read_bytes()
    from experiments.charm_structure_audit.main import main
    main(['--mode','circuit','--circuit-input',str(path),'--output',str(tmp_path/'out'),'--bootstrap','10'])
    assert path.read_bytes()==before
    assert (tmp_path/'out/head_rules.csv').exists()
    assert (tmp_path/'out/axis_controls.png').exists()
    status=json.loads((tmp_path/'out/status.json').read_text())
    assert status['paired_positions']==4 and status['pairs']==1
