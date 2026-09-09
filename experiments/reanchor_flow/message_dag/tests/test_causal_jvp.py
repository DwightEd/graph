"""Causal-support and cut-closure regressions without a pretrained checkpoint."""
import copy

import numpy as np

import pytest
import torch
import torch.nn.functional as F

from experiments.reanchor_flow.message_dag.differential import DifferentialLayer, rotate
from experiments.reanchor_flow.message_dag.transport import last_crossing_edges
from experiments.reanchor_flow.message_dag.cut_artifact import CutRecorder


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def operator(seed=1, chunk=3):
    """Nontrivial GQA/RoPE operands; use the production JVP/VJP methods."""
    torch.manual_seed(seed)
    op = object.__new__(DifferentialLayer)
    op.d, op.h, op.kv, op.hd = 128, 8, 2, 16
    op.device, op.eps, op.chunk, op.layer = 'cpu', 1e-5, chunk, 0
    op.rows = torch.arange(4, 13)
    r = len(op.rows)
    op.x = torch.randn(r, op.d)
    op.w = {
        'input_norm': torch.ones(op.d),
        'value': torch.randn(op.kv*op.hd, op.d)/op.d**.5,
        'output': torch.randn(op.d, op.d)/op.d**.5,
    }
    op.wq = torch.randn(op.d, op.d)*3/op.d**.5
    op.wk = torch.randn(op.kv*op.hd, op.d)*3/op.d**.5
    angle = torch.randn(r, op.hd//2).repeat(1, 2)
    op.cos, op.sin = angle.cos(), angle.sin()
    op.scale = op.hd**-.5
    op.q = torch.randn(op.h, r, op.hd)*3
    op.k = torch.randn(op.kv, 13, op.hd).repeat_interleave(op.h//op.kv, 0)*3
    op.v = torch.randn(op.kv, 13, op.hd).repeat_interleave(op.h//op.kv, 0)
    score = torch.einsum('hqd,hsd->hqs', op.q, op.k)*op.scale
    score.masked_fill_(torch.arange(13)[None,None,:] > op.rows[None,:,None], -torch.inf)
    op.cached_attention = score.softmax(-1)
    # Nonlinear same-position suffixes must also participate in cut closure.
    op.post = torch.randn_like(op.x)
    op.w.update(post_norm=torch.ones(op.d),
                gate=torch.randn(2*op.d, op.d)/op.d**.5,
                up=torch.randn(2*op.d, op.d)/op.d**.5,
                down=torch.randn(op.d, 2*op.d)/(2*op.d)**.5)
    z = op.post * torch.rsqrt(op.post.square().mean(-1, keepdim=True)+op.eps)
    op.gate, op.up = F.linear(z, op.w['gate']), F.linear(z, op.w['up'])
    sigmoid = op.gate.sigmoid()
    op.silu_prime = sigmoid*(1+op.gate*(1-sigmoid))
    op.activation = F.silu(op.gate)
    return op


@pytest.mark.parametrize('routing', [False, True])
@pytest.mark.parametrize('chunk', [1, 3, 8])
def test_same_position_seed_cannot_become_a_position_hop(routing, chunk):
    op = operator(chunk=chunk)
    delta = torch.zeros(2, len(op.rows), op.d)
    for event, row in enumerate((2, 5)):
        delta[event, row] = torch.randn(op.d)*10
    same, cross, codes = op.attention_jvp(delta, routing=routing)
    for event, row in enumerate((2, 5)):
        # This is a support invariant, not an approximate numerical comparison.
        assert torch.count_nonzero(cross[event, :row+1]) == 0
        assert torch.count_nonzero(same[event, :row]) == 0
        assert torch.count_nonzero(cross[event, row+1:]) > 0
    total = F.linear(codes.transpose(1,2).flatten(-2), op.w['output'])
    torch.testing.assert_close(same+cross, total, atol=5e-5, rtol=3e-5)


def native_attention(op, perturbation, routing):
    """Independent differentiable forward oracle around the fixed operands."""
    def norm(x):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+op.eps)*op.w['input_norm']
    dz = norm(op.x+perturbation)-norm(op.x)
    dv = F.linear(dz, op.w['value']).reshape(-1,op.kv,op.hd).transpose(0,1)
    v = op.v.clone()
    v[:,op.rows] = v[:,op.rows]+dv.repeat_interleave(op.h//op.kv,0)
    a = op.cached_attention
    if routing:
        dq = F.linear(dz,op.wq).reshape(-1,op.h,op.hd).transpose(0,1)
        dk = F.linear(dz,op.wk).reshape(-1,op.kv,op.hd).transpose(0,1)
        q = op.q+rotate(dq,op.cos,op.sin)
        k = op.k.clone()
        k[:,op.rows] = k[:,op.rows]+rotate(dk,op.cos,op.sin).repeat_interleave(op.h//op.kv,0)
        score = torch.einsum('hqd,hsd->hqs',q,k)*op.scale
        score = score.masked_fill(torch.arange(k.shape[1])[None,None,:]>op.rows[None,:,None],-torch.inf)
        a = score.softmax(-1)
    value = torch.einsum('hqs,hsd->hqd',a,v)
    return F.linear(value.transpose(0,1).flatten(-2),op.w['output'])


@pytest.mark.parametrize('routing', [False, True])
def test_split_matches_independent_float64_autograd(routing):
    op = operator()
    delta = torch.randn_like(op.x)
    same,cross,_ = op.attention_jvp(delta[None],routing=routing)
    oracle = copy.copy(op)
    for name,value in vars(op).items():
        if torch.is_tensor(value) and value.is_floating_point():
            setattr(oracle,name,value.double())
    oracle.w = {name:value.double() for name,value in op.w.items()}
    score = torch.einsum('hqd,hsd->hqs',oracle.q,oracle.k)*oracle.scale
    score.masked_fill_(torch.arange(13)[None,None,:]>oracle.rows[None,:,None],-torch.inf)
    oracle.cached_attention = score.softmax(-1)
    _,expected = torch.autograd.functional.jvp(
        lambda change: native_attention(oracle,change,routing),
        torch.zeros_like(oracle.x), delta.double())
    torch.testing.assert_close((same+cross)[0].double(), expected, atol=8e-5, rtol=5e-5)


def test_same_position_adjoint_is_unchanged():
    op = operator()
    delta,reader = torch.randn_like(op.x),torch.randn_like(op.x)
    same,_,_ = op.attention_jvp(delta[None])
    torch.testing.assert_close((same[0]*reader).sum(),
                               (delta*op.attention_same_vjp(reader)).sum(),
                               atol=2e-4,rtol=3e-5)


@pytest.mark.parametrize('seed', [1,2,3,4,5])
def test_cross_branch_matches_physical_edges_without_relaxing_tolerance(seed):
    op = operator(seed)
    delta = torch.zeros(1,len(op.rows),op.d)
    delta[0,2] = torch.randn(op.d)*10
    reader = torch.randn_like(op.x)
    _,cross,_ = op.attention_jvp(delta)
    expected = (cross[0]*reader).sum(-1)[:-1]
    net,gross = torch.zeros_like(expected),torch.zeros_like(expected)
    for begin,end,_,edges in last_crossing_edges(op,delta,reader):
        net[begin:end] = edges.sum((0,1,3,4))
        gross[begin:end] = edges.abs().sum((0,1,3,4))
    allowance = 2e-7+2e-4*torch.maximum(expected.abs(),gross)
    assert torch.all((net-expected).abs() <= allowance)


@pytest.mark.parametrize('chunk', [1,3,8])
def test_multilayer_cut_closes_with_mlp_and_repeated_seed(chunk):
    ops = [operator(seed=seed,chunk=chunk) for seed in (1,2,3,4)]
    row,r,d = 2,len(ops[0].rows),ops[0].d
    direction = torch.randn(r,d)
    direction[-1] = 0
    reader = direction.clone()
    suffix = {}
    for layer in reversed(range(len(ops))):
        op = ops[layer]
        post_reader = reader+op.mlp_vjp(reader)
        suffix[layer] = post_reader
        reader = post_reader+op.attention_same_vjp(post_reader)
    state = torch.zeros(3,r,d)
    positive,negative = torch.zeros(2,r-1),torch.zeros(2,r-1)
    for layer,op in enumerate(ops):
        for begin,end,_,edges in last_crossing_edges(op,state.sum(0)[None],suffix[layer]):
            # Same partition as CutRecorder: last source is root versus a relay.
            values = edges[0]
            for accumulator,part in ((positive,values.clamp_min(0)),
                                     (negative,(-values).clamp_min(0))):
                accumulator[0,begin:end] += part[:,:,row].sum((0,2))
                accumulator[1,begin:end] += part[:,:,row+1:].sum((0,2,3))
        same,cross,_ = op.attention_jvp(state)
        post = state+same
        post[1] += cross[0]
        post[2] += cross[1]+cross[2]
        if layer in (0,1):
            post[0,row] += torch.randn(d)*10
        state = post+op.mlp_jvp(post)
        assert torch.count_nonzero(state[1:,:row+1]) == 0
    expected = (state[1:]*direction).sum(-1)[:,:-1]
    error = positive-negative-expected
    scale = torch.maximum(expected.abs(),positive+negative)
    assert torch.all(error.abs() <= 2e-7+2e-4*scale)


@pytest.mark.parametrize('nonfinite', [False, True])
def test_closure_failure_stays_fatal_and_reports_coordinates(nonfinite):
    recorder = object.__new__(CutRecorder)
    recorder.row, recorder.rows = 2, np.arange(4, 13)
    recorder.hop_positive = np.zeros((2,8,2), dtype=np.float32)
    recorder.hop_negative = np.zeros_like(recorder.hop_positive)
    event = dict(margin_response=np.zeros((3,3,8), dtype=np.float32))
    event['margin_response'][0,2,5] = np.nan if nonfinite else 1
    match = ('nonfinite' if nonfinite else
             r'event_position=6, hop=2\+, target_position=10, .*allowance=')
    # No archive is attached: failure must precede any publication attempt.
    with pytest.raises(ValueError, match=match):
        recorder.finish(event)
