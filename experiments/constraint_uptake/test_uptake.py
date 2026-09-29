import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.decision_risk_flow.native import prefill,replay
from .capture import lens_gradient
from .score import key_effects,head_features


def test_local_lens_gradient_matches_autograd():
    torch.manual_seed(4)
    config = LlamaConfig(vocab_size=23,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2)
    model = LlamaForCausalLM(config).double().eval().requires_grad_(False)
    state = torch.randn(3,16,dtype=torch.double,requires_grad=True)
    contrast = torch.randn(3,3,16,dtype=torch.double)
    # Llama RMSNorm explicitly computes variance in FP32.
    state = state.float().detach().requires_grad_(True)
    contrast = contrast.float()
    expected = []
    for candidate in range(3):
        score = (model.model.norm(state)*contrast[:,candidate]).sum()
        expected.append(torch.autograd.grad(score,state)[0])
    torch.testing.assert_close(lens_gradient(state,contrast,model.model.norm),torch.stack(expected,1),atol=1e-6,rtol=1e-5)


def test_native_factor_matches_finite_message_perturbation():
    torch.manual_seed(8)
    config = LlamaConfig(vocab_size=23,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).double().eval().requires_grad_(False)
    prompt,answer = [1,2,3],[4,5]
    tokens = prompt+answer[:-1]
    cache,_,_ = prefill(model,prompt,answer,checkpoints=(2,))
    position = torch.tensor([3])
    final,_,captured = replay(model,cache,tokens,position,checkpoints=(2,))
    contrast = model.lm_head.weight[5]-model.lm_head.weight[7]
    score = (final*contrast).sum()
    gradient = torch.autograd.grad(score,captured.writes[0])[0][0]
    factor = (gradient@model.model.layers[0].self_attn.o_proj.weight).reshape(1,4,4).permute(1,0,2)[:,:,None]
    attention,_,own = captured.messages[0]
    effect = key_effects(factor.numpy(),cache.layers[0].values[0].numpy(),own.numpy(),attention.numpy())
    dose = 1e-3
    delta = torch.zeros(1,len(tokens),dtype=torch.double)
    delta[0,1] = attention[0,0,1]
    measured = []
    for sign in (-1,1):
        changed,_,_ = replay(model,cache,tokens,position,checkpoints=(2,),
            gate=dict(layer=0,head=0,attention_delta=delta,dose=sign*dose))
        measured.append(float((changed*contrast).sum().detach()))
    np.testing.assert_allclose((measured[1]-measured[0])/(2*dose),effect[0,0,0,1],rtol=2e-3,atol=2e-6)
    from .sink_center import exchange_effects
    exchanged,_ = exchange_effects(factor.numpy(),cache.layers[0].values[0].numpy(),own.numpy(),
        attention.numpy(),np.zeros(4,dtype=int),3,np.array([1]))
    delta *= attention[0,0,0]
    delta[0,0] = -delta.sum()
    assert abs(float(delta.sum()))<1e-12
    # RMSNorm uses FP32 even in this double model; a gate derivative avoids
    # subtractive rounding at tiny finite doses. Actual 8B doses are checked separately.
    scale = torch.tensor(0.,dtype=torch.double,requires_grad=True)
    changed,_,_ = replay(model,cache,tokens,position,checkpoints=(2,),
        gate=dict(layer=0,head=0,attention_delta=delta,dose=scale))
    actual = torch.autograd.grad((changed*contrast).sum(),scale)[0]
    np.testing.assert_allclose(actual.detach().numpy(),exchanged[0,0,0,1],rtol=1e-5,atol=1e-7)


def test_self_key_source_assignment_and_no_negative_conflict():
    native = np.zeros((1,2,1,5))
    native[0,0,0,-1] = -2
    native[0,1,0,0] = -1
    native[0,1,0,-1] = 1
    attention = np.zeros((1,2,5))
    attention[:,:,-1] = 1
    feature,_ = head_features(native,native,attention,3,np.array([0,1]))
    np.testing.assert_allclose(feature[0,:,0,0],[0,1])
    np.testing.assert_allclose(feature[0,:,0,4],[1,0])


def test_two_source_blocks_have_no_fabricated_control():
    from .finite import choose_blocks
    selected = choose_blocks(np.array([2.,-1.]),[[0,5],[5,10]])
    assert selected==dict(positive=[0],negative=[1],joint=[0,1])


def test_vanishing_head_cannot_dominate_effect_weighted_readout():
    from .reweight import weighted_heads
    values = np.array([.2,1.]).reshape(1,2,1,1)
    energy = np.array([1.,1e-10]).reshape(1,2,1,1)
    np.testing.assert_allclose(weighted_heads(values,energy),[[.2]],atol=1e-9)


def test_batched_factor_product_equals_explicit_tensor_contraction():
    rng = np.random.default_rng(18)
    gradient = rng.normal(size=(4,5,3,8))
    values = rng.normal(size=(2,7,8))
    own = rng.normal(size=(4,5,8))
    attention = rng.uniform(size=(4,5,8))
    expected = np.einsum('hqcd,hkd->hqck',gradient,np.repeat(values,2,axis=0))*attention[:,:,None,:-1]
    actual = key_effects(gradient,values,own,attention)
    np.testing.assert_allclose(actual[...,:-1],expected,atol=1e-12)


def test_exchange_ignores_common_value_payload():
    from .sink_center import exchange_effects
    rng = np.random.default_rng(18)
    gradient = rng.normal(size=(4,5,3,8))
    values = rng.normal(size=(2,7,8))
    own = rng.normal(size=(4,5,8))
    attention = rng.uniform(size=(4,5,8))
    attention /= attention.sum(-1,keepdims=True)
    shift = rng.normal(size=(2,1,8))
    args = (attention,np.zeros(4,dtype=int),3,np.arange(5))
    before,_ = exchange_effects(gradient,values,own,*args)
    after,_ = exchange_effects(gradient,values+shift,own+np.repeat(shift,2,axis=0),*args)
    np.testing.assert_allclose(before,after,atol=1e-12)
