import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from experiments.internal_flow.audit_capture import local_logp_gradient,group_masks,GROUPS,capture_audit


def test_local_projection_matches_true_local_gradient():
    torch.manual_seed(42)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=37,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    state=torch.randn(3,16,requires_grad=True)
    targets=torch.tensor([2,5,8])
    logp=model.lm_head(model.model.norm(state)).log_softmax(-1)[torch.arange(3),targets].sum()
    expected=torch.autograd.grad(logp,state)[0]
    actual=local_logp_gradient(model,state.detach(),targets)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_groups_are_causal_and_reconstruction_is_native():
    item=dict(prompt_ids=[1,2,3],answer_ids=[4,5,6,7],spans=[[1,3]],
        groups=dict(source=[1],evidence=[1],constraint=[1],other_source=[],instruction=[0,2]))
    masks=group_masks(item,[0,1,3],6,'cpu').numpy()
    assert not masks[0,GROUPS.index('local_history')].any()
    assert not masks[1,GROUPS.index('claim_prefix')].any()
    assert np.flatnonzero(masks[2,GROUPS.index('claim_prefix')]).tolist()==[4,5]
    config=LlamaConfig(vocab_size=37,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval()
    result=capture_audit(model,item,[0,1,3])
    assert result['stats'].shape==(2,2,3,8,4)
    assert result['reconstruction'].max()<1e-5
    # Disjoint source + instruction + local + remote must account for all attention.
    mass=result['stats'][...,0]
    np.testing.assert_allclose(mass[...,0]+mass[...,4]+mass[...,5]+mass[...,6],1,atol=1e-6)


def test_reanchor_rule_matches_original_channel_implementation():
    from types import SimpleNamespace
    from experiments.internal_flow.audit_reanchors import switch_measures
    from experiments.unsupervised_token_graph.span_audit.onset_changes import measure_head
    torch.manual_seed(42)
    logits=torch.randn(1,2,12,12)
    logits.masked_fill_(torch.ones(12,12,dtype=torch.bool).triu(1),-float('inf'))
    attention=logits.softmax(-1)
    item=dict(prompt_ids=[0,1,2,3,4],answer_ids=[5,6,7,8,9,10,11,12],special_ids=[0,3])
    actual=switch_measures(attention,item)
    token_ids=np.arange(13)
    special=np.isin(token_ids,item['special_ids'])
    for head in range(2):
        raw=attention[0,head].numpy()
        channel=SimpleNamespace(queries=np.arange(12),row=lambda i:(np.arange(i+1),raw[i,:i+1]))
        expected,_=measure_head(channel,token_ids,5,special,3,10)
        np.testing.assert_allclose(actual[head],expected[:8],atol=3e-7,equal_nan=True)


def test_candidate_sequence_scores_align_with_prediction_positions():
    from experiments.internal_flow.audit_candidates import score
    from experiments.path_conflict.native import Intervention
    config=LlamaConfig(vocab_size=37,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval()
    case=dict(prefix_ids=[1,2,3],candidates=dict(correct=[4,5],wrong=[6,7,8]),groups=dict(source=[1]))
    with torch.no_grad():
        lp=model(torch.tensor([[1,2,3,4]])).logits[0].log_softmax(-1)
    actual=score(model,case,side='correct')
    assert abs(actual['sum']-float(lp[2,4]+lp[3,5]))<1e-6
    base=score(model,case)
    sham=score(model,case,Intervention(layer=0,groups=('source',),sources=(1,),dose=0.))
    assert base==sham
