import torch
from .measure import completion_direction,response


def test_completion_preserves_mass_and_convex_weights():
    attention = torch.tensor([[[.1,.2,.3]]])
    relation = torch.tensor([[[0.,1.,0.],[.5,0.,.5],[1.,0.,0.]]])
    direction = completion_direction(attention,relation)
    torch.testing.assert_close(direction.sum(-1),torch.zeros(1,1),atol=1e-7,rtol=0)
    assert torch.all(attention+.25*direction>=0)


def test_gate_conversion_matches_native_autograd_direction():
    attention = torch.tensor([[[.05,.15,.8]]],dtype=torch.float64)
    values = torch.tensor([[1.,2.],[-2.,3.],[4.,-.5]],dtype=torch.float64)
    gate = torch.ones_like(attention,requires_grad=True)
    output = (attention*gate)@values
    metric = output.sin().sum()
    gradient = torch.autograd.grad(metric,gate)[0]
    direction = torch.tensor([[[.1,-.05,-.05]]],dtype=torch.float64)
    slope = response(attention,gradient,direction)
    epsilon = 1e-5
    changed = (((attention+epsilon*direction)@values).sin().sum()-metric.detach())/epsilon
    torch.testing.assert_close(slope.squeeze(),changed,atol=1e-5,rtol=1e-4)


def test_tiny_attention_does_not_erase_newly_read_message():
    attention = torch.tensor([[[1e-20,1.]]],dtype=torch.float64)
    sensitivity = torch.tensor([[[-2.,1.]]],dtype=torch.float64)
    direction = torch.tensor([[[.5,-.5]]],dtype=torch.float64)
    actual = response(attention,attention*sensitivity,direction)
    torch.testing.assert_close(actual,torch.tensor([[-1.5]],dtype=torch.float64))


def test_native_source_reallocation_matches_gate_derivative():
    from transformers import LlamaConfig,LlamaForCausalLM
    from experiments.decision_risk_flow.native import prefill,replay
    from experiments.decision_risk_flow.kernel import contract_edges
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
                         num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    prompt,answer = [1,3,5,7,9],[11,13,15]
    cache,_,_ = prefill(model,prompt,answer,checkpoints=(2,))
    positions = torch.tensor([5])
    tokens = prompt+answer[:-1]
    final,_,capture = replay(model,cache,tokens,positions,checkpoints=(2,))
    logits = model.lm_head(final)
    margin = logits[0,13]-logits[0,17]
    gradient = torch.autograd.grad(margin,capture.writes)[0]
    edges = contract_edges(gradient,capture.messages[0],model.model.layers[0].self_attn.o_proj.weight)
    attention = capture.messages[0][0][0,0,:5]
    delta = torch.zeros(1,len(tokens))
    delta[0,:5] = attention.mean()-attention
    expected = (edges[0,0,:5]/attention*delta[0,:5]).sum()
    gate = dict(layer=0,head=0,attention_delta=delta,dose=0.)
    unchanged,_,_ = replay(model,cache,tokens,positions,checkpoints=(2,),gate=gate)
    torch.testing.assert_close(unchanged,final)
    gate['dose'] = .01
    changed,_,_ = replay(model,cache,tokens,positions,checkpoints=(2,),gate=gate)
    scores = model.lm_head(changed)
    actual = (scores[0,13]-scores[0,17]-margin.detach())/.01
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=.03)
