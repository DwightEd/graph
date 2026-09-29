"""Save exact native head factors for signed multi-candidate message responses."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.native import prefill,replay
from experiments.decision_risk_flow.run import load_model


def lens_gradient(state,contrast,norm):
    """Derivative of final-norm logit contrast at the immediate post-attention state."""
    variance = state.square().mean(-1,keepdim=True)+norm.variance_epsilon
    direction = contrast*norm.weight.float()
    inner = (direction*state[:,None]).mean(-1,keepdim=True)
    return (direction-state[:,None]*inner/variance[:,None])*variance.rsqrt()[:,None]


@torch.no_grad()
def choose_candidates(model,normalized,answer,positions):
    logits = model.lm_head(normalized[positions]).float()
    targets = torch.tensor(answer,device=model.device)[positions]
    logp = logits.log_softmax(-1)
    nll = -logp.gather(1,targets[:,None])[:,0]
    entropy = -(logp.exp()*logp).sum(-1)
    logits.scatter_(1,targets[:,None],-torch.inf)
    candidates = logits.topk(3,dim=-1).indices
    return targets,candidates,nll,entropy


def allocate(directory,layers,heads,queries,candidates,keys,dimension,kv_heads):
    shapes = dict(attention=(layers,heads,queries,keys+1),
        native=(layers,heads,queries,candidates,dimension),
        local=(layers,heads,queries,candidates,dimension),
        self_values=(layers,heads,queries,dimension),
        values=(layers,kv_heads,keys,dimension))
    return {name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype='float32',shape=shape)
            for name,shape in shapes.items()}


def save_batch(model,capture,contrast,native_gradients,files,begin,end):
    for layer,module in enumerate(model.model.layers):
        heads = model.config.num_attention_heads
        dimension = module.self_attn.head_dim
        projection = module.self_attn.o_proj.weight.float()
        native = torch.stack([values[layer][0] for values in native_gradients],1)
        immediate = capture.residuals[layer]+capture.writes[layer][0].detach()
        local = lens_gradient(immediate,contrast,model.model.norm)
        for name,gradient in (('native',native),('local',local)):
            factor = (gradient@projection).reshape(end-begin,3,heads,dimension)
            files[name][layer,:,begin:end] = factor.permute(2,0,1,3).detach().cpu().numpy()
        attention,_,self_value = capture.messages[layer]
        files['attention'][layer,:,begin:end] = attention.cpu().numpy()
        files['self_values'][layer,:,begin:end] = self_value.cpu().numpy()


def capture_record(model,row,output,batch):
    directory = output/row['key']
    directory.mkdir()
    prompt,answer = row['prompt'],row['answer']
    tokens = prompt+answer[:-1]
    cache,_,normalized = prefill(model,prompt,answer,checkpoints=(32,))
    indices = torch.tensor(row['positions'],device=model.device)
    targets,candidates,nll,entropy = choose_candidates(model,normalized,answer,indices)
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    dimension = model.config.hidden_size//heads
    files = allocate(directory,layers,heads,len(indices),3,len(tokens),dimension,model.config.num_key_value_heads)
    for layer in range(layers):
        files['values'][layer] = cache.layers[layer].values[0].cpu().numpy()
    errors,margins = [],[]
    for begin in range(0,len(indices),batch):
        end = min(begin+batch,len(indices))
        positions = len(prompt)-1+indices[begin:end]
        final,_,captured = replay(model,cache,tokens,positions,checkpoints=(32,))
        contrast = model.lm_head.weight[targets[begin:end]].float()[:,None]-model.lm_head.weight[candidates[begin:end]].float()
        values = (final[:,None]*contrast).sum(-1)
        reference = (normalized[indices[begin:end],None]*contrast).sum(-1)
        errors.append(float((values-reference).abs().max().detach()))
        assert errors[-1]<.005, (row['key'],errors[-1])
        gradients = [torch.autograd.grad(values[:,candidate].sum(),captured.writes,retain_graph=candidate<2)
                     for candidate in range(3)]
        save_batch(model,captured,contrast,gradients,files,begin,end)
        margins.append(values.detach().cpu().numpy())
        del gradients,captured,final,values
    for value in files.values():
        value.flush()
    np.savez_compressed(directory/'readout.npz',positions=indices.cpu().numpy(),
        candidates=candidates.cpu().numpy(),target=targets.cpu().numpy(),margin=np.concatenate(margins),
        nll=nll.cpu().numpy(),entropy=entropy.cpu().numpy(),replay_error=max(errors))
    return dict(key=row['key'],queries=len(indices),max_margin_replay_error=max(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    model = load_model(manifest['model'])
    started = perf_counter()
    results = []
    for row in manifest['records']:
        result = capture_record(model,row,args.output,manifest['query_batch'])
        results.append(result)
        write_json(args.output/'progress.json',dict(completed=results,seconds=perf_counter()-started))
        print(result,round(perf_counter()-started,1),flush=True)
    write_json(args.output/'capture_complete.json',dict(status='complete',records=results,
        seconds=perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        execution='frozen BF16 weights, FP32 operations and exact current-query downstream derivatives',
        scope='original past K/V held fixed; no cross-time total derivative'))


if __name__=='__main__':
    main()
