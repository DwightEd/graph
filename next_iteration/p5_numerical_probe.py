"""Engineering-only P5 first-response batch/cache SDPA backend comparison, no GT."""
import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext
import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from torch.nn.attention import sdpa_kernel, SDPBackend
from .local_grounding import messages, units, common_prefix
from .grounding_contrast import digest, INPUT_SHA


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    b=Path('/share/home/tm902089733300000/a903202310/lys')
    path=b/'research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl'
    assert digest(path)==INPUT_SHA
    row=next(r for r in map(json.loads,path.read_text().splitlines()) if str(r['id'])=='17019')
    torch.set_num_threads(4)
    start=time.time()
    t=AutoTokenizer.from_pretrained(str(b/'models/Qwen3-8B'),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(b/'models/Qwen3-8B'),local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    # Stored pretrained weights unchanged; cast one linear at a time for IEEE FP32 compute.
    # Avoid a resident32GB FP32 copy of the whole8B model on the24GB GPU.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    def fp32_linear(self, x):
        return torch.nn.functional.linear(x.float(),self.weight.float(),
            self.bias.float() if self.bias is not None else None)
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):module.forward=types.MethodType(fp32_linear,module)
    model.model.embed_tokens.register_forward_hook(lambda module,args,result: result.float())
    lo,hi=row['source_span']; source=row['prompt'][lo:hi]
    instruction=row['prompt'][:lo]+'[SOURCE OMITTED HERE]'+row['prompt'][hi:]
    seqs=[t.apply_chat_template(messages(source,instruction,row['response'],s),tokenize=True,
          add_generation_prompt=True,enable_thinking=False) for s in units(row['response'])[:4]]
    labels=[t.encode(s,add_special_tokens=False)[0] for s in ('A','B')]
    prefix=common_prefix(seqs)
    def readout(hidden,lengths):
        n=hidden.shape[0]
        z=model.lm_head(hidden[torch.arange(n,device='cuda'),torch.tensor(lengths,device='cuda')-1]).float()
        return z[:,labels].cpu().tolist()
    def batch_sequences(sequences):
        width=max(map(len,sequences));n=len(sequences)
        ids=torch.full((n,width),t.pad_token_id,device='cuda',dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,s in enumerate(sequences):
            ids[i,:len(s)]=torch.tensor(s,device='cuda');mask[i,:len(s)]=1
        return ids,mask
    report={'id':'17019','no_annotations':True,'code_sha256':digest(__file__),'query_lengths':list(map(len,seqs)),'backends':{}}
    for name,backend in [('fp32_math',[SDPBackend.MATH])]:
        with (nullcontext() if backend is None else sdpa_kernel(backend)):
            singles=[]
            for s in seqs:
                x=torch.tensor([s],device='cuda')
                singles.extend(readout(model.model(input_ids=x,use_cache=False).last_hidden_state,[len(s)]))
            ids,mask=batch_sequences(seqs)
            batched=readout(model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state,list(map(len,seqs)))
            base=model.model(input_ids=torch.tensor([seqs[0][:prefix]],device='cuda'),use_cache=True).past_key_values
            cache=DynamicCache(ddp_cache_data=((k.repeat_interleave(4,dim=0),v.repeat_interleave(4,dim=0)) for k,v in base.to_legacy_cache()))
            suffixes=[s[prefix:] for s in seqs];ids,mask=batch_sequences(suffixes)
            mask=torch.cat([torch.ones((4,prefix),device='cuda',dtype=torch.long),mask],dim=1)
            pos=torch.arange(prefix,prefix+ids.shape[1],device='cuda')
            hidden=model.model(input_ids=ids,attention_mask=mask,position_ids=pos[None,:].expand(4,-1),
                cache_position=pos,past_key_values=cache,use_cache=True).last_hidden_state
            cached=readout(hidden,list(map(len,suffixes)))
            report['backends'][name]={'single_full':singles,'batch_full':batched,'batch_cached':cached}
            print(json.dumps({name:report['backends'][name]}),flush=True)
            del cache,base,hidden
    report['elapsed_seconds']=time.time()-start
    report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    with output.open('x') as f:json.dump(report,f,indent=2)
    Path(str(output)+'.executed_code.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
