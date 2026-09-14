"""B8 fixed synthetic factorial. No natural labels, fitting or P7 alteration."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import types
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
INPUTS=ROOT/'outputs/b8_binding_factorial_inputs_20260914_v1'
EXPECTED={
 'inputs.jsonl':'ef722eb71af55561b91313582fd7cacb8d106395a6a844191e01aa2c412271b3',
 'manifest.json':'1f3a1c765f43b09f9686e1cb24801a598189aa98cbec19e9ee55f5e5b10c6331',
 'tokenized_queries.json':'0020027a473160ff573a147dcf8923ddb5d1c06acb31f7c2697db4fd4a613ff0',
 'preflight.json':'406b9c9902e5bced557da755bac004234853bfd38e9df4c0aae4400a0b009ac9',
}
P7_SHA='f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f'
DEPENDENCY_SHA='ab0c13df4bd4ac909e76d50ad19b411c36d3d2d92f9d4e668c4f879ef5374362'
P7_FREEZE_SHA='4cc72137c8ca36ade8578ccddcd64d597e2208675a3f5c7044bff1fa23ea0830'
P7_ROSTER_SHA='cb16edbc11ad1d5cf637481d038d2ac2bec0f72136df97fc461103b9f06d196e'
WEIGHT_SHA={
 'model-00001-of-00005.safetensors':'31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f',
 'model-00002-of-00005.safetensors':'5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282',
 'model-00003-of-00005.safetensors':'c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836',
 'model-00004-of-00005.safetensors':'b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a',
 'model-00005-of-00005.safetensors':'20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff',
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def p7_completion_identity():
    launch_path=ROOT/'outputs/P7_CONFIRMATION_LAUNCH_20260914.json'
    manifest_path=ROOT/'outputs/p7_reasoned_confirmation_20260914_v1/manifest.json'
    launch=json.loads(launch_path.read_text())
    m=json.loads(manifest_path.read_text())
    if (launch.get('status')!='completed' or launch.get('actual_subprocess_returncode')!=0
        or launch.get('attempt')!=1 or m.get('complete') is not True or m.get('error') is not None):
        raise ValueError('P7 must complete successfully before B8 uses GPU')
    roster=ROOT/'outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl'
    if sha(roster)!=P7_ROSTER_SHA or sha(ROOT/'outputs/P7_CONFIRMATION_FREEZE_20260914.json')!=P7_FREEZE_SHA:
        raise ValueError('P7 roster/freeze identity')
    ids=[str(json.loads(line)['id']) for line in roster.read_text().splitlines()]
    if len(ids)!=128 or len(set(ids))!=128 or m.get('planned_ids')!=ids or m.get('completed_ids')!=ids:
        raise ValueError('exact ordered unique P7 full128 IDs required')
    if (m.get('confirmation_freeze_sha256')!=P7_FREEZE_SHA
        or m.get('code_sha256')!='5bc5b2639cd4934404fa2cc36917a6ad5d164d8665797c0541f5a1792764a865'
        or launch.get('output')!=str(manifest_path.parent)):
        raise ValueError('P7 completed method/output binding')
    if sha(Path(launch['log']))!=launch.get('log_sha256'):
        raise ValueError('P7 actual-exit log binding')
    return dict(launch_sha256=sha(launch_path),manifest_sha256=sha(manifest_path),freeze_sha256=P7_FREEZE_SHA)

def idle_gpu():
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if len(memory.splitlines())!=1 or int(memory.strip())>=500:raise ValueError('one idle GPU required')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if apps.strip():raise ValueError('another compute process is present')

@contextmanager
def task_gpu_lock():
    # Cooperative local-task exclusion, NOT a platform-wide GPU reservation.
    import fcntl
    with (ROOT/'outputs/B8_GPU0.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another cooperating B8 process holds the GPU lock')
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def load_inputs():
    for name,expected in EXPECTED.items():
        if sha(INPUTS/name)!=expected:raise ValueError(f'input drift: {name}')
    if sha(ROOT/'next_iteration/local_grounding_reasoned.py')!=P7_SHA:
        raise ValueError('parent query changed')
    if sha(ROOT/'next_iteration/grounding_contrast.py')!=DEPENDENCY_SHA:
        raise ValueError('imported dependency changed')
    preflight=json.loads((INPUTS/'preflight.json').read_text())
    if preflight['status']!='pass':raise ValueError('length-matching preflight did not pass')
    rows=[json.loads(s) for s in (INPUTS/'inputs.jsonl').read_text().splitlines()]
    queries=json.loads((INPUTS/'tokenized_queries.json').read_text())
    if len(rows)!=160 or len(queries)!=160 or [r['id'] for r in rows]!=[q['id'] for q in queries]:
        raise ValueError('exact complete160 identity required')
    return rows,queries,preflight

def run(args):
    rows,queries,preflight=load_inputs()
    if not args.execute:
        print(json.dumps(dict(status='ready_inputs_only',rows=160,model_forwards=0)));return
    p7_identity=p7_completion_identity()
    idle_gpu()
    model_dir=BASE/'models/Qwen3-8B'
    for name,expected in preflight['tokenizer_metadata_sha256'].items():
        if sha(model_dir/name)!=expected:raise ValueError(f'model metadata drift: {name}')
    actual_weights={name:sha(model_dir/name) for name in WEIGHT_SHA}
    if actual_weights!=WEIGHT_SHA:raise ValueError('model shard identity drift')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    state=dict(complete=False,planned_ids=[r['id'] for r in rows],completed_ids=[],
        input_sha256=EXPECTED,preflight_sha256=sha(INPUTS/'preflight.json'),code_sha256=sha(__file__),
        started_unix=start,scope='synthetic_by_construction; external Qwen final verifier only',
        natural_annotations_read=0,p7_predictions_or_metrics_read=False,
        p7_completion=p7_identity,model_shard_sha256=actual_weights,
        dependency_sha256=DEPENDENCY_SHA,lock_scope='cooperating B8 processes only; not platform reservation',
        score='raw z(B)-z(A), no normalization, fitting or natural AUROC',
        precision='unchanged BF16 stored weights; per-layer FP32 linear, FP32 activations, math SDPA, TF32 disabled',
        forwards=0,checks=[],results=[])
    def save():
        (out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    (out/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    save()
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from torch.nn.attention import sdpa_kernel,SDPBackend
        torch.set_num_threads(4)
        torch.manual_seed(20260914)
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        # Reconstruct every exact message token stream before loading GPU weights.
        from .local_grounding_reasoned import messages,add_audit
        for row,query in zip(rows,queries,strict=True):
            msg=messages(row['source'],row['instruction'],row['answer'],row['target_span'])
            if row['draft'] is not None:msg=add_audit(msg,row['draft'],False)
            ids=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if msg!=query['messages'] or ids!=query['input_ids']:raise ValueError('query/token stream drift')
        idle_gpu()  # Recheck after CPU hashing/token reconstruction, under the cooperative lock.
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        def fp32_linear(module,x):
            return torch.nn.functional.linear(x.float(),module.weight.float(),
                module.bias.float() if module.bias is not None else None)
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):module.forward=types.MethodType(fp32_linear,module)
        model.model.embed_tokens.register_forward_hook(lambda module,inputs,result:result.float())
        if [tokenizer.encode(x,add_special_tokens=False) for x in ('A','B')]!=[[32],[33]]:
            raise ValueError('A/B token identity changed')
        state['versions']=dict(torch=torch.__version__,transformers=__import__('transformers').__version__)
        state['model_metadata_sha256']=preflight['tokenizer_metadata_sha256']
        state['label_ids']=[32,33]
        @torch.inference_mode()
        @sdpa_kernel([SDPBackend.MATH])
        def score(ids):
            if len(ids)>model.config.max_position_embeddings:raise ValueError('no truncation allowed')
            x=torch.tensor([ids],device='cuda')
            hidden=model.model(input_ids=x,use_cache=False).last_hidden_state[:,-1]
            z=model.lm_head(hidden).float()[0]
            state['forwards']+=1
            return z[[32,33]].cpu().tolist()
        for i,(row,query) in enumerate(zip(rows,queries,strict=True)):
            ab=score(query['input_ids'])
            if not all(math.isfinite(v) for v in ab):raise ValueError('non-finite raw logits')
            risk=ab[1]-ab[0]
            if not math.isfinite(risk):raise ValueError('non-finite derived risk')
            if i%20==0:
                repeat=score(query['input_ids'])
                if not all(math.isfinite(v) for v in repeat):raise ValueError('non-finite repeat logits')
                delta=max(abs(a-b) for a,b in zip(ab,repeat))
                odds_delta=abs(risk-(repeat[1]-repeat[0]))
                if not all(math.isfinite(v) for v in (delta,odds_delta)):
                    raise ValueError('non-finite repeat difference')
                state['checks'].append(dict(id=row['id'],ab=ab,repeat=repeat,max_ab_delta=delta,odds_delta=odds_delta))
                if delta>0.005 or odds_delta>0.005:raise ValueError('fixed numerical guard failed')
            result=dict(id=row['id'],logits=ab,risk=risk,input_tokens=len(query['input_ids']),
                        input_ids_sha256=hashlib.sha256(json.dumps(query['input_ids']).encode()).hexdigest())
            state['results'].append(result)
            state['completed_ids'].append(row['id'])
            state['elapsed_seconds']=time.time()-start
            save()
            if (i+1)%20==0:print(json.dumps(dict(completed=i+1,planned=160,elapsed=state['elapsed_seconds'])),flush=True)
        state.update(complete=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-start)
        save()
        print(json.dumps(dict(complete=True,rows=160,forwards=state['forwards'],manifest_sha256=sha(out/'manifest.json'))))
    except BaseException as exc:
        state.update(error=repr(exc),elapsed_seconds=time.time()-start)
        save()
        raise
    finally:
        # Tear down this model before releasing the cooperative lock.
        if 'model' in locals():del model
        if 'torch' in locals():
            import gc
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:return run(args)
    with task_gpu_lock():return run(args)

if __name__=='__main__':main()
