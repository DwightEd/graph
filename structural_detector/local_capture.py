"""One teacher-forced replay per response, storing only local endpoint attention.

Consumes reanchor population inputs.jsonl (no annotations), not its five scalar
columns. Eager attention is released after each layer by a read-only hook that
replaces only the returned diagnostic attention tensor with None.
"""
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_records(population, tasks=('QA',), generator='llama-2-7b-chat'):
    rows = []
    with (Path(population) / 'inputs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if (tasks == ('all',) or r['task'] in tasks) and (generator == 'all' or r['generator'] == generator):
                rows.append(r)
    if not rows or len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('empty or duplicate selected roster')
    if any(not str(r['id']).isdigit() for r in rows):
        raise ValueError('numeric response IDs required')
    return rows


def attention_band(attention, prompt_length, response_tokens, window, source_mask):
    """Row t forecasts response[t], key lag d is response[t-d], including d=1.

    d=1 is query self at the preceding emitted token, NOT the future target.
    Full-row normalization corrects quantized row sums; local rows stay subunit.
    """
    import torch
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError('batch-one eager [B,H,N,N] attention required')
    a = attention[0]
    p, t = prompt_length, response_tokens
    if p < 1 or window < 1 or a.shape[-1] < p + t - 1 or len(source_mask) != p:
        raise ValueError('attention, prompt and forecast length mismatch')
    rows = a[:, p-1:p+t-1, :]
    row_float = rows.float()
    norm = row_float.sum(-1)
    if not torch.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError('invalid native attention row')
    positions = torch.arange(t, device=a.device)
    lag = torch.arange(1, window+1, device=a.device)
    valid = positions[:,None] >= lag[None,:]
    keys = (p + positions[:,None] - lag[None,:]).clamp_min(0)
    gathered = rows.gather(-1,keys[None].expand(a.shape[0],-1,-1)).float()
    band = ((gathered / norm[:,:,None])*valid[None]).permute(1,0,2)
    sm = torch.as_tensor(source_mask, dtype=torch.bool, device=a.device)
    evidence = (row_float[:, :, :p][:, :, sm].sum(-1) / norm).T
    # S10 uses keys <= query-16; local lag 1..16 is its excluded region.
    columns = torch.arange(a.shape[-1],device=a.device)
    remote_mask = (columns[None,:]>=p) & (columns[None,:] <= p+positions[:,None]-1-16)
    remote = ((row_float*remote_mask[None]).sum(-1)/norm).T
    return band.cpu().numpy(), evidence.cpu().numpy(), remote.cpu().numpy()



def replay(model, record, window=16, logit_chunk=16):
    import torch
    p = int(record['prompt_length'])
    token_ids = record['token_ids']
    t = len(record['offsets'])
    if len(token_ids) != p + t or len(record['source_mask']) != p or not t:
        raise ValueError('input roster is not aligned to original token IDs')
    layers = model.model.layers
    bands, sources, remotes, handles = {}, {}, {}, []
    def hook_for(layer):
        def hook(module, args, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise ValueError('eager attention weights unavailable; no scalar-cache fallback')
            bands[layer], sources[layer], remotes[layer] = attention_band(
                output[1], p, t, window, record['source_mask'])
            return (output[0], None, *output[2:])
        return hook
    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(hook_for(i)))
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            ids = torch.tensor([token_ids[:-1]], device=device)
            hidden = model.model(input_ids=ids, use_cache=False, output_attentions=True,
                                 output_hidden_states=False, return_dict=True).last_hidden_state[0, p-1:]
            entropies, margins = [], []
            for start in range(0, t, logit_chunk):
                z = model.lm_head(hidden[start:start+logit_chunk]).float()
                logp = z.log_softmax(-1)
                entropies.extend((-(logp.exp()*logp).sum(-1)).cpu().tolist())
                top2 = z.topk(2, dim=-1).values
                margins.extend((top2[:,1]-top2[:,0]).cpu().tolist())
            del hidden
    finally:
        for handle in handles:
            handle.remove()
    if len(bands) != len(layers):
        raise ValueError('not all attention layers were captured')
    order = sorted(bands)
    evidence = np.concatenate([sources[i] for i in order], axis=1)
    remote = np.concatenate([remotes[i] for i in order], axis=1)
    spread = np.stack([sources[i].std(axis=1) for i in order]).mean(axis=0)
    values = np.column_stack((entropies, margins, evidence.mean(1), remote.mean(1), spread))
    channels = np.array([(i,h) for i in order for h in range(bands[i].shape[1])], np.int32)
    return dict(values=values.astype(np.float32), edges=np.concatenate([bands[i] for i in order],1).astype(np.float32),
                evidence=evidence.astype(np.float32), offsets=np.asarray(record['offsets'],np.int32), channels=channels)


def capture(population, output, *, tasks=('QA',), generator='llama-2-7b-chat', model_path=None,
            device='cuda:0', window=16, max_tokens=4096, dtype='bfloat16', resume=False):
    import torch
    rows = select_records(population, tasks, generator)
    settings = json.loads((Path(population)/'settings.json').read_text())
    model_path = str(Path(model_path or settings['model']).resolve())
    model_files=[(p.name,p.stat().st_size,p.stat().st_mtime_ns)
                 for p in sorted(Path(model_path).glob('*')) if p.is_file()]
    contract = dict(schema='local-risk-edges-v1', window=window, model=model_path, dtype=dtype,
                    roster_digest=digest(rows), model_files=digest(model_files), tasks=list(tasks), generator=generator,
                    forecast='row t = P+t-1; lag d key = P+t-d', labels_used=False)
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError('capture output exists; use --resume or a new output')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'settings.json').exists():
            if json.loads((out/'settings.json').read_text()) != contract:
                raise ValueError('resume model/roster/window differs')
        else:
            write(out/'settings.json', contract)
        long = [(r['id'],len(r['token_ids'])-1) for r in rows if len(r['token_ids'])-1>max_tokens]
        if long:
            raise ValueError(f'{len(long)} inputs exceed max_tokens={max_tokens}; no truncation. Examples: {long[:4]}')
        model, completed = None, []
        try:
            for i, r in enumerate(rows):
                rid = str(r['id']); path = out/(rid+'.npz')
                if path.exists():
                    with np.load(path,allow_pickle=False) as old:
                        if str(old['input_digest']) != digest(r):
                            raise ValueError('cached original input changed: '+rid)
                else:
                    if model is None:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                  torch_dtype=getattr(torch,dtype),attn_implementation='eager').to(device).eval()
                        if not hasattr(model,'model') or not hasattr(model.model,'layers'):
                            raise ValueError('capture supports Llama-style causal decoders only')
                    print(f'capture {i+1}/{len(rows)} id={rid} input_tokens={len(r["token_ids"])-1}',flush=True)
                    arrays = replay(model,r,window)
                    tmp = path.with_suffix('.partial.npz')
                    np.savez_compressed(tmp, input_digest=digest(r), **arrays)
                    tmp.replace(path)
                    del arrays
                completed.append({k:r[k] for k in ('id','source_id','task','generator','official_split','response_sha256')})
                print(f'local attention {i+1}/{len(rows)} {r["task"]}/{rid}; one replay per uncached answer',flush=True)
            write(out/'records.json',completed)
            write(out/'complete.json',dict(complete=True,responses=len(rows),labels_used=False))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out
