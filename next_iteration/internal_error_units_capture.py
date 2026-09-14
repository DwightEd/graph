"""Capture native Llama internal states/messages; never reads construction or natural gold."""
import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .internal_error_units import LAYERS, SEED, digest, write_json
from .revisit_state import head_statistics, summarize_events


def groups_for_queries(prompt_length, total, source_mask):
    p = prompt_length
    source = np.zeros(total, bool)
    source[:p] = source_mask
    q = np.arange(p, total)
    history = (np.arange(total)[None] >= p) & (np.arange(total)[None] < q[:, None])
    return source, history


def reverse_history_indices(prompt_length, total):
    """Permute only already-observed history, separately for every query."""
    q = np.arange(prompt_length, total)[:, None]
    k = np.broadcast_to(np.arange(total), (total-prompt_length, total)).copy()
    allowed = (k >= prompt_length) & (k < q)
    return np.where(allowed, prompt_length + q - 1 - k, k)


class Capture:
    def __init__(self, model, tokenizer):
        import torch
        from transformers.models.llama import modeling_llama
        self.torch, self.module = torch, modeling_llama
        self.model, self.tokenizer = model, tokenizer
        self.original = modeling_llama.eager_attention_forward
        self.enabled = False
        self.forward_count = 0
        self.handles = []
        modeling_llama.eager_attention_forward = self.attention
        for li in LAYERS:
            def node_hook(module, args, output, li=li):
                if self.enabled:
                    out = output[0] if isinstance(output, tuple) else output
                    self.arrays[f'nodes_{li}'] = self.pack(out[0])
            def mlp_hook(module, args, output, li=li):
                if self.enabled:
                    self.arrays[f'mlp_{li}'] = self.pack(output[0, self.p:])
            self.handles.append(model.model.layers[li].register_forward_hook(node_hook))
            self.handles.append(model.model.layers[li].mlp.register_forward_hook(mlp_hook))

    def close(self):
        self.module.eager_attention_forward = self.original
        for h in self.handles:
            h.remove()

    def pack(self, tensor):
        a = tensor.detach().float().cpu().numpy()
        packed = a.astype(np.float16)
        if not np.isfinite(packed).all():
            raise ValueError('non-finite float16 stored feature')
        self.quantization = max(self.quantization, float(np.max(np.abs(a-packed.astype(np.float32)))))
        self.underflows += int(np.sum((a != 0) & (packed == 0)))
        return packed

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        torch = self.torch
        output, a = self.original(module, query, key, value, attention_mask, **kwargs)
        if not self.enabled:
            return output, a
        li = module.layer_idx
        future = torch.arange(self.total,device=a.device)[None,:] > torch.arange(self.total,device=a.device)[:,None]
        if torch.count_nonzero(a[0] * future[None]):
            raise ValueError('all-layer attention has future mass')
        pre = a[0, :, self.p-1:self.total-1, :].float()
        shift, mass = head_statistics(pre.cpu().numpy(), self.source[:self.p], self.special)
        self.shifts.append(shift)
        self.common.append(mass)
        self.visited.append(li)
        self.mass_source += pre[:, :, torch.as_tensor(self.source, device=a.device)].sum(-1).mean(0).cpu().numpy()
        qpre = torch.arange(self.p-1, self.total-1, device=a.device)
        keys = torch.arange(self.total, device=a.device)
        histpre = (keys[None] >= self.p) & (keys[None] <= qpre[:, None])
        self.mass_history += (pre * histpre[None]).sum(-1).mean(0).cpu().numpy()
        if li in LAYERS:
            post = a[0, :, self.p:, :].float()
            post_queries = torch.arange(self.p,self.total,device=a.device)
            if torch.count_nonzero(post * (keys[None] > post_queries[:,None])[None]):
                raise ValueError('post-token attention contains future mass')
            v = self.module.repeat_kv(value, module.num_key_value_groups)[0].float()
            self.arrays[f'attention_{li}'] = self.pack(post)
            row_error = np.max(np.abs(self.arrays[f'attention_{li}'].astype(np.float32).sum(-1)-post.sum(-1).cpu().numpy()))
            self.attention_storage_mass_error = max(self.attention_storage_mass_error,float(row_error))
            self.arrays[f'values_{li}'] = self.pack(value[0])
            source_idx = torch.as_tensor(np.flatnonzero(self.source), device=a.device)
            hist = torch.as_tensor(self.history, device=a.device)
            rev_idx = torch.as_tensor(self.reverse, device=a.device)
            source_head = post[:, :, source_idx] @ v[:, source_idx]
            source_wrong = post[:, :, source_idx].flip(-1) @ v[:, source_idx]
            hist_a = post * hist[None]
            hist_head = hist_a @ v
            hist_wrong = hist_a.gather(2, rev_idx[None].expand(post.shape[0], -1, -1)) @ v
            def project(heads):
                flat = heads.transpose(0, 1).reshape(self.n, -1)
                return torch.nn.functional.linear(flat, module.o_proj.weight.float())
            self.arrays[f'messages_{li}'] = self.pack(torch.stack([
                project(source_head), project(hist_head), project(source_wrong), project(hist_wrong)], 1))
            other_prompt = torch.as_tensor((np.arange(self.total)<self.p)&~self.source,device=a.device)
            other_head = post[:,:,other_prompt] @ v[:,other_prompt]
            self_a = post * (keys[None]==post_queries[:,None])[None]
            self_head = self_a @ v
            # Audit the same grouped source/history tensors actually saved, plus complementary groups.
            group_sum = project(source_head)+project(hist_head)+project(other_head)+project(self_head)
            exact_projected = project(output[0,self.p:].transpose(0,1).float())
            o_error = float(torch.linalg.vector_norm(group_sum-exact_projected) /
                            torch.linalg.vector_norm(exact_projected).clamp_min(1e-8))
            self.o_reconstruction = max(self.o_reconstruction,o_error)
            if o_error > .02:
                raise ValueError(f'grouped O-message reconstruction relative error {o_error}')
            # Arithmetic audit on ALL keys, independent of the grouping ablation.
            reconstructed = post @ v
            exact = output[0, self.p:].transpose(0, 1).float()
            rel = float(torch.linalg.vector_norm(reconstructed-exact) /
                        torch.linalg.vector_norm(exact).clamp_min(1e-8))
            self.reconstruction = max(self.reconstruction, rel)
            if rel > .02:
                raise ValueError(f'head reconstruction relative error {rel}')
        return output, a

    def run(self, row, *, verify=False):
        torch = self.torch
        self.p, self.total = row['prompt_length'], len(row['token_ids'])
        self.n = self.total-self.p
        self.source, self.history = groups_for_queries(self.p, self.total, row['source_mask'])
        self.reverse = reverse_history_indices(self.p, self.total)
        self.special = np.isin(row['token_ids'], self.tokenizer.all_special_ids)
        self.arrays, self.shifts, self.common, self.visited = {}, [], [], []
        self.quantization, self.reconstruction, self.o_reconstruction = 0., 0., 0.
        self.underflows, self.attention_storage_mass_error = 0, 0.
        self.mass_source, self.mass_history = np.zeros(self.n), np.zeros(self.n)
        ids = torch.tensor([row['token_ids']], device='cuda')
        with torch.inference_mode():
            self.enabled = True
            self.forward_count += 1
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            self.enabled = False
            hook_delta, prefix_delta, suffix_delta = None, None, None
            suffix_alternative_delta = None
            extra_forwards = 0
            if verify:
                self.forward_count += 1
                check = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                hook_delta = float((hidden-check).abs().max())
                if hook_delta != 0:
                    raise ValueError('capture hooks changed model hidden outputs')
                prefix_delta = suffix_delta = suffix_alternative_delta = 0.
                extra_forwards = 1
                # Every observed response position with a future suffix, independent of any gold core.
                for cut in range(self.p,self.total):
                    self.forward_count += 1
                    short = self.model.model(input_ids=ids[:, :cut], use_cache=False).last_hidden_state[0]
                    prefix_delta = max(prefix_delta,float((hidden[cut-1].float()-short[-1].float()).abs().max()))
                    extra_forwards += 1
                    del short
                    changed = ids.clone()
                    changed[:,cut:] = self.tokenizer.eos_token_id
                    if torch.equal(changed[:,cut:],ids[:,cut:]):
                        raise ValueError('future replacement made no change')
                    self.forward_count += 1
                    alternative = self.model.model(input_ids=changed, use_cache=False).last_hidden_state[0]
                    suffix_delta = max(suffix_delta,float((hidden[cut-1].float()-alternative[cut-1].float()).abs().max()))
                    extra_forwards += 1
                    del alternative,changed
                    if cut == self.p:
                        changed = ids.clone()
                        changed[:,cut:] = self.tokenizer.bos_token_id
                        if torch.equal(changed[:,cut:],ids[:,cut:]):
                            raise ValueError('alternative future replacement made no change')
                        self.forward_count += 1
                        alternative = self.model.model(input_ids=changed, use_cache=False).last_hidden_state[0]
                        suffix_alternative_delta = float((hidden[cut-1].float()-alternative[cut-1].float()).abs().max())
                        extra_forwards += 1
                        del alternative,changed
                if suffix_delta != 0 or suffix_alternative_delta != 0:
                    raise ValueError(f'fixed-shape future replacement changed prefix state: {suffix_delta}')
                del check
            controls = []
            for start in range(0, self.n, 16):
                end = min(self.n, start+16)
                z = self.model.lm_head(hidden[self.p-1+start:self.p-1+end]).float()
                lp = z.log_softmax(-1)
                top = z.topk(2).values
                idx = ids[0, self.p+start:self.p+end]
                controls.append(torch.stack([-(lp.exp()*lp).sum(-1),
                    -lp.gather(1, idx[:, None])[:, 0], top[:, 1]-top[:, 0]], -1).cpu().numpy())
        if self.visited != list(range(len(self.model.model.layers))):
            raise ValueError('all-layer event capture missing')
        event = summarize_events(np.stack(self.shifts), np.stack(self.common))
        source, history = self.mass_source/len(self.visited), self.mass_history/len(self.visited)
        self.arrays.update(controls=np.concatenate(controls), source_mass=source, history_mass=history,
            displacement=history-source, event=event['event'], revisit=np.nan_to_num(event['revisit']),
            full_token_ids=np.asarray(row['token_ids']), source_mask=self.source,
            offsets=np.asarray(row['offsets']), prompt_length=np.array(self.p),
            head_shift=np.stack(self.shifts), head_common_mass=np.stack(self.common))
        audit = dict(quantization_max_abs=self.quantization, reconstruction_relative=self.reconstruction,
            grouped_o_reconstruction_relative=self.o_reconstruction,stored_underflow_count=self.underflows,
            attention_storage_row_mass_error=self.attention_storage_mass_error,
            hook_hidden_max_abs=hook_delta, prefix_hidden_max_abs=prefix_delta,
            fixed_shape_suffix_hidden_max_abs=suffix_delta,
            alternative_suffix_hidden_max_abs=suffix_alternative_delta,
            model_forwards=1+extra_forwards, observed_tokens=self.n,
            timing='pre q=P+t-1; nodes/messages post q=P+t; no future response aggregation')
        del hidden
        return self.arrays, audit


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    inputs = Path(args.inputs)
    parent = json.loads((inputs.parent/'manifest.json').read_text())
    if not parent['complete'] or digest(inputs) != parent['inputs_sha256']:
        raise ValueError('input manifest identity mismatch')
    args.model = str(Path(args.model).resolve())
    metadata = {p.name:digest(p) for p in Path(args.model).glob('*.json')}
    if parent.get('model') != args.model or parent.get('tokenizer_metadata_sha256') != metadata:
        raise ValueError('prepared tokenizer/model identity mismatch')
    rows = [json.loads(s) for s in inputs.read_text().splitlines()]
    # gold.json is deliberately never opened in this module.
    if args.phase == 'pilot':
        rows = [r for r in rows if r['source_id'] == 'location-00' and '-o0-' in r['id']]
        if len(rows) != 4:
            raise ValueError('pilot factorial scope must be exactly four')
    elif len(rows) != 800:
        raise ValueError('full corpus must have 768 controlled + 32 observer responses')
    lock_path = Path(__file__).resolve().parents[2]/'reanchor/runs/relation_interleave_20260913.lock'
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import subprocess
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.splitlines()[0]) >= 500:
        raise RuntimeError('GPU occupied')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 30*1024**3:
        raise RuntimeError('less than 30 GiB free storage')
    manifest = dict(complete=False, planned=len(rows), completed_ids=[], phase=args.phase,
        input_sha256=digest(inputs), model=args.model, seed=SEED,
        labels_read=False, events='original all-layer/all-head W1; window16 quantile.95',
        layers=LAYERS, output_sha256={}, model_forwards=0,
        message_channels=['source','history','source_reversed','history_reversed'],
        timing='W1/logits pre; binding messages and MLP post; self audit only',
        code_sha256={Path(p).name:digest(p) for p in [__file__,
            Path(__file__).with_name('internal_error_units.py'), Path(__file__).with_name('revisit_state.py')]},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__))
    write_json(out/'manifest.json', manifest)
    for name in manifest['code_sha256']:
        shutil.copyfile(Path(__file__).with_name(name), out/('executed_'+name))
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        manifest['model_weights_sha256'] = {p.name:digest(p) for p in sorted(Path(args.model).glob('*.safetensors'))}
        if not manifest['model_weights_sha256']:
            raise ValueError('no resolved safetensors checkpoint files')
        write_json(out/'manifest.json',manifest)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        if model.config.model_type != 'llama' or len(model.model.layers) != 32:
            raise ValueError('this capture is specified for Llama 32 layers')
        manifest['model_metadata_sha256'] = {p.name:digest(p) for p in Path(args.model).glob('*.json')}
        capture = Capture(model, tok)
        try:
            for i, row in enumerate(rows):
                if time.monotonic()-started > args.max_seconds:
                    raise RuntimeError('fixed capture wall budget exceeded')
                arrays, audit = capture.run(row, verify=(args.phase == 'pilot'))
                artifact = out/(row['id']+'.npz')
                np.savez_compressed(artifact, **arrays)
                result = dict(id=row['id'], source_id=row['source_id'], family=row['family'],
                    split=row['split'], mode=row['mode'], audit=audit, artifact_sha256=digest(artifact))
                write_json(out/(row['id']+'.json'), result)
                manifest['completed_ids'].append(row['id'])
                manifest['model_forwards'] += audit['model_forwards']
                manifest['output_sha256'][artifact.name] = result['artifact_sha256']
                manifest['output_sha256'][row['id']+'.json'] = digest(out/(row['id']+'.json'))
                manifest['elapsed_seconds'] = time.monotonic()-started
                write_json(out/'manifest.json', manifest)
                if i%8 == 0 or i+1 == len(rows):
                    print(json.dumps(dict(completed=i+1, planned=len(rows), id=row['id'],
                                          seconds=manifest['elapsed_seconds'])), flush=True)
                del arrays
        finally:
            manifest['actual_model_forwards_including_failed_rows'] = capture.forward_count
            write_json(out/'manifest.json',manifest)
            capture.close()
        manifest.update(complete=True, elapsed_seconds=time.monotonic()-started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(out/'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(error=repr(exc), elapsed_seconds=time.monotonic()-started)
        write_json(out/'manifest.json', manifest)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['pilot', 'full'], required=True)
    p.add_argument('--max-seconds', type=int, default=5400)
    run(p.parse_args())
