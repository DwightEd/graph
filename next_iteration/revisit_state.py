"""P3: preserve the existing head-wise revisit selector; test event-held state."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def head_statistics(attention, source_mask, special_mask):
    """One layer, [heads,prediction queries,all keys]. No head selection."""
    a = np.asarray(attention)
    p, n = len(source_mask), a.shape[1]
    if a.ndim != 3 or a.shape[2] != p+n or len(special_mask) != p+n:
        raise ValueError('attention alignment')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('invalid attention')
    shift, mass = np.zeros((n, a.shape[0])), np.zeros((n, a.shape[0]))
    allowed = ~np.asarray(special_mask, dtype=bool).copy()
    allowed[:p] &= np.asarray(source_mask, dtype=bool)
    for t in range(n):
        q = p+t-1
        if np.count_nonzero(a[:, t, q+1:]):
            raise ValueError('visible future keys')
        if not np.allclose(a[:, t, :q+1].sum(-1), 1, atol=.005, rtol=0):
            raise ValueError('invalid row mass')
        if not t:
            continue
        # Keys strictly before old query: exclude BOTH current and previous self.
        current = a[:, t, :q-1].astype(np.float64) * allowed[:q-1]
        old = a[:, t-1, :q-1].astype(np.float64) * allowed[:q-1]
        cm, om = current.sum(-1), old.sum(-1)
        current = np.divide(current, cm[:, None], out=np.zeros_like(current), where=cm[:, None]>0)
        old = np.divide(old, om[:, None], out=np.zeros_like(old), where=om[:, None]>0)
        mass[t] = np.minimum(cm, om)
        shift[t] = np.abs(np.cumsum(current-old, axis=-1)).sum(-1)
        shift[t, mass[t] == 0] = 0
    return shift, mass


def causal_events(values, window=16, quantile=.95):
    values = np.asarray(values, dtype=float)
    threshold = np.full(len(values), np.nan)
    active = np.zeros(len(values), bool)
    for t in range(len(values)):
        old = values[max(0, t-window):t]
        old = old[np.isfinite(old)]
        if len(old):
            threshold[t] = np.quantile(old, quantile)
        active[t] = t > window and values[t] > threshold[t] + 1e-10
    event = active & ~np.r_[False, active[:-1]]
    return dict(threshold=threshold, active=active, event=event)


def summarize_events(shifts, masses, window=16, quantile=.95):
    # [layers,steps,heads], as saved by head_statistics.
    if shifts.shape != masses.shape or shifts.ndim != 3:
        raise ValueError('head statistics alignment')
    n = shifts.shape[1]
    revisit = np.full(n, np.nan)
    for t in range(2, n):
        excess = np.maximum(0., shifts[:, t] - np.median(shifts[:, max(1,t-window):t], axis=1))
        revisit[t] = (excess*masses[:, t]).mean() if masses[:, t].any() else np.nan
    return dict(revisit=revisit, **causal_events(revisit, window, quantile))


def hold_state(entropy, events):
    h, event = np.asarray(entropy, float), np.asarray(events, bool)
    if h.ndim != 1 or h.shape != event.shape or not np.isfinite(h).all():
        raise ValueError('state alignment')
    score, state = h.copy(), None
    for t in range(len(h)):
        if event[t]:
            state = h[t]
        if state is not None:
            score[t] = state
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--reference-predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['pilot', 'development'], required=True)
    args = p.parse_args()
    from next_iteration.grounding_contrast import INPUT_SHA, validate_rows, digest, write_json
    if digest(args.inputs) != INPUT_SHA:
        raise ValueError('input SHA mismatch')
    all_rows = [json.loads(s) for s in Path(args.inputs).read_text().splitlines()]
    validate_rows(all_rows)
    rows = [r for r in all_rows if (str(r['id']) in ['17019','17020'] if args.phase=='pilot' else r['split']=='development')]
    if len(rows) != (2 if args.phase=='pilot' else 32):
        raise ValueError('frozen roster mismatch')
    reference_root = Path(args.reference_predictions)
    ref_manifest = json.loads((reference_root/'manifest.json').read_text())
    if not ref_manifest['complete'] or ref_manifest.get('error'):
        raise ValueError('incomplete reference')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(complete=False, settings=vars(args), input_sha256=INPUT_SHA,
        code_sha256=digest(__file__), planned_ids=[str(r['id']) for r in rows], completed_ids=[],
        started_unix=started, primary_score='event_hold_entropy', labels_used=False,
        reference_manifest_sha256=digest(reference_root/'manifest.json'),
        window=16, quantile=.95, timing='q=P+t-1; all scores causal pre-token except native_nll',
        scope='development; Llama observer; no original-generator or binding mechanism claim')
    (out/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    protocol = Path(__file__).resolve().parent.parent/'docs/P3_REVISIT_PROTOCOL.md'
    (out/'protocol.md').write_bytes(protocol.read_bytes())
    manifest['protocol_sha256'] = digest(protocol)
    write_json(out/'manifest.json', manifest)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.llama import modeling_llama
        torch.set_num_threads(4)
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
        if model.config.model_type != 'llama':
            raise ValueError('unsupported model')
        manifest['versions'] = dict(torch=torch.__version__, transformers=transformers.__version__)
        original = modeling_llama.eager_attention_forward
        capture = {}

        def hook(module, query, key, value, attention_mask, **kwargs):
            output, a = original(module, query, key, value, attention_mask, **kwargs)
            rows_a = a[0, :, capture['positions'], :].float()
            capture['sum'].add_(rows_a.mean(0))
            shift, mass = head_statistics(rows_a.cpu().numpy(), capture['source'], capture['special'])
            capture['shift'].append(shift)
            capture['mass'].append(mass)
            capture['layers'].append(module.layer_idx)
            return output, None

        modeling_llama.eager_attention_forward = hook
        try:
            for row in rows:
                rid, P, total = str(row['id']), row['prompt_length'], len(row['token_ids'])
                n, positions = total-P, row['prediction_query_positions']
                if positions != list(range(P-1, total-1)) or len(row['offsets']) != n:
                    raise ValueError('causal query alignment')
                source = np.array(row['source_mask'], bool)
                special = np.isin(row['token_ids'], tok.all_special_ids)
                capture.update(positions=positions, source=source, special=special,
                    sum=torch.zeros(n,total,device='cuda'), shift=[],mass=[],layers=[])
                ids = torch.tensor([row['token_ids']],device='cuda')
                controls = dict(native_entropy=[], native_nll=[], native_negative_margin=[])
                with torch.inference_mode():
                    hidden = model.model(input_ids=ids,use_cache=False,output_attentions=False).last_hidden_state[0]
                    for t in range(0,n,32):
                        z = model.lm_head(hidden[positions[t:t+32]]).float()
                        lp = torch.log_softmax(z,-1)
                        top = z.topk(2,dim=-1).values
                        controls['native_entropy'].extend((-(lp.exp()*lp).sum(-1)).cpu().tolist())
                        controls['native_nll'].extend((-lp.gather(1,ids[0,P+t:P+t+len(z),None])[:,0]).cpu().tolist())
                        controls['native_negative_margin'].extend((top[:,1]-top[:,0]).cpu().tolist())
                if capture['layers'] != list(range(model.config.num_hidden_layers)):
                    raise ValueError('missing layers')
                shift, mass = np.stack(capture['shift']),np.stack(capture['mass'])
                events = summarize_events(shift,mass)
                mean = (capture['sum']/len(capture['layers'])).cpu().numpy()
                ms, mm = head_statistics(mean[None],source,special)
                mean_events = summarize_events(ms[None],mm[None])
                h = np.array(controls['native_entropy'])
                entropy_events = causal_events(h)['event']
                scores = {k:np.array(v) for k,v in controls.items()}
                scores.update(event_hold_entropy=hold_state(h,events['event']),
                    entropy_event_hold=hold_state(h,entropy_events),
                    periodic_hold_entropy=hold_state(h,np.arange(n)%16==0),
                    revisit_magnitude=np.nan_to_num(events['revisit']),
                    mean_revisit_magnitude=np.nan_to_num(mean_events['revisit']))
                ref_path = reference_root/f'response_{rid}.json'
                if digest(ref_path) != ref_manifest['output_sha256'][ref_path.name]:
                    raise ValueError('reference hash mismatch')
                ref = json.loads(ref_path.read_text())
                if ref['token_ids'] != row['token_ids'][P:] or ref['response_sha256'] != row['response_sha256']:
                    raise ValueError('reference identity mismatch')
                deltas = {k:float(np.max(np.abs(np.array(ref['scores'][k])-v))) for k,v in controls.items()}
                if max(deltas.values()) > 1e-5:
                    raise ValueError(f'native output changed from same eager reference: {deltas}')
                graph = out/f'attention_{rid}.npz'
                np.savez_compressed(graph,mean_attention=mean,source_mask=source,special_mask=special,
                    full_token_ids=np.array(row['token_ids']),prediction_positions=np.array(positions),
                    head_shift=shift,head_common_mass=mass,revisit=events['revisit'],
                    threshold=events['threshold'],active=events['active'],event=events['event'],
                    entropy_event=entropy_events,mean_event=mean_events['event'])
                result = {k:row[k] for k in ('id','source_id','task','split','response_sha256','prompt_sha256','offsets')}
                result.update(token_ids=row['token_ids'][P:],scores={k:v.tolist() for k,v in scores.items()},
                    event=events['event'].tolist(),entropy_event=entropy_events.tolist(),mean_event=mean_events['event'].tolist(),
                    native_vs_p2_max_abs=deltas,graph_sha256=digest(graph))
                write_json(out/f'response_{rid}.json',result)
                manifest['completed_ids'].append(rid)
                manifest['elapsed_seconds'] = time.time()-started
                write_json(out/'manifest.json',manifest)
                print(json.dumps(dict(id=rid,completed=len(manifest['completed_ids']),events=int(events['event'].sum()),elapsed=manifest['elapsed_seconds'])),flush=True)
                del hidden
        finally:
            modeling_llama.eager_attention_forward = original
        manifest.update(complete=True,elapsed_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            output_sha256={p.name:digest(p) for p in sorted(out.glob('response_*.json'))},
            graph_sha256={p.name:digest(p) for p in sorted(out.glob('attention_*.npz'))})
        write_json(out/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(error=repr(exc),elapsed_seconds=time.time()-started)
        write_json(out/'manifest.json',manifest)
        raise


if __name__ == '__main__':
    main()
