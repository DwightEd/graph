"""P2: fixed causal uncertainty transport, without detector fitting or labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def transport(entropy, generated_attention, source_attention):
    h = np.asarray(entropy, dtype=float)
    a = np.asarray(generated_attention, dtype=float)
    s = np.asarray(source_attention, dtype=float)
    n = len(h)
    if a.shape != (n, n) or s.shape != (n,) or not all(np.isfinite(v).all() for v in (h, a, s)):
        raise ValueError('invalid aligned arrays')
    if np.any(a < 0) or np.any(s < 0) or np.any(h < 0):
        raise ValueError('negative input')
    if np.any(np.triu(a) > 1e-8):
        raise ValueError('future or current target edges forbidden')
    risk = np.zeros(n)
    chain, uniform, reverse = np.zeros(n), np.zeros(n), np.zeros(n)
    mass = np.zeros(n)
    for t in range(n):
        denom = s[t] + a[t, :t].sum()
        weights = a[t, :t] / denom if denom > 0 else np.zeros(t)
        mass[t] = weights.sum()
        innovation = (1. - mass[t]) * h[t]
        risk[t] = innovation + weights @ risk[:t]
        chain[t] = innovation + (mass[t] * chain[t - 1] if t else 0.)
        uniform[t] = innovation + (mass[t] * uniform[:t].mean() if t else 0.)
        reverse[t] = innovation + weights[::-1] @ reverse[:t]
    return dict(risk_transport=risk, mass_matched_chain=chain, mass_matched_uniform=uniform,
                mass_matched_reverse=reverse, generated_evidence_mass=mass)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--reference-predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['pilot', 'development', 'validation'], required=True)
    args = p.parse_args()
    from next_iteration.grounding_contrast import INPUT_SHA, validate_rows, digest, write_json, focus_prefix
    if digest(args.inputs) != INPUT_SHA:
        raise ValueError('input SHA mismatch')
    all_rows = [json.loads(x) for x in Path(args.inputs).read_text().splitlines()]
    validate_rows(all_rows)
    rows = [r for r in all_rows if (str(r['id']) in ['17019', '17020'] if args.phase == 'pilot' else r['split'] == args.phase)]
    if len(rows) != (2 if args.phase == 'pilot' else 32):
        raise ValueError('wrong frozen roster count')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(complete=False, settings=vars(args), input_sha256=INPUT_SHA,
                    code_sha256=digest(__file__), helper_code_sha256=digest(Path(__file__).with_name('grounding_contrast.py')),
                    planned_ids=[str(r['id']) for r in rows], completed_ids=[], started_unix=started,
                    primary_score='risk_transport', stochastic=False,
                    provenance='fixed label-free entropy transport on mean native attention; observer not original generator',
                    mechanism_claim=False, development_history='P2 designed after P1 development results; not blind model selection')
    (out / 'executed_code.py').write_bytes(Path(__file__).read_bytes())
    (out / 'executed_helpers.py').write_bytes(Path(__file__).with_name('grounding_contrast.py').read_bytes())
    write_json(out / 'manifest.json', manifest)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.llama import modeling_llama
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation='eager').to('cuda').eval()
        if model.config.model_type != 'llama':
            raise ValueError('only audited Llama attention implementation is supported')
        original_attention = modeling_llama.eager_attention_forward
        capture = {}

        def capture_attention(module, query, key, value, attention_mask, **kwargs):
            output, weights = original_attention(module, query, key, value, attention_mask, **kwargs)
            # Consume the actual weights used for A@V, then free per-head/layer tensors.
            mean = weights[0, :, capture['positions'], :].float().mean(0)
            capture['sum'].add_(mean)
            capture['layers'].append(module.layer_idx)
            return output, None

        modeling_llama.eager_attention_forward = capture_attention
        manifest['versions'] = dict(torch=torch.__version__, transformers=__import__('transformers').__version__)
        try:
            for row in rows:
                item_started = time.time()
                P, total = row['prompt_length'], len(row['token_ids'])
                positions = row['prediction_query_positions']
                n = total - P
                if positions != list(range(P - 1, total - 1)) or len(row['offsets']) != n:
                    raise ValueError('causal alignment mismatch')
                capture.update(positions=positions, sum=torch.zeros(n, total, device='cuda'), layers=[])
                ids = torch.tensor([row['token_ids']], device='cuda')
                controls = dict(native_nll=[], native_entropy=[], native_negative_margin=[])
                with torch.inference_mode():
                    hidden = model.model(input_ids=ids, use_cache=False, output_attentions=False).last_hidden_state[0]
                    for start in range(0, n, 32):
                        z = model.lm_head(hidden[positions[start:start + 32]]).float()
                        lp = torch.log_softmax(z, -1)
                        target = ids[0, P + start:P + start + len(z)]
                        controls['native_nll'].extend((-lp.gather(1, target[:, None])[:, 0]).cpu().tolist())
                        controls['native_entropy'].extend((-(lp.exp() * lp).sum(-1)).cpu().tolist())
                        top = z.topk(2, dim=-1).values
                        controls['native_negative_margin'].extend((top[:, 1] - top[:, 0]).cpu().tolist())
                if capture['layers'] != list(range(model.config.num_hidden_layers)):
                    raise ValueError('missing or reordered layers')
                mean_attention = (capture['sum'] / len(capture['layers'])).cpu().numpy()
                source_mask = np.array(row['source_mask'], dtype=bool)
                if len(source_mask) != P or not source_mask.any():
                    raise ValueError('invalid source mask')
                source_mass = mean_attention[:, :P][:, source_mask].sum(-1)
                generated = mean_attention[:, P:]
                scores = transport(controls['native_entropy'], generated, source_mass)
                mass = scores.pop('generated_evidence_mass')
                scores.update({k: np.array(v) for k, v in controls.items()})
                # Fixed simple persistence control, strictly current-prefix only.
                h = scores['native_entropy']
                clause_max = np.zeros(n)
                previous_boundary, running = None, 0.
                response_ids = row['token_ids'][P:]
                for t in range(n):
                    prefix = tokenizer.decode(response_ids[:t + 1], skip_special_tokens=False)
                    boundary = len(focus_prefix(prefix)[0])
                    if boundary != previous_boundary:
                        running = 0.; previous_boundary = boundary
                    running = max(running, h[t]); clause_max[t] = running
                scores['causal_clause_max_entropy'] = clause_max
                reference = Path(args.reference_predictions) / f"response_{row['id']}.json"
                agreement = None
                if reference.exists():
                    prior = json.loads(reference.read_text())
                    if prior['token_ids'] != response_ids or prior['response_sha256'] != row['response_sha256']:
                        raise ValueError('reference mismatch')
                    agreement = {key: float(np.max(np.abs(np.asarray(prior['scores'][key]) - scores[key]))) for key in controls}
                graph_path = out / f"attention_{row['id']}.npz"
                np.savez_compressed(graph_path, mean_attention=mean_attention, source_mask=source_mask,
                                    generated_evidence_mass=mass, prediction_positions=np.asarray(positions),
                                    full_token_ids=np.asarray(row['token_ids']))
                if not all(np.isfinite(v).all() and len(v) == n for v in scores.values()):
                    raise ValueError('invalid output scores')
                result = dict(id=row['id'], source_id=row['source_id'], task=row['task'], split=row['split'],
                              response_sha256=row['response_sha256'], prompt_sha256=row['prompt_sha256'],
                              token_ids=response_ids, offsets=row['offsets'], scores={k: v.tolist() for k, v in scores.items()},
                              eager_vs_reference_max_abs=agreement, source_mass=source_mass.tolist(),
                              generated_evidence_mass=mass.tolist(), graph_sha256=digest(graph_path),
                              attention_row_sum_max_error=float(np.max(np.abs(mean_attention.sum(-1) - 1))),
                              elapsed_seconds=time.time() - item_started)
                write_json(out / f"response_{row['id']}.json", result)
                manifest['completed_ids'].append(str(row['id']))
                manifest['elapsed_seconds'] = time.time() - started
                write_json(out / 'manifest.json', manifest)
                print(json.dumps(dict(id=row['id'], completed=len(manifest['completed_ids']), elapsed=manifest['elapsed_seconds'],
                                      reference_delta=agreement)), flush=True)
                del hidden
        finally:
            modeling_llama.eager_attention_forward = original_attention
        manifest.update(complete=True, elapsed_seconds=time.time() - started,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        output_sha256={p.name: digest(p) for p in sorted(out.glob('response_*.json'))},
                        graph_sha256={p.name: digest(p) for p in sorted(out.glob('attention_*.npz'))})
        write_json(out / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(complete=False, error=repr(exc), elapsed_seconds=time.time() - started)
        write_json(out / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
