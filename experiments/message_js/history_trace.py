"""Finite native history formation: edit one past message, recompute future states."""

from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


@contextmanager
def message_gate(model, layer, head, query_position, key_position, scale):
    module = model.model.layers[layer].self_attn
    original = module.forward

    def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
        output, weights = original(hidden_states, position_embeddings, attention_mask, **kwargs)
        cosine, sine = position_embeddings
        query = module.q_proj(hidden_states[:, query_position:query_position+1])
        key = module.k_proj(hidden_states[:, :query_position+1])
        value = module.v_proj(hidden_states[:, key_position:key_position+1])
        query = query.view(1, 1, -1, module.head_dim).transpose(1, 2)
        key = key.view(1, query_position+1, -1, module.head_dim).transpose(1, 2)
        value = value.view(1, 1, -1, module.head_dim).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(query, query, cosine[:, query_position:query_position+1], sine[:, query_position:query_position+1])
        key, _ = apply_rotary_pos_emb(key, key, cosine[:, :query_position+1], sine[:, :query_position+1])
        key, value = [repeat_kv(x, module.num_key_value_groups) for x in (key, value)]
        attention = (query.float() @ key.float().transpose(-1, -2) * module.scaling).softmax(-1)
        projection = module.o_proj.weight.reshape(output.shape[-1], query.shape[1], module.head_dim)[:, head]
        message = projection.float() @ value[0, head, 0].float()
        message = attention[0, head, 0, key_position] * message
        changed = output.clone()
        changed[:, query_position] += (scale - 1) * message
        return changed, weights

    with patch.object(module, 'forward', MethodType(forward, module)):
        yield


@torch.no_grad()
def full_trace(model, tokens, layer, head, query, key, epsilon=.01):
    ids = torch.tensor([tokens], device=model.device)
    baseline = model.model(ids, use_cache=False).last_hidden_state[0]
    changed = []
    for scale in (1 - epsilon, 1 + epsilon):
        with message_gate(model, layer, head, query, key, scale):
            changed.append(model.model(ids, use_cache=False).last_hidden_state[0])
    return baseline, (changed[1] - changed[0]) / (2 * epsilon), changed


@torch.no_grad()
def output_js(model, changed, prompt):
    count = len(changed[0]) - prompt + 1
    rows = []
    for start in range(0, count, 16):
        region = slice(prompt - 1 + start, prompt - 1 + min(start + 16, count))
        logp = [model.lm_head(value[region]).double().log_softmax(-1) for value in changed]
        mixture = torch.logaddexp(logp[0], logp[1]) - np.log(2.)
        rows.append((.5 * sum((value.exp() * (value - mixture)).sum(-1) for value in logp)).cpu().numpy())
    return np.concatenate(rows)


def main():
    from experiments.decision_risk_flow.data import read_json, write_json
    from experiments.decision_risk_flow.run import load_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('outputs/message_js_20260928_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.base / 'manifest.json')
    records = [row for row in manifest['records'] if row['role'] == 'natural']
    samples = Path(manifest['samples'])
    write_json(args.output / 'protocol.json', dict(answers=len(records), labels_read=False,
        selection='largest q90 influence-root-JS from token 2 to first half; strongest absolute prompt margin edge at middle layer; no special keys',
        intervention='one native post-softmax message gate +/- .01; all future states recomputed',
        caveat='original text teacher-forced, no resampling; finite effect, not discrete generation derivative'))
    model = load_model(args.model)
    rows = []
    for record in records:
        with np.load(samples / record['trace']) as trace:
            prompt = int(trace['prompt_length'])
            tokens = trace['token_ids'][:-1].tolist()
            special = trace['special_mask'][:prompt]
        with np.load(args.base / record['key'] / 'readouts.npz') as saved:
            count = len(saved['token_ids'])
            events = np.nanquantile(saved['measured'][:, :, :, 1].reshape(1024, count), .9, axis=0)
            event = 2 + int(np.nanargmax(events[2:max(3, count // 2)]))
            actual = int(saved['token_ids'][event])
            other = int(saved['alternative'][event])
        layer = len(model.model.layers) // 2
        derivative = np.load(args.base / record['key'] / 'derivative.npy', mmap_mode='r')
        magnitude = np.abs(derivative[layer, :, event, :prompt]).copy()
        magnitude[:, special] = -1
        head, key = np.unravel_index(magnitude.argmax(), magnitude.shape)
        expected = float(derivative[layer, head, event, key])
        query = prompt - 1 + event
        baseline, tangent, changed = full_trace(model, tokens, layer, int(head), query, int(key))
        assert float(tangent[:query].abs().max()) == 0., 'Past states changed under a future message gate'
        with torch.no_grad():
            response = model.lm_head(tangent[query])
            observed = float(response[actual] - response[other])
        relative = abs(observed - expected) / max(abs(expected), 1e-8)
        if relative > .05:
            raise ValueError(f'Full history/current query derivative mismatch: {record["key"]} {relative}')
        js = output_js(model, changed, prompt)
        np.savez_compressed(args.output / (record['key'] + '.npz'),
            baseline=baseline[prompt-1:].cpu().numpy(), tangent=tangent[prompt-1:].cpu().numpy(), output_js_nats=js)
        rows.append(dict(key=record['key'], source_id=record['source_id'], event=event, layer=layer,
            head=int(head), prompt_key=int(key), expected_margin_derivative=expected,
            observed_margin_derivative=observed, relative_error=relative,
            current_state_effect=float(tangent[query].norm()),
            future_state_effect=float(tangent[query+1:].norm()), future_tokens=len(tokens)-query-1))
        print(rows[-1], flush=True)
    write_json(args.output / 'complete.json', dict(status='complete', rows=rows,
        labels_read=False, score_or_detector_fit=False))


if __name__ == '__main__':
    main()
