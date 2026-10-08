"""Post-score diagnosis of normal high-score and annotated-error queries.

Selection uses evaluation labels for diagnosis only; no score is changed.
Stored native gradients are reused, with zero new backward passes. Contributions
are current-query AV perturbations, not complete historical-source attribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from experiments.decision_risk_flow.run import load_model
from .capture import truncate_native_past
from .mechanism import capture_query, channel_masks, copy_past, prefill_past
from .readout import file_hash


@torch.no_grad()
def key_responses(model, past, query_id, gradient, source_mask, prompt_length):
    cache = copy_past(past)
    position = cache.get_seq_length()
    ids = torch.tensor([[query_id]], device=model.device)
    with capture_query(model) as captured:
        hidden = model.model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
        logits = model.lm_head(hidden[0, 0]).float().cpu()
    queries, rotary, _, _ = captured
    contributions, attentions = [], []
    for layer, block in enumerate(model.model.layers):
        attention = block.self_attn
        heads = attention.q_proj.out_features // attention.head_dim
        query = queries[layer].view(1, 1, heads, attention.head_dim).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(query, query, *rotary[layer])
        keys = repeat_kv(cache.layers[layer].keys, attention.num_key_value_groups)
        values = repeat_kv(cache.layers[layer].values, attention.num_key_value_groups)[0].float()
        weights = (query[0].float() @ keys[0].float().transpose(-1, -2) * attention.scaling).softmax(-1)[:, 0]
        adjoint = gradient[:, layer].to(device=model.device, dtype=torch.float32)
        contribution = torch.einsum('ohd,hkd->ohk', adjoint, values) * weights[None]
        contributions.append(contribution.cpu())
        attentions.append(weights.cpu())
    contribution = torch.stack(contributions, dim=1)
    attention = torch.stack(attentions)
    masks = channel_masks(source_mask, prompt_length, position, 16, 'cpu')
    grouped = torch.einsum('olhk,gk->olgh', contribution, masks.float())
    return contribution, attention, grouped, logits.log_softmax(-1)


def select_queries(root):
    """Freeze diagnostic selection; do not use it as deployment token selection."""
    chosen = {('12219', 223), ('12297', 106)}
    for folder, method in (('scores', 'source_candidate_regret'),
                           ('elasticity_scores', 'source_loss_elasticity')):
        rows = json.loads((root / folder / 'tokens.json').read_text())
        normal = sorted((row for row in rows if row['label'] == 0), key=lambda row: -row['scores'][method])
        chosen.update((row['id'], row['target']) for row in normal[:3])
        for identity in ('12219', '12297', '15604'):
            errors = [row['target'] for row in rows if row['id'] == identity and row['label']]
            chosen.update((identity, errors[index]) for index in (0, len(errors) // 2, len(errors) - 1))
    return sorted(chosen)


def describe_keys(contribution, attention, record, tokenizer, special_ids):
    """Report influential source keys and attention keys independently."""
    tokens = record['prompt'] + record['response']['answer_ids']
    source = np.flatnonzero(record['source_mask'])
    net = contribution[1].sum((0, 1)).numpy()
    mass = attention.mean((0, 1)).numpy()
    keys = [('positive_source', sorted(source, key=lambda key: -net[key])[:5]),
            ('negative_source', sorted(source, key=lambda key: net[key])[:5]),
            ('largest_source_attention', sorted(source, key=lambda key: -mass[key])[:5])]
    described = {}
    for name, positions in keys:
        rows = []
        for key in positions:
            literal = tokenizer.decode([tokens[key]])
            rows.append(dict(position=int(key), token_id=tokens[key], text=literal,
                context=tokenizer.decode(tokens[max(0, key - 8):key + 9]),
                net_margin_response=float(net[key]), average_head_attention=float(mass[key]),
                special_token=tokens[key] in special_ids,
                punctuation_only=bool(literal.strip()) and not any(char.isalnum() for char in literal)))
        described[name] = rows
    return described


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.root / 'complete_roster' / 'INPUTS.json'
    document = json.loads(inputs.read_text())
    records = {record['id']: record for record in document['records']}
    selected = select_queries(args.root)
    (args.output / 'SELECTION.json').write_text(json.dumps(dict(queries=selected,
        purpose='label-guided diagnosis after frozen detection, zero score tuning',
        input_sha256=file_hash(inputs), code_sha256=file_hash(__file__),
        selection_inputs={str(args.root / folder / 'tokens.json'):
            file_hash(args.root / folder / 'tokens.json')
            for folder in ('scores', 'elasticity_scores')}), indent=2) + '\n')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(document['model'], local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(document['model'])
    results, maximum = [], 0.
    for identity in sorted({identity for identity, _ in selected}):
        record = records[identity]
        tokens = record['prompt'] + record['response']['answer_ids']
        cache = prefill_past(model, tokens[:-1])
        with np.load(args.root / 'complete_capture' / identity / 'arrays.npz') as data:
            actual, rivals, logps = data['actual_id'], data['rival_id'], data['actual_logp']
        for _, target in (item for item in selected if item[0] == identity):
            position = record['prompt_length'] + target - 1
            node_path = args.root / 'complete_capture' / identity / 'nodes' / f'{target:06d}.npz'
            with np.load(node_path) as data:
                gradient = torch.from_numpy(data['gradient'].copy())
                expected = torch.from_numpy(data['response'].copy())
            contribution, attention, grouped, native = key_responses(model,
                truncate_native_past(cache, position), tokens[position], gradient,
                record['source_mask'], record['prompt_length'])
            error = float((grouped - expected).abs().max())
            maximum = max(maximum, error)
            assert error <= 2e-3, f'{identity}:{target} key contributions do not reconstruct group response'
            assert abs(float(native[actual[target]]) - float(logps[target])) <= 1e-3
            np.savez_compressed(args.output / f'{identity}_{target}.npz',
                contribution=contribution.numpy(), attention=attention.numpy(), token_ids=np.asarray(tokens[:position + 1]))
            results.append(dict(id=identity, target=target, text=record['response']['token_text'][target],
                native_rival=tokenizer.decode([int(rivals[target])]), group_reconstruction_error=error,
                **describe_keys(contribution, attention, record, tokenizer, special_ids)))
            print(json.dumps(dict(id=identity, target=target, reconstruction_error=error)), flush=True)
        del cache
    (args.output / 'RESULTS.json').write_text(json.dumps(dict(status='DONE', queries=len(results),
        new_backward_calls=0, maximum_group_error=maximum, new_risk_scores=0,
        selection_sha256=file_hash(args.output / 'SELECTION.json'),
        tensor_hashes={path.name: file_hash(path) for path in sorted(args.output.glob('*.npz'))},
        native_node_hashes={f'{identity}:{target}': file_hash(args.root /
            'complete_capture' / identity / 'nodes' / f'{target:06d}.npz')
            for identity, target in selected},
        scope='current-query AV key perturbations; source-reading identity is diagnostic, not semantic truth',
        results=results), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
