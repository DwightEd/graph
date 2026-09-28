"""Annotation-free signed writes on complete A/V/residual caches, on CPU.

Freeze final RMS scaling and project every message on sampled-token versus native
best-other-token logit margin. This is direct logit attribution, not a derivative
through later layers or a test of factual support.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
import torch

from .features import js_divergence, normalize
from .run import load_readout_weights, write_json

FIELDS = ('positive', 'negative_magnitude', 'net', 'absolute', 'attention')
GROUPS = ('prompt', 'history', 'special')


def project_messages(attention, values, output_weight, direction):
    """Return [head,target,key] direct output contributions with GQA expanded."""
    heads = attention.shape[0]
    grouped = np.repeat(values, heads // values.shape[1], axis=1)
    pulled = (direction @ output_weight).reshape(len(direction), heads, -1)
    payload = np.einsum('thd,khd->htk', pulled, grouped, optimize=True)
    return attention * payload


def signed_stats(contribution, attention, masks):
    rows = []
    for mask in masks:
        value = contribution[:, :, mask]
        rows.append(np.stack((np.maximum(value, 0).sum(-1),
            np.maximum(-value, 0).sum(-1), value.sum(-1),
            np.abs(value).sum(-1), attention[:, :, mask].sum(-1)), -1))
    return np.stack(rows, -2)


def output_direction(graph, weights, norm, epsilon):
    prompt = int(graph['prompt_length'])
    targets = graph['target_ids']
    top = graph['top_ids']
    alternatives = np.where(top[:, 0] == targets, top[:, 1], top[:, 0])
    residual = graph['residual'][-1, prompt - 1:]
    rms = np.sqrt(np.mean(residual ** 2, axis=-1, keepdims=True) + epsilon)
    # Subtract in float32, preserving each saved bfloat16 weight before subtraction.
    delta = weights[torch.tensor(targets)].float().numpy() - weights[torch.tensor(alternatives)].float().numpy()
    return delta * norm / rms, alternatives


def measure_graph(directory, model, weights, norm, epsilon, special_ids, output):
    names = ('residual', 'post_attention', 'mlp_update', 'attention', 'values',
             'target_ids', 'input_ids', 'top_ids', 'top_logits', 'prompt_length')
    graph = {name: np.load(directory / (name + '.npy'), mmap_mode='r') for name in names}
    prompt = int(graph['prompt_length'])
    direction, alternatives = output_direction(graph, weights, norm, epsilon)
    keys = np.arange(len(graph['input_ids']))
    special = np.isin(graph['input_ids'], special_ids)
    masks = ((keys < prompt) & ~special, (keys >= prompt) & ~special, special)
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    stats, top_keys, top_scores, rows = [], [], [], []
    embedding_margin = np.einsum('td,td->t', direction, graph['residual'][0, prompt - 1:])
    cumulative = embedding_margin.copy()
    for layer in range(32):
        key = f'model.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(model / index[key], framework='pt', device='cpu') as saved:
            output_weight = saved.get_tensor(key).float().numpy()
        attention = np.asarray(graph['attention'][layer, :, prompt - 1:])
        contribution = project_messages(attention, graph['values'][layer], output_weight, direction)
        summary = signed_stats(contribution, attention, masks)
        stats.append(summary)
        strongest = np.argsort(np.abs(contribution), axis=-1)[..., -4:][..., ::-1]
        top_keys.append(strongest.astype(np.int32))
        top_scores.append(np.take_along_axis(contribution, strongest, axis=-1))
        after = graph['post_attention'][layer, prompt - 1:]
        before = graph['residual'][layer, prompt - 1:]
        next_state = graph['residual'][layer + 1, prompt - 1:]
        mlp = np.einsum('td,td->t', direction, graph['mlp_update'][layer, prompt - 1:])
        attention_direct = contribution.sum((0, 2))
        actual_attention = np.einsum('td,td->t', direction, after - before)
        actual_block = np.einsum('td,td->t', direction, next_state - before)
        cumulative += attention_direct + mlp
        address_gap = js_divergence(normalize(attention[:, :, ~special]),
                                    normalize(np.abs(contribution[:, :, ~special]))).mean(0)
        append_layer_rows(rows, layer, summary, direction, next_state, mlp,
                          attention_direct, actual_attention, actual_block, cumulative, address_gap)
        print('messages', directory.name, 'layer', layer + 1, flush=True)
    np.savez_compressed(output / (directory.name + '.npz'), stats=np.stack(stats),
        top_keys=np.stack(top_keys), top_scores=np.stack(top_scores),
        targets=graph['target_ids'], alternatives=alternatives, embedding_margin=embedding_margin)
    pd.DataFrame(rows).to_csv(output / (directory.name + '.csv'), index=False)


def append_layer_rows(rows, layer, summary, direction, next_state, mlp,
                      direct, actual_attention, actual_block, cumulative, address_gap):
    for position in range(len(direction)):
        row = dict(layer=layer + 1, position=position, mlp_direct=mlp[position],
            attention_direct=direct[position], attention_roundoff=actual_attention[position] - direct[position],
            block_roundoff=actual_block[position] - direct[position] - mlp[position],
            cumulative_direct=cumulative[position],
            frozen_scale_state_margin=np.dot(direction[position], next_state[position]),
            read_write_address_js=address_gap[position])
        for group, group_name in enumerate(GROUPS):
            for field, field_name in enumerate(FIELDS):
                row[group_name + '_' + field_name] = summary[:, position, group, field].sum()
        rows.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weights, norm, epsilon = load_readout_weights(args.model)
    tokenizer = json.loads((args.model / 'tokenizer_config.json').read_text())
    special_ids = [int(key) for key, value in tokenizer['added_tokens_decoder'].items() if value['special']]
    write_json(args.output / 'protocol.json', dict(schema='role-free-direct-writes-v1',
        graphs=str(args.graphs.resolve()), fields=FIELDS, groups=GROUPS,
        projection='actual token minus highest native alternative; frozen final RMS scale',
        semantics='direct logit attribution; not factual support or downstream causal effect',
        provenance='existing complete-forward caches on original natural texts; NOT exact generation capture',
        forbidden_inputs=['roles.npy', 'source_mask', 'evidence', 'correct_candidate', 'error_span'],
        labels_used=False, selection='all tokens of both available full A/V graphs'))
    for directory in sorted(args.graphs.iterdir()):
        if directory.is_dir() and (directory / 'values.npy').is_file():
            measure_graph(directory, args.model, weights, norm, epsilon, special_ids, args.output)
    write_json(args.output / 'complete.json', dict(status='complete', new_model_forwards=0))


if __name__ == '__main__':
    main()
