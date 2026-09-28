"""Compare native source relations with earlier generated-choice origins.

Products are descriptive same-head path comparisons, not native cross-layer
Jacobian propagation, semantic evidence identification, or factuality proofs.
"""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json

FIELDS = ('read_direct_js', 'read_relation_js', 'read_state_js', 'read_shuffled_js',
          'use_direct_js', 'use_relation_js', 'use_state_js', 'use_shuffled_js',
          'read_relation_excess', 'use_relation_excess', 'source_mass', 'history_mass')


def normalize(value):
    return value / value.sum(-1, keepdim=True).clamp_min(1e-20)


def js(left, right):
    valid = (left.sum(-1) > 1e-20) & (right.sum(-1) > 1e-20)
    left, right = normalize(left), normalize(right)
    middle = .5 * (left + right)
    def kl(value):
        return (value * (value.clamp_min(1e-30).log() - middle.clamp_min(1e-30).log())).sum(-1)
    return torch.where(valid, .5*(kl(left)+kl(right)), torch.nan)


def source_relation(query, key, positions):
    """Native conditional prompt attention, strict earlier source keys only."""
    key = key.repeat_interleave(query.shape[0] // key.shape[0], dim=0)
    logits = query @ key.transpose(-1, -2) / query.shape[-1] ** .5
    visible = positions[None, :] < positions[:, None]
    logits = logits.masked_fill(~visible, -torch.inf)
    return torch.nan_to_num(logits.softmax(-1), nan=0.)


def compare_paths(direct, history, relation):
    """History key s maps to prediction row s, always strictly before t.

    State-row s+1 is a separate control, and can contain row t.
    """
    direct, history = normalize(direct), normalize(history)
    relay = history @ direct[:, :-1]
    state = history @ direct[:, 1:]
    context = direct @ relation
    return js(direct, relay), js(context, relay), js(context, state)


@torch.no_grad()
def measure(output, device, direction='backward'):
    manifest = read_json(output / 'manifest.json')
    base = Path(manifest['base'])
    started = perf_counter()
    for row in manifest['records']:
        directory = output / row['key']
        directory.mkdir()
        source = output / 'sources' / str(row['source_id'])
        inp = np.load(source / 'input.npz')
        positions = np.flatnonzero(inp['source_mask'])
        length = len(inp['token_ids'])
        q = np.load(source / 'query.npy', mmap_mode='r')
        k = np.load(source / 'key.npy', mmap_mode='r')
        attn = np.load(base / row['key'] / 'attention.npy', mmap_mode='r')
        grad = np.load(base / row['key'] / 'derivative.npy', mmap_mode='r')
        count = attn.shape[2]
        assert attn.shape[-1] == length+count-1
        results = np.empty((32, 32, count, len(FIELDS)), dtype=np.float32)
        pos = torch.tensor(positions, device=device)
        # Deterministic key-identity permutation, followed by causal remasking.
        # This changes lag statistics too; it is an identity-scramble control,
        # not a perfectly matched native intervention.
        perm = torch.tensor(np.random.default_rng(42).permutation(len(pos)), device=device)
        visible = pos[None, :] < pos[:, None]
        for layer in range(32):
            query = torch.tensor(np.asarray(q[layer])[:, positions], device=device)
            key = torch.tensor(np.asarray(k[layer])[:, positions], device=device)
            relation = source_relation(query, key, pos)
            if direction=='forward':
                # Later prompt states read earlier keys. Reversing this graph
                # exposes suffix associations; it is NOT native causal flow.
                relation = normalize(relation.transpose(-1,-2))
            allowed = visible if direction=='backward' else visible.T
            shuffled = normalize(relation[..., perm] * allowed)
            fields = []
            for raw in (attn, grad):
                value = torch.tensor(np.array(raw[layer]), device=device).abs()
                direct = value[..., positions]
                history = value[..., length:]
                # The captured causal matrices must not read current/future choices.
                invalid = torch.arange(count-1, device=device)[None, :] >= torch.arange(count, device=device)[:, None]
                assert not torch.any(history[:, invalid] != 0), row['key']
                direct_js, relation_js, state_js = compare_paths(direct, history, relation)
                relay = normalize(history) @ normalize(direct)[:, :-1]
                shuffled_js = js(normalize(direct) @ shuffled, relay)
                fields.extend((direct_js, relation_js, state_js, shuffled_js))
            a = torch.tensor(np.array(attn[layer]), device=device)
            fields.extend((fields[1]-fields[0], fields[5]-fields[4],
                           a[..., positions].sum(-1), a[..., length:].sum(-1)))
            results[layer] = torch.stack(fields, -1).cpu().numpy()
        np.savez_compressed(directory / 'relations.npz', values=results, fields=FIELDS,
                            source_positions=positions, prompt_length=length)
        print(dict(key=row['key'], tokens=count, seconds=perf_counter()-started), flush=True)
    write_json(output / 'features_complete.json', dict(status='complete', fields=FIELDS,
        axes=['layer', 'physical_head', 'prediction_token', 'field'], labels_used=False,
        direction=direction, answers=len(manifest['records']), seconds=perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--direction',choices=('backward','forward'),default='backward')
    parser.add_argument('--source-cache',type=Path)
    args = parser.parse_args()
    if args.source_cache is not None:
        args.output.mkdir(exist_ok=False)
        manifest = read_json(args.source_cache/'manifest.json')
        manifest.update(measurement='source_relation_'+args.direction,relation_base=str(args.source_cache.resolve()))
        write_json(args.output/'manifest.json',manifest)
        for name in ('sources','capture_complete.json'):
            (args.output/name).symlink_to((args.source_cache/name).resolve())
    read_json(args.output / 'capture_complete.json')
    measure(args.output, args.device,args.direction)


if __name__ == '__main__':
    main()
