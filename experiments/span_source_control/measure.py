"""Label-free full-head readout of normalized source attention sensitivity."""
import argparse
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from experiments.anchored_flow.edges import layer_arrays
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_maintenance.state import FIELDS

MODEL = '/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct'
KEYS = ('15604', '9022', '00005', '00006', '00012', '00013',
        'gsm8k-43', 'gsm8k-49', 'gsm8k-123', 'gsm8k-240', 'gsm8k-243', 'gsm8k-313')
COLUMNS = ('prompt_positive', 'prompt_negative', 'history_positive', 'history_negative',
           'prompt_attention', 'top_effect', 'top_attention_effect')


def centered_effect(attention, gate_effect):
    """dF/d attention-logit, from post-softmax amplitude derivatives [H,T,K]."""
    return gate_effect - attention * gate_effect.sum(-1, keepdims=True)


def lexical_groups(ids, tokenizer):
    """Automatic lexical source addresses; these are not evidence/entity labels."""
    texts = [tokenizer.decode([token]) for token in ids]
    ends = np.cumsum([len(text) for text in texts])
    starts = np.r_[0, ends[:-1]]
    groups = []
    for match in re.finditer(r"\w+(?:[-’']\w+)*", ''.join(texts)):
        overlap = np.minimum(ends, match.end()) - np.maximum(starts, match.start())
        keys = [int(key) for key in np.flatnonzero(overlap > 0)
                if ids[key] not in tokenizer.all_special_ids]
        if keys:
            groups.append(dict(text=match.group(), keys=keys))
    # A token straddling lexical chunks stays in only its first group.
    used = set()
    result = []
    for group in groups:
        group['keys'] = [key for key in group['keys'] if key not in used]
        used.update(group['keys'])
        if group['keys']:
            result.append(group)
    return result


def head_readout(attention, effect, prompt, valid_keys):
    centered = centered_effect(attention, effect)
    positive = np.maximum(centered, 0)
    negative = np.maximum(-centered, 0)
    lexical_effect = centered[..., valid_keys]
    effect_slot = np.argmax(np.abs(lexical_effect), axis=-1)
    attention_slot = np.argmax(attention[..., valid_keys], axis=-1)
    best = np.take_along_axis(lexical_effect, effect_slot[..., None], -1)[..., 0]
    attended = np.take_along_axis(lexical_effect, attention_slot[..., None], -1)[..., 0]
    measured = np.stack((positive[..., :prompt].sum(-1), negative[..., :prompt].sum(-1),
        positive[..., prompt:].sum(-1), negative[..., prompt:].sum(-1),
        attention[..., :prompt].sum(-1), best, attended), -1)
    return measured, valid_keys[effect_slot], valid_keys[attention_slot]


def select_receivers(row, measured, continuity):
    strength = np.max(np.abs(measured[..., COLUMNS.index('top_effect')]), axis=(0, 1))
    lexical = np.array([bool(re.search(r'\w', text)) for text in row['response']['token_text']])
    median = np.median(continuity[lexical])
    selected = []
    for name, mask in [('persistent', continuity >= median), ('changing', continuity < median)]:
        candidates = np.flatnonzero(mask & lexical)
        target = int(candidates[np.argmax(strength[candidates])])
        layer, head = np.unravel_index(np.argmax(np.abs(measured[:, :, target, 5])), (32, 32))
        selected.append(dict(regime=name, target=target, layer=int(layer), head=int(head),
            token=row['response']['token_text'][target], continuation=float(continuity[target])))
    return selected


def group_treatments(attention, effect, groups):
    centered = centered_effect(attention, effect)
    mass = np.array([attention[group['keys']].sum() for group in groups])
    slope = np.array([centered[group['keys']].sum() for group in groups])
    effect_index = int(np.argmax(np.abs(slope)))
    attention_index = int(np.argmax(mass))
    length = len(groups[effect_index]['keys'])
    distance = np.abs(np.log((mass + 1e-9) / (mass[effect_index] + 1e-9)))
    distance += 10 * np.abs(np.array([len(group['keys']) for group in groups]) - length)
    distance[effect_index] = np.inf
    control = int(np.argmin(distance))
    return [dict(kind=name, **groups[index], attention=float(mass[index]), slope=float(slope[index]))
            for name, index in [('effect', effect_index), ('attention', attention_index), ('control', control)]]


def measure_record(row, tokenizer, output):
    directory = output / row['key']
    directory.mkdir(parents=True, exist_ok=True)
    groups = lexical_groups(row['prompt'], tokenizer)
    valid = np.array(sorted({key for group in groups for key in group['keys']}))
    observed, effect_keys, attention_keys = [], [], []
    for layer in range(32):
        attention, effect = layer_arrays(row, layer)
        measured, best, attended = head_readout(attention, effect, len(row['prompt']), valid)
        observed.append(measured)
        effect_keys.append(best)
        attention_keys.append(attended)
    measured = np.array(observed, dtype=np.float32)
    state_path = Path('outputs/span_maintenance_20260929_v1') / row['key'] / 'states.npz'
    with np.load(state_path) as saved:
        state = saved['state']
    continuity = state[..., FIELDS.index('continuation')].mean((0, 1))
    probes = select_receivers(row, measured, continuity)
    for probe in probes:
        attention, effect = layer_arrays(row, probe['layer'])
        probe['treatments'] = group_treatments(attention[probe['head'], probe['target']],
            effect[probe['head'], probe['target']], groups)
    np.savez_compressed(directory / 'heads.npz', measured=measured, effect_keys=effect_keys,
        attention_keys=attention_keys, continuation=state[..., FIELDS.index('continuation')])
    write_json(directory / 'probes.json', probes)
    write_json(directory / 'sources.json', groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json('outputs/span_maintenance_20260929_v1/manifest.json')
    manifest['records'] = [row for row in manifest['records'] if row['key'] in KEYS]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'manifest.json', manifest)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for row in manifest['records']:
        measure_record(row, tokenizer, args.output)
        print('measured', row['key'], flush=True)
    write_json(args.output / 'measured.json', dict(labels_used=False, columns=COLUMNS,
        heads=1024, answers=len(manifest['records']), dose=.05, source_groups='automatic lexical'))


if __name__ == '__main__':
    main()
