"""Describe signed head patterns and native readout precision without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .grounded_projection_data import write_json
from .repetition_analysis import source_bootstrap
from .repetition_teacher import BASE, SelfReadout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = torch.load(BASE / 'prepared.pt', weights_only=False)['test']
    pairs = json.loads((args.output / 'matched_tokens.json').read_text())['strict']
    selection = json.loads((args.output / 'head_selection.json').read_text())
    teacher = SelfReadout(42)
    tangents = np.load(args.output / 'teacher_tangents.npz')
    direction = (tangents['seed42']+tangents['seed123'])/2
    sources, differences = [], []
    for row in rows:
        standardized = teacher.standardized(row).numpy()
        for target, control in pairs[row['id']]:
            sources.append(row['source_id'])
            differences.append(standardized[target]-standardized[control])
    differences = np.asarray(differences)
    profile = []
    for head in selection['important']:
        effect = source_bootstrap(sources, differences[:, head])
        profile.append(dict(layer=head//32, head=head%32, tangent_weight=float(direction[head]),
            paired_delta_z=effect, signed_contribution=float(direction[head]*effect['mean'])))
    precision = []
    directory = args.output / 'native_all_layers'
    cases = json.loads((directory / 'inputs.json').read_text())['cases']
    features = np.load(args.output / 'features.npz')
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    for case in cases:
        old = np.load(directory / (case['id']+'_original_self.npz'))['selected']
        fresh = np.load(directory / case['id'] / 'observations.npz')
        new = np.concatenate([fresh[f'{layer}:self_diagonal'].T for layer in range(32)])
        fresh_probability = []
        for model in teachers:
            standardized = torch.tensor((np.log1p(1000*new.T)-model.mean)/model.scale)
            with torch.no_grad():
                fresh_probability.append(model.logits(standardized).sigmoid().numpy())
        old_probability = features['teacher:'+case['id']][[case['target'], case['control']]]
        precision.append(dict(id=case['id'], max_difference=float(abs(new-old).max()),
            mean_difference=float(abs(new-old).mean()), old_teacher=old_probability.tolist(),
            fresh_teacher=np.mean(fresh_probability, axis=0).tolist()))
    write_json(args.output / 'head_profile.json', dict(heads=profile, native_precision=precision,
        feature='training-standardized log1p(1000*post-token self attention)',
        scope='supervised signed marker, not generative cause; individual CIs exploratory'))
    print(json.dumps(profile[:12], indent=2))


if __name__ == '__main__':
    main()
