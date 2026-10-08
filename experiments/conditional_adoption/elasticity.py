"""One exploratory post-pilot change of objective: log predictive loss.

No new fit, entropy input, label weights or parameter search. This run follows
the first evaluated pilot and must not be presented as its frozen primary.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .readout import file_hash


LOSS_FLOOR = 1e-4


def run(scores, output):
    frozen = json.loads((scores / 'frozen_scores.json').read_text())
    if file_hash(scores / 'scores.npz') != frozen['score_sha256']:
        raise ValueError('input scores changed after freezing')
    output.mkdir(parents=True, exist_ok=False)
    values, records = {}, []
    with np.load(scores / 'scores.npz', allow_pickle=False) as original:
        for record in frozen['records']:
            identity = record['id']
            for name in original.files:
                if name.startswith(identity + '/'):
                    values[name] = original[name].copy()
            loss = original[f'{identity}/native_actual_nll'].astype(np.float64)
            if (loss < -1e-6).any():
                raise ValueError('log probability cannot produce negative NLL')
            denominator = np.maximum(loss, 0) + LOSS_FLOOR
            names = list(record['score_names'])
            if 'source_candidate_regret' in names:
                name = 'source_loss_elasticity'
                values[f'{identity}/{name}'] = original[f'{identity}/source_candidate_regret'] / denominator
                names.append(name)
            if 'regret_both' in names:
                for condition in ('current', 'past', 'both'):
                    name = f'loss_elasticity_{condition}'
                    changed_loss = loss + original[f'{identity}/regret_{condition}']
                    if (changed_loss < -1e-6).any():
                        raise ValueError('finite intervention has invalid predictive loss')
                    values[f'{identity}/{name}'] = np.log(np.maximum(changed_loss, 0) + LOSS_FLOOR) - np.log(denominator)
                    names.append(name)
            records.append(dict(record, score_names=names))
    np.savez_compressed(output / 'scores.npz', **values)
    result = dict(frozen, records=records, primary='source_loss_elasticity' if
        frozen['primary'] == 'source_candidate_regret' else 'loss_elasticity_both',
        exploratory_post_pilot=True, original_primary=frozen['primary'], loss_floor=LOSS_FLOOR,
        score_sha256=file_hash(output / 'scores.npz'), code_hash=file_hash(__file__),
        input_score_sha256=frozen['score_sha256'], natural_label_fit=False,
        formulas={'derivative': '-d logp(y)/d amplitude divided by (NLL(y)+1e-4)',
                  'finite': 'log(NLL_intervened(y)+1e-4)-log(NLL_native(y)+1e-4)'})
    (output / 'frozen_scores.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.scores, args.output)
    print(json.dumps(dict(primary=result['primary'], score_sha256=result['score_sha256'])), flush=True)


if __name__ == '__main__':
    main()
