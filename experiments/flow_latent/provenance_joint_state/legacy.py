"""Extract existing fixed source/route scores; never access training labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.flow_latent.data import digest_files
from experiments.token_backtrace.grounded_projection_data import write_json


PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')
LEGACY = Path('outputs/unsupervised_graph_20260928')


def extract(output, sources_only=False):
    inputs = json.loads((output / 'inputs.json').read_text())
    values, source_values, paths = {}, {}, []
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test'):
            metadata_path = PACKS / f'{task}_{split}.json'
            pack_path = PACKS / f'{task}_{split}.npz'
            metadata = json.loads(metadata_path.read_text())
            wanted = {c['id']: c for c in inputs['cases'] if c['task'] == task and c['split'] == split}
            with np.load(pack_path) as pack:
                targets, tokens = pack['target'], pack['token_id']
                development = pack['development']
                source_gaps = pack['context'][:, :2] + pack['observations'][:, :2]
            paths += [metadata_path, pack_path]
            if split == 'train':
                offsets = {True: np.r_[0, development.cumsum()], False: np.r_[0, (~development).cumsum()]}
                predicted = {}
                for dev, name in ((False, 'fit'), (True, 'development')):
                    path = LEGACY / task / f'{name}_scores.npz'
                    with np.load(path) as saved:
                        predicted[dev] = saved['fixed_unsupervised']
                    paths.append(path)
            else:
                path = LEGACY / task / 'test_scores.npz'
                with np.load(path) as saved:
                    predicted = saved['fixed_unsupervised']
                paths.append(path)
            for row in metadata['records']:
                if row['id'] not in wanted:
                    continue
                case = wanted[row['id']]
                begin, end = row['packed_start'], row['packed_stop']
                assert row['source_id'] == case['source_id']
                assert np.array_equal(tokens[begin:end], case['response']['answer_ids'])
                assert np.array_equal(targets[begin:end], np.arange(len(case['response']['answer_ids'])))
                source_values[row['id']] = source_gaps[begin:end]
                if split == 'train':
                    dev = bool(development[begin])
                    assert (development[begin:end] == dev).all()
                    first, last = offsets[dev][begin], offsets[dev][end]
                    values[row['id']] = predicted[dev][first:last]
                else:
                    values[row['id']] = predicted[begin:end]
    assert set(values) == {c['id'] for c in inputs['cases']}
    if sources_only:
        destination = output / 'legacy_source_gaps.npz'
        assert not destination.exists()
        np.savez_compressed(destination, **source_values)
        write_json(output / 'legacy_source_gaps_freeze.json', dict(hashes=digest_files(paths + [destination]),
            labels_read=False, channels=['local token contrast', 'full token contrast'],
            exact_token_ids=True, historical_template_differs=True))
        return
    destination = output / 'legacy_baseline.npz'
    assert not destination.exists()
    np.savez_compressed(destination, **values)
    write_json(output / 'legacy_baseline_freeze.json', dict(hashes=digest_files(paths + [destination]),
        score='existing fixed_unsupervised .75 source + .25 route; historical world',
        labels_read=False, exact_token_ids=True, existing_scores_unchanged=True,
        historical_template_differs=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    extract(args.output, args.sources_only)
