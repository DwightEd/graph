"""Keep the original full-reference unsupervised baseline instead of refitting it on four sources."""
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json

ORIGINAL = Path('outputs/unsupervised_graph_20260928')
PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')


def cached_baselines(wanted, train=True):
    result = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test') if train else ('test',):
            metadata = read_json(PACKS/f'{task}_{split}.json')
            with np.load(PACKS/f'{task}_{split}.npz') as saved:
                positions = saved['target']
            files = {'fit': 'fit_scores', 'dev': 'development_scores'} if split=='train' else {'test': 'test_scores'}
            arrays = {part: np.load(ORIGINAL/task/(name+'.npz'))['fixed_unsupervised'] for part, name in files.items()}
            cursor = {part: 0 for part in files}
            for row in metadata['records']:
                part = row['partition']
                count = row['packed_stop']-row['packed_start']
                region = slice(cursor[part], cursor[part]+count)
                if row['id'] in wanted:
                    values = np.full(row['tokens'], np.nan)
                    selected = positions[row['packed_start']:row['packed_stop']]
                    values[selected] = arrays[part][region]
                    result[row['id']] = values
                cursor[part] += count
            assert all(cursor[part]==len(array) for part, array in arrays.items())
    return result


def original_threshold(task):
    return read_json(ORIGINAL/task/'selection.json')['mixed_thresholds']['fixed_unsupervised']
