"""Match the null edge-weight distribution without changing frozen primary scores."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from .model import solve_correction


def match_weights(native,shuffled):
    matched = np.empty_like(native)
    matched[np.argsort(shuffled,kind='stable')] = np.sort(native)
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json',manifest)
    frozen = read_json(args.previous/'scores_frozen.json')
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('observations.npz','observations_ranked.npz'):
            (directory/name).symlink_to((args.previous/key/name).resolve())
        with np.load(args.previous/key/'scores.npz') as saved:
            scores = {name:saved[name] for name in saved.files}
        if row['role']=='case':
            (directory/'edges.npz').symlink_to((args.previous/key/'edges.npz').resolve())
            edge = np.load(directory/'edges.npz')
            matched = match_weights(edge['edges'],edge['shuffled'])
            np.testing.assert_array_equal(np.sort(matched),np.sort(edge['edges']))
            correction,gap = solve_correction(scores['token_observation']-scores['base'],matched)
            scores['matched_shuffle_tv'] = scores['base']+correction
            frozen['dual_gaps'][key]['matched_shuffle_tv'] = gap
            np.savez_compressed(directory/'matched_control.npz',edges=matched)
        np.savez_compressed(directory/'scores.npz',**scores)
    for name in ('thresholds.json','references.joblib','capture_complete.json','edges_complete.json'):
        (args.output/name).symlink_to((args.previous/name).resolve())
    frozen['methods'].append('matched_shuffle_tv')
    frozen['matched_control'] = 'rank-match shuffled head edges to exact native weight multiset within each answer'
    frozen['primary_unchanged'] = True
    write_json(args.output/'scores_frozen.json',frozen)
    print('primary unchanged; matched-strength control frozen',flush=True)


if __name__=='__main__':
    main()
