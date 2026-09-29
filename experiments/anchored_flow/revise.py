"""Second iteration: keep the source anchor, refine the routing component only."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from .score import rank,METHODS
from .model import score_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json',manifest)
    fitted = joblib.load(args.previous/'references.joblib')
    gaps = {}
    for row in manifest['records']:
        key = row['key']
        target = args.output/key
        target.mkdir()
        files = ['observations.npz','observations_ranked.npz']
        if row['role']=='case':
            files.append('edges.npz')
        for name in files:
            (target/name).symlink_to((args.previous/key/name).resolve())
        with np.load(args.previous/key/'scores.npz') as saved:
            base = saved['base'].copy()
            if row['role']=='case':
                prior = {f'previous_{name}':saved[name] for name in ('anchored_flow','token_observation')}
        if row['role']!='case':
            np.savez_compressed(target/'scores.npz',base=base)
            continue
        raw = np.load(target/'observations_ranked.npz')
        reference = fitted[row['task']]['route']
        route_innovation = .25*(rank(raw['raw_route'],reference)-rank(raw['route'],reference))
        edge = np.load(target/'edges.npz')
        scores,gaps[key] = score_observations(base,base+route_innovation,edge['edges'],edge['shuffled'])
        scores.update(prior)
        np.savez_compressed(target/'scores.npz',**scores)
    for name in ('thresholds.json','references.joblib','capture_complete.json','edges_complete.json'):
        (args.output/name).symlink_to((args.previous/name).resolve())
    previous = read_json(args.previous/'scores_frozen.json')
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=list(METHODS)+list(prior),
        primary='anchored_flow',labels_used=False,dual_gaps=gaps,baseline_max_errors=previous['baseline_max_errors'],
        correction='route only; source-pair anchor exactly retained; otherwise same convex problem and threshold',
        provenance='second iteration after exposed v1 results; not independent confirmation',
        previous=str(args.previous.resolve())))
    print('route-only correction frozen',flush=True)


if __name__=='__main__':
    main()
