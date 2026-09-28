"""No hard seed: pool continuous observations with an explicit correlation model."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import logit
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.context_response.restore import original_threshold
from experiments.context_response.score import calibrate

BASE = ('original_full_reference_fixed', 'strong_fused', 'corroborated_only')


def restart_weights(graph):
    undirected = graph+graph.T
    total = undirected.sum(-1, keepdims=True)
    transition = np.divide(undirected, total, out=np.zeros_like(undirected), where=total>0)
    weights = .5*np.eye(len(graph))+.5*transition
    weights[np.flatnonzero(total[:,0]==0), np.flatnonzero(total[:,0]==0)] = 1.
    return weights


def pool(z, weights, rho):
    count = len(z)
    distance = np.abs(np.arange(count)[:,None]-np.arange(count)[None,:])
    covariance = rho**distance
    variance = ((weights@covariance)*weights).sum(-1)
    return (weights@z)/np.sqrt(variance)


def fit_observation(values, rows, output):
    selected = []
    adjacent = []
    for row in rows:
        if row['role']!='fit':
            continue
        valid = valid_tokens(row, output, OLD)
        raw = values[row['key']]
        selected.append(raw[valid])
        adjacent.append((raw[:-1][valid[:-1]&valid[1:]], raw[1:][valid[:-1]&valid[1:]]))
    mean = np.mean([value.mean(0) for value in selected], axis=0)
    variance = np.mean([((value-mean)**2).mean(0) for value in selected], axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    correlation = np.mean([((left-mean)*(right-mean)/variance).mean(0) for left,right in adjacent], axis=0)
    return dict(mean=mean, scale=scale, rho=np.clip(correlation, 0, .99))


def score_record(raw, base, graphs, fitted):
    z = (raw-fitted['mean'])/fitted['scale']
    scores = dict(base)
    scores['source_z'] = z[:,0]
    scores['rejection_z'] = z[:,1]
    scores['joint_direct'] = np.max(z, axis=1)
    for name in ('selective', 'attention', 'matched'):
        weights = restart_weights(graphs[name])
        channels = np.stack([pool(z[:,i], weights, fitted['rho'][i]) for i in (0,1)], -1)
        scores[name+'_source'] = channels[:,0]
        scores[name+'_rejection'] = channels[:,1]
        scores[name+'_joint'] = np.max(channels, axis=1)
    weights = restart_weights(graphs['selective'])
    independent = np.stack([pool(z[:,i], weights, 0.) for i in (0,1)], -1)
    scores['independent_joint'] = np.max(independent, axis=1)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output/'features_complete.json')
    manifest = read_json(args.output/'manifest.json')
    previous = Path(manifest['previous'])
    observations, baselines = {}, {}
    for row in manifest['records']:
        key = row['key']
        with np.load(previous/key/'scores.npz') as saved:
            baselines[key] = {name: saved[name] for name in BASE}
        with np.load(Path(manifest['base'])/key/'readouts.npz') as saved:
            rejection = -saved['confidence'][:,2]
        risk = logit(np.clip(baselines[key]['original_full_reference_fixed'], 1e-4, 1-1e-4))
        observations[key] = np.stack((risk, rejection), -1)
    fitted, thresholds = {}, {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        fitted[task] = fit_observation(observations, rows, args.output)
        scores = {}
        for row in rows:
            key = row['key']
            with np.load(args.output/key/'graph.npz') as saved:
                graphs = {name: saved[name] for name in ('selective', 'attention', 'matched')}
            scores[key] = score_record(observations[key], baselines[key], graphs, fitted[task])
            np.savez_compressed(args.output/key/'scores.npz', **scores[key])
        thresholds[task] = calibrate(args.output, rows, scores)
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
    write_json(args.output/'fit.json', {task: {name:value.tolist() for name,value in fit.items()} for task,fit in fitted.items()})
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', main='selective_joint',
        methods=list(next(iter(scores.values()))), labels_used=False, prior_discovery_label_informed=True,
        meaning='correlation-normalized continuous risk and observer rejection, not Gaussian factuality probabilities',
        covariance='AR1 approximation rho in [0,.99]; unlabeled mixture, no normal-FPR guarantee',
        graph='actual history key/query message read/use excess over lag/copy expectation; offline symmetric kernel'))


if __name__=='__main__':
    main()
