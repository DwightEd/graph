"""Step scores from signed output trajectories; full-coordinate LDA is diagnostic only."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
from experiments.gsm8k_recurrence.measure import write_json
from experiments.gsm8k_recurrence.score import fit_reference
from experiments.gsm8k_recurrence.evaluate import step_label

LAYERS = [7,15,23,31]
UNSUPERVISED = ('attention_base','final_margin','final_nll','final_entropy','final_conditional_nll',
                'trajectory_rejection','late_margin','layer_js','hidden_path_ratio','spectral_concentration')
VIEWS = ('end','delta','trajectory','position')


def energy_rank(matrix, center=True):
    """Energy-spectrum effective rank; rows are observations, columns coordinates."""
    values = np.asarray(matrix, dtype=np.float64)
    maximum = min(len(values) - int(center), values.shape[1])
    if maximum < 1:
        return dict(rank=None, normalized=None, participation=None)
    if center:
        values = values - values.mean(0)
    energy = np.maximum(np.linalg.eigvalsh(values @ values.T), 0)
    if energy.sum() == 0:
        return dict(rank=0., normalized=0., participation=0.)
    probability = energy / energy.sum()
    effective = float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))))
    return dict(rank=effective, normalized=effective / maximum,
                participation=float(1 / np.sum(probability ** 2)))


def entropy(probability):
    return -np.sum(probability * np.log(np.maximum(probability, 1e-30)), axis=-1)


def attention_geometry(attention, start, stop):
    """Use the prediction queries start-1:stop-1, retaining physical heads."""
    weights = attention[:, start-1:stop-1, :stop-1].astype(np.float64)
    weights /= weights.sum(-1, keepdims=True)
    visible = np.arange(start, stop)
    total_entropy = entropy(weights)
    prior = weights[:, :, :start]
    prior_mass = prior.sum(-1)
    prior = np.divide(prior, prior_mass[:, :, None],
                      out=np.zeros_like(prior), where=prior_mass[:, :, None] > 0)
    return dict(attention_entropy=(total_entropy / np.log(np.maximum(visible, 2))).mean(0),
                prior_entropy=(entropy(prior) / np.log(max(start, 2))).mean(0),
                prior_mass=prior_mass.mean(0),
                head_disagreement=(entropy(weights.mean(0))-total_entropy.mean(0))/np.log(len(weights)))


def hidden_geometry(states):
    """Within-step token geometry; token increments are not layer-block updates."""
    rank = energy_rank(states)
    changes = energy_rank(np.diff(states.astype(float), axis=0), center=False)
    early_count = max(1, int(np.ceil(len(states) * .25)))
    early = energy_rank(states[:early_count])
    return dict(hidden_rank=rank['rank'], hidden_rank_normalized=rank['normalized'],
                hidden_participation=rank['participation'],
                token_change_rank_normalized=changes['normalized'],
                early_hidden_rank_normalized=early['normalized'])


def geometry_record(row, output, hidden_path):
    with np.load(row['cache']) as saved:
        first = int(saved['response_idx'])
        ranges = saved['step_ranges']
        ids = saved['token_ids']
        attention = saved['attention'][0]
    activation = None
    if hidden_path is not None:
        with np.load(hidden_path) as saved:
            np.testing.assert_array_equal(ids, saved['token_ids'])
            np.testing.assert_array_equal(ranges, saved['step_ranges'])
            assert int(saved['activation_layer']) == 15
            activation = saved['activation']
    with np.load(output / (row['id']+'.npz')) as saved:
        end = saved['end'].astype(float)
    rows = []
    for step, (start, stop) in enumerate(ranges):
        signals = attention_geometry(attention, start, stop)
        early_count = max(1, int(np.ceil((stop-start)*.25)))
        result = dict(id=row['id'], problem=row['problem'], role=row['role'],
                      generator=row['generator'], step=step, length=int(stop-start),
                      position=int(start-first), hidden_cache=str(hidden_path) if hidden_path else None)
        for name, values in signals.items():
            result[name] = float(values.mean())
            result['early_'+name] = float(values[:early_count].mean())
        result['end_layer_update_rank'] = energy_rank(np.diff(end[:, step], axis=0), center=False)['rank']
        if activation is not None:
            result.update(hidden_geometry(activation[start-1:stop-1]))
        rows.append(result)
    return rows


def measure_geometry(output, hidden_cache):
    """Read caches only; freeze every feature before opening official error labels."""
    import hashlib
    from time import perf_counter
    destination = output / 'geometry_features.json'
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((output/'manifest.json').read_text())
    hidden_files = {}
    for path in sorted(hidden_cache.glob('*.npz')):
        with np.load(path) as saved:
            hidden_files[str(saved['sample_id'])] = path.resolve()
    protocol = dict(status='measuring', direction='higher dispersion/rank predicts first error',
                    labels_used_for_features=False, new_llm_forwards=0,
                    prediction_query='response position minus one', early_fraction=.25,
                    rank='entropy of squared singular values, centered across tokens',
                    rank_scope='HS15 within-token geometry for 61; step-end cross-layer control for 400',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    hidden_cache=str(hidden_cache.resolve()))
    write_json(output/'geometry_protocol.json', protocol)
    started = perf_counter()
    rows = []
    for index, row in enumerate(manifest['records']):
        rows.extend(geometry_record(row, output, hidden_files.get(row['id'])))
        if (index+1) % 25 == 0:
            print('geometry measured', index+1, 'seconds', round(perf_counter()-started, 1), flush=True)
    write_json(destination, dict(status='frozen', rows=rows, labels_used=False,
                                answers=len(manifest['records']), hidden_answers=len(hidden_files),
                                seconds=perf_counter()-started))
    print('geometry features frozen', len(rows), flush=True)


def average_steps(value, ranges):
    return np.stack([value[...,a:b].mean(-1) for a,b in ranges])


def observations(saved,row,previous):
    ranges = saved['step_ranges']
    logp = saved['conditional_logp'].astype(float)
    probability = np.exp(logp)
    middle = (probability[1:]+probability[:-1])/2
    js = .5*(rel_entr(probability[1:],middle).sum(-1)+rel_entr(probability[:-1],middle).sum(-1))
    margin = average_steps(saved['margin'],ranges)
    rejection = average_steps(-logp[:,:,0],ranges)
    divergence = average_steps(js,ranges)
    end = saved['end'].astype(float).transpose(1,0,2)
    entry = saved['entry'].astype(float).transpose(1,0,2)
    difference = end-entry
    increments = np.linalg.norm(np.diff(end,axis=1),axis=-1)
    net = np.linalg.norm(end[:,-1]-end[:,-8],axis=-1)
    updates = np.diff(end,axis=1)
    gram = updates@updates.transpose(0,2,1)
    energy = np.maximum(np.linalg.eigvalsh(gram),0)
    energy /= np.maximum(energy.sum(-1,keepdims=True),1e-30)
    effective_rank = np.exp(-np.sum(energy*np.log(np.maximum(energy,1e-30)),axis=-1))
    with np.load(previous/'scores'/(row['id']+'.npz')) as old:
        attention = old['base']
    scores = dict(attention_base=attention,
        final_margin=average_steps(-saved['final_margin'],ranges),
        final_nll=average_steps(saved['nll'],ranges),
        final_entropy=average_steps(saved['entropy'],ranges),
        final_conditional_nll=rejection[:,-1],trajectory_rejection=rejection[:,-8:].mean(-1),
        late_margin=-margin[:,-8:].mean(-1),layer_js=divergence[:,-15:].mean(-1),
        hidden_path_ratio=increments[:,-7:].sum(-1)/np.maximum(net,1e-8),
        spectral_concentration=1-effective_rank/31)
    trajectory = np.concatenate((margin,rejection,divergence,
        np.log1p(np.linalg.norm(difference,axis=-1)),np.log1p(increments)),axis=-1)
    position = np.column_stack((np.arange(len(ranges))/len(ranges),np.log1p(ranges[:,1]-ranges[:,0]),
        np.full(len(ranges),np.log(row['prompt_tokens'])),np.log1p(ranges[:,0])))
    views = dict(end=end[:,LAYERS].reshape(len(ranges),-1),
        delta=difference[:,LAYERS].reshape(len(ranges),-1),trajectory=trajectory,position=position)
    return scores,views


def fit_lda(features, labels, weights, shrinkage=.5):
    """Diagonal-standardized pooled LDA via Woodbury; retains every input coordinate."""
    weights = weights/weights.sum()
    center = weights@features
    scale = np.sqrt(weights@((features-center)**2))
    scale = np.maximum(scale,1e-4)
    normalized = (features-center)/scale
    means = np.stack([np.average(normalized[labels==label],axis=0,weights=weights[labels==label]) for label in (0,1)])
    residual = (normalized-means[labels])*np.sqrt(weights[:,None])
    delta = means[1]-means[0]
    dual = residual@residual.T+shrinkage/(1-shrinkage)*np.eye(len(residual))
    direction = (delta-residual.T@np.linalg.solve(dual,residual@delta))/shrinkage
    prior = weights[labels==1].sum()
    intercept = -.5*(means[0]+means[1])@direction+np.log(prior/(1-prior))
    return dict(center=center,scale=scale,direction=direction,intercept=intercept)


def apply_lda(features,model):
    return ((features-model['center'])/model['scale'])@model['direction']+model['intercept']


def train_diagnostics(rows,features,originals,output):
    fit = [r for r in rows if r['role']=='fit']
    counts = Counter(r['problem'] for r in fit)
    labels,weights,indices = [],[],[]
    for row in fit:
        gold = originals[row['id']]['label']
        y = np.array([step_label(s,gold) for s in range(row['steps'])])
        keep = np.flatnonzero(y>=0)
        labels.extend(y[keep])
        weights.extend(np.full(len(keep),1/(len(keep)*counts[row['problem']])))
        indices.append((row['id'],keep))
    scores = {r['id']:{} for r in rows}
    for view in VIEWS:
        matrix = np.concatenate([features[key][view][keep] for key,keep in indices])
        model = fit_lda(matrix,np.array(labels),np.array(weights))
        np.savez_compressed(output/('lda_'+view+'.npz'),**model)
        for row in rows:
            scores[row['id']]['lda_'+view] = apply_lda(features[row['id']][view],model)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry',action='store_true')
    parser.add_argument('--hidden-cache',type=Path)
    args = parser.parse_args()
    if args.geometry:
        if args.hidden_cache is None:
            parser.error('--geometry requires --hidden-cache')
        measure_geometry(args.output, args.hidden_cache)
        return
    json.loads((args.output/'capture_complete.json').read_text())
    manifest = json.loads((args.output/'manifest.json').read_text())
    rows = manifest['records']
    previous = Path(manifest['previous'])
    (args.output/'scores').mkdir()
    all_scores,features = {},{}
    for row in rows:
        with np.load(args.output/(row['id']+'.npz')) as saved:
            all_scores[row['id']],features[row['id']] = observations(saved,row,previous)
    dev = [r for r in rows if r['role']=='dev']
    thresholds = {}
    for name in UNSUPERVISED:
        cdf = fit_reference(dev,{r['id']:all_scores[r['id']][name] for r in dev})
        thresholds[name] = float(cdf[0][np.searchsorted(cdf[1][1:],.95)])
    # Supervised diagnostics are explicitly separate from all unsupervised features/scores above.
    originals = {r['id']:r for r in json.loads(Path(manifest['data']).read_text())}
    diagnostics = train_diagnostics(rows,features,originals,args.output)
    for row in rows:
        all_scores[row['id']].update(diagnostics[row['id']])
    for view in VIEWS:
        name = 'lda_'+view
        values = {}
        for row in dev:
            gold = originals[row['id']]['label']
            keep = np.array([step_label(s,gold)==0 for s in range(row['steps'])])
            if keep.any():
                values[row['id']] = all_scores[row['id']][name][keep]
        selected = [r for r in dev if r['id'] in values]
        cdf = fit_reference(selected,values)
        thresholds[name] = float(cdf[0][np.searchsorted(cdf[1][1:],.95)])
    for key,scores in all_scores.items():
        np.savez_compressed(args.output/'scores'/(key+'.npz'),**scores)
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',main='trajectory_rejection',
        methods=list(next(iter(all_scores.values()))),unsupervised=list(UNSUPERVISED),
        supervised_diagnostics=['lda_'+v for v in VIEWS],hidden_layers=LAYERS,
        supervised_labels='fit training and dev normal thresholds only; evaluation not used',
        unsupervised_labels_used=False,selection='no evaluation selection or direction reversal'))
    print('frozen full-state step scores',flush=True)


if __name__=='__main__':
    main()
