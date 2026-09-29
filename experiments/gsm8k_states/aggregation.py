"""Second-round fixed step aggregation ablation; no label fitting or test selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
from experiments.gsm8k_recurrence.measure import write_json
from experiments.gsm8k_recurrence.score import fit_reference


def top_fraction(values,ranges,fraction=.2):
    pooled = []
    for start,end in ranges:
        count = max(1,int(np.ceil((end-start)*fraction)))
        pooled.append(np.sort(values[start:end])[-count:].mean())
    return np.array(pooled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    (args.output/'scores').mkdir()
    manifest = json.loads((args.source/'manifest.json').read_text())
    write_json(args.output/'manifest.json',manifest)
    rows = manifest['records']
    scores = {}
    for row in rows:
        with np.load(args.source/(row['id']+'.npz')) as raw, np.load(args.source/'scores'/(row['id']+'.npz')) as old:
            logp = raw['conditional_logp'].astype(float)
            p = np.exp(logp)
            middle = (p[1:]+p[:-1])/2
            js = .5*(rel_entr(p[1:],middle).sum(-1)+rel_entr(p[:-1],middle).sum(-1))
            signals = dict(nll_top20=raw['nll'],margin_top20=-raw['final_margin'],
                trajectory_top20=-logp[-8:,:,0].mean(0),js_top20=js[-15:].mean(0))
            scores[row['id']] = {name:top_fraction(values,raw['step_ranges']) for name,values in signals.items()}
            for name in ('attention_base','trajectory_rejection','final_nll','final_margin','layer_js'):
                scores[row['id']][name] = old[name]
    # Mixed, question-equal empirical ranks learned on fit without any truth labels.
    fit = [r for r in rows if r['role']=='fit']
    ranks = {}
    for name in ('attention_base','trajectory_top20'):
        support,cdf = fit_reference(fit,{r['id']:scores[r['id']][name] for r in fit})
        ranks[name] = {r['id']:cdf[np.searchsorted(support,scores[r['id']][name],side='right')] for r in rows}
    for row in rows:
        scores[row['id']]['read_select_mean'] = .5*(ranks['attention_base'][row['id']]+ranks['trajectory_top20'][row['id']])
    names = list(next(iter(scores.values())))
    dev = [r for r in rows if r['role']=='dev']
    thresholds = {}
    for name in names:
        support,cdf = fit_reference(dev,{r['id']:scores[r['id']][name] for r in dev})
        thresholds[name] = float(support[np.searchsorted(cdf[1:],.95)])
    for key,values in scores.items():
        np.savez_compressed(args.output/'scores'/(key+'.npz'),**values)
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',main='read_select_mean',
        methods=names,unsupervised=names,supervised_diagnostics=[],labels_used=False,
        scope='v2 designed after viewing v1 evaluation; all candidates reported, exploratory'))
    print('frozen aggregation ablation scores',flush=True)


if __name__=='__main__':
    main()
