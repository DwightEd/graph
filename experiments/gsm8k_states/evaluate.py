"""First-error evaluation and exact supervised/unsupervised failure locations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from experiments.gsm8k_recurrence.measure import write_json
from experiments.gsm8k_recurrence.evaluate import metrics,step_label


def cluster_interval(pairs):
    """Average within problem before resampling; tokens/layers are not replicates."""
    problems = sorted({row['problem'] for row in pairs})
    if not problems:
        return dict(pairs=0, problems=0, mean_difference=None, ci95=None)
    values = np.array([np.mean([r['difference'] for r in pairs if r['problem']==problem])
                       for problem in problems])
    rng = np.random.default_rng(930)
    draws = rng.choice(values, (1000, len(values)), replace=True).mean(1)
    return dict(pairs=len(pairs), problems=len(problems), mean_difference=float(values.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist())


def geometry_summary(rows, name):
    from scipy.stats import spearmanr
    available = [r for r in rows if r.get(name) is not None]
    development = [r[name] for r in available if r['role']=='dev']
    threshold = float(np.quantile(development, .95, method='higher'))
    evaluation = [r for r in available if r['role']=='evaluation']
    known = [r for r in evaluation if r['label']>=0]
    y = np.array([r['label'] for r in known])
    values = np.array([r[name] for r in known])
    normal, first = values[y==0], values[y==1]
    mean_normal, mean_first = float(normal.mean()), float(first.mean())
    result = dict(known_steps=len(known), first_errors=len(first), normal_steps=len(normal),
                  normal_mean=mean_normal, first_error_mean=mean_first,
                  difference=mean_first-mean_normal,
                  relative_difference=(mean_first-mean_normal)/mean_normal if mean_normal else None,
                  auroc=float(roc_auc_score(y, values)),
                  length_spearman=float(spearmanr(values, [r['length'] for r in known]).statistic),
                  threshold=threshold, first_error_hits=0, error_answers=0,
                  correct_no_alarm=0, correct_answers=0)
    for key in dict.fromkeys(r['id'] for r in evaluation):
        group = [r for r in evaluation if r['id']==key]
        alarms = [r['step'] for r in group if r[name]>threshold]
        prediction = min(alarms) if alarms else -1
        gold = group[0]['first_error']
        result['error_answers' if gold>=0 else 'correct_answers'] += 1
        if prediction==gold:
            result['first_error_hits' if gold>=0 else 'correct_no_alarm'] += 1
    return result


def adjacent_geometry_pairs(rows, name, target_label):
    lookup = {(r['id'], r['step']):r for r in rows if r['role']=='evaluation'}
    pairs = []
    for current in lookup.values():
        previous = lookup.get((current['id'], current['step']-1))
        if previous is None or previous['label']!=0 or current['label']!=target_label:
            continue
        if current.get(name) is None or previous.get(name) is None:
            continue
        pairs.append(dict(id=current['id'], problem=current['problem'], step=current['step'],
                          method=name, current=current[name], previous=previous[name],
                          difference=current[name]-previous[name],
                          current_length=current['length'], previous_length=previous['length']))
    return pairs


def equal_length_hidden_pairs(rows, target_label=1):
    """Uniformly sample states; calculate real token increments before sampling."""
    from .readout import energy_rank
    lookup = {(r['id'], r['step']):r for r in rows}
    first = [r for r in rows if r['role']=='evaluation' and r['label']==target_label and r['step']>0
             and r.get('hidden_cache') and lookup[(r['id'],r['step']-1)]['label']==0]
    pairs = []
    for row in first:
        with np.load(row['hidden_cache']) as saved:
            ranges = saved['step_ranges']
            states = saved['activation']
        matrices = [states[a-1:b-1] for a,b in ranges[row['step']-1:row['step']+1]]
        for view in ('state', 'token_change', 'early_state', 'state_prefix', 'state_suffix'):
            values = matrices
            if view=='token_change':
                values = [np.diff(value.astype(float), axis=0) for value in matrices]
            elif view=='early_state':
                values = [value[:max(1, int(np.ceil(len(value)*.25)))] for value in matrices]
            count = min(map(len, values))
            if count<2:
                continue
            if view=='state_prefix':
                sampled = [value[:count] for value in values]
            elif view=='state_suffix':
                sampled = [value[-count:] for value in values]
            else:
                sampled = [value[np.linspace(0, len(value)-1, count).astype(int)] for value in values]
            ranks = [energy_rank(value, center=view!='token_change')['rank'] for value in sampled]
            pairs.append(dict(id=row['id'], problem=row['problem'], step=row['step'],
                              method=view, target_label=target_label, matched_count=count, previous=ranks[0], current=ranks[1],
                              difference=ranks[1]-ranks[0]))
    return pairs


def plot_geometry(output):
    """Standalone diagnostic figure; no new scoring or label-based selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = json.loads((output/'geometry_evaluation.json').read_text())
    pairs = json.loads((output/'geometry_pairs.json').read_text())
    with (output/'geometry_steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    selected = [r for r in pairs['adjacent'] if r['method']=='early_attention_entropy']
    axes[0].scatter([r['previous'] for r in selected], [r['current'] for r in selected], s=15, alpha=.65)
    axes[0].plot([.2,.85], [.2,.85], color='gray', linestyle='--')
    axes[0].set(xlabel='Previous normal step entropy', ylabel='First-error step entropy',
                title=f'Early attention ({len(selected)} pairs)')
    selected = [r for r in rows if r['role']=='evaluation' and r['hidden_rank'] and int(r['label'])>=0]
    for label, name, color in [(0,'Normal','#24678d'),(1,'First error','#c35432')]:
        group = [r for r in selected if int(r['label'])==label]
        axes[1].scatter([int(r['length']) for r in group], [float(r['hidden_rank']) for r in group],
                        label=name, color=color, s=17, alpha=.7)
    axes[1].set(xlabel='Step token count', ylabel='Centered hidden energy effective rank',
                title='Rank versus length (HS15)')
    axes[1].legend(frameon=False)
    for index, name in enumerate(('state','state_prefix','state_suffix','early_state')):
        value = results['equal_length_hidden'][name]
        low, high = value['ci95']
        axes[2].plot([low,high], [index,index], color='#24678d')
        axes[2].scatter(value['mean_difference'], index, color='#24678d')
    axes[2].axvline(0, color='gray', linestyle='--')
    axes[2].set(yticks=range(4), yticklabels=['Uniform over step','Contiguous prefix','Contiguous suffix','Early quarter'],
                xlabel='First error minus previous rank; 95% interval', title='Equal-count rank (20 pairs)')
    figure.savefig(output/'geometry_diagnostics.png', dpi=180, bbox_inches='tight')
    figure.savefig(output/'geometry_diagnostics.svg', bbox_inches='tight')
    plt.close(figure)


def evaluate_geometry(output):
    frozen = json.loads((output/'geometry_features.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    original = {r['id']:r for r in json.loads(Path(manifest['data']).read_text())}
    rows = frozen['rows']
    for row in rows:
        gold = original[row['id']]['label']
        row.update(label=step_label(row['step'], gold), first_error=gold,
                   text=original[row['id']]['steps'][row['step']])
    names = ['attention_entropy','prior_entropy','prior_mass','head_disagreement',
             'early_attention_entropy','early_prior_entropy','early_head_disagreement',
             'hidden_rank','hidden_rank_normalized','hidden_participation',
             'token_change_rank_normalized','early_hidden_rank_normalized','end_layer_update_rank','length']
    results = {}
    pairs = []
    for name in names:
        result = geometry_summary(rows, name)
        first_pairs = adjacent_geometry_pairs(rows, name, 1)
        normal_pairs = adjacent_geometry_pairs(rows, name, 0)
        result['first_minus_previous'] = cluster_interval(first_pairs)
        result['normal_minus_previous'] = cluster_interval(normal_pairs)
        results[name] = result
        pairs.extend(first_pairs)
    matched = equal_length_hidden_pairs(rows)
    equal_length = {name:cluster_interval([r for r in matched if r['method']==name])
                    for name in ('state','token_change','early_state','state_prefix','state_suffix')}
    normal_matched = equal_length_hidden_pairs(rows, target_label=0)
    normal_equal_length = {name:cluster_interval([r for r in normal_matched if r['method']==name])
                           for name in equal_length}
    hidden_rows = [r for r in rows if r.get('hidden_cache')]
    hidden_length = geometry_summary(hidden_rows, 'length')
    write_json(output/'geometry_evaluation.json', dict(status='complete', methods=results,
        equal_length_hidden=equal_length, normal_equal_length_hidden=normal_equal_length,
        hidden_subset_length_control=hidden_length,
        posthoc_controls='length-only, contiguous prefix/suffix, normal-normal equal-count transitions',
        bootstrap='1000 question clusters; exploratory, unadjusted',
        labels_scope='frozen feature diagnosis, no label training or direction selection',
        detector='dev mixture95; high scores only; first over-threshold step',
        scope='400 attention and step summaries; 61 single-layer token hidden; observer replay'))
    write_json(output/'geometry_pairs.json', dict(adjacent=pairs, equal_length=matched,
                                                 normal_equal_length=normal_matched))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/'geometry_steps.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_geometry(output)
    print(json.dumps({name:dict(auc=r['auroc'], pair=r['first_minus_previous'])
                      for name,r in results.items()}, indent=2), flush=True)


def make_rows(output,manifest,methods):
    original = {r['id']:r for r in json.loads(Path(manifest['data']).read_text())}
    result = []
    for row in manifest['records']:
        record = original[row['id']]
        gold = record['label']
        with np.load(output/'scores'/(row['id']+'.npz')) as saved:
            for name in methods:
                for step,score in enumerate(saved[name]):
                    result.append(dict(id=row['id'],problem=row['problem'],generator=row['generator'],
                        role=row['role'],method=name,step=step,label=step_label(step,gold),first_error=gold,
                        score=float(score),text=record['steps'][step]))
    return result


def within_answer(rows):
    values = []
    for key in dict.fromkeys(r['id'] for r in rows):
        group = [r for r in rows if r['id']==key and r['label']>=0]
        y = [r['label'] for r in group]
        if len(set(y))==2:
            values.append(roc_auc_score(y,[r['score'] for r in group]))
    return dict(mean=float(np.mean(values)),answers=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry',action='store_true')
    args = parser.parse_args()
    if args.geometry:
        evaluate_geometry(args.output)
        return
    output = args.output
    manifest = json.loads((output/'manifest.json').read_text())
    frozen = json.loads((output/'scores_frozen.json').read_text())
    thresholds = json.loads((output/'thresholds.json').read_text())
    rows = make_rows(output,manifest,frozen['methods'])
    result, errors = {},[]
    for name in frozen['methods']:
        result[name] = {}
        for role in ('fit','dev','evaluation'):
            selected = [r for r in rows if r['method']==name and r['role']==role]
            value = metrics(selected,thresholds[name])
            value['within_answer'] = within_answer(selected)
            result[name][role] = value
            if role!='evaluation':continue
            for row in selected:
                label = row['label']
                alarm = row['score']>thresholds[name]
                if (label==1 and not alarm) or (label==0 and alarm):
                    errors.append(dict(**row,status='FN' if label else 'FP',threshold=thresholds[name]))
        # Independent rank definition of AUROC on the known evaluation steps.
        known = [r for r in rows if r['method']==name and r['role']=='evaluation' and r['label']>=0]
        y = np.array([r['label'] for r in known])
        ranks = rankdata([r['score'] for r in known])
        positive = y.sum()
        auc = (ranks[y==1].sum()-positive*(positive+1)/2)/(positive*(len(y)-positive))
        assert abs(auc-result[name]['evaluation']['step_auroc'])<1e-12
    for file,items in [('steps.csv',rows),('errors.csv',errors)]:
        with (output/file).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    write_json(output/'evaluation.json',dict(status='complete',metrics=result,
        primary=frozen['main'],supervised_diagnostics=frozen['supervised_diagnostics'],
        steps='first error and earlier/all-correct known, later unknown',
        verification='all evaluation AUROCs independently rank-recomputed, same agent'))
    print({name:result[name]['evaluation']['step_auroc'] for name in result},flush=True)


if __name__=='__main__':
    main()
