"""Check saved representations, split isolation, and question-cluster uncertainty."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.gsm8k_recurrence.measure import write_json
from experiments.gsm8k_recurrence.evaluate import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output/'manifest.json').read_text())
    previous = Path(manifest['previous'])
    partitions = defaultdict(set)
    last_margin_differences = []
    for row in manifest['records']:
        partitions[row['role']].add(row['problem'])
        with np.load(output/(row['id']+'.npz')) as saved:
            for name in saved.files:
                assert np.isfinite(saved[name]).all(),(row['id'],name)
            assert saved['end'].shape==(32,row['steps'],4096)
            for name in ('entry','mean'):
                assert saved[name].shape==saved['end'].shape
            assert saved['margin'].shape==(32,row['tokens'])
            np.testing.assert_allclose(np.exp(saved['conditional_logp']).sum(-1),1,atol=2e-6)
            with np.load(row['cache']) as original:
                first = int(original['response_idx'])
                np.testing.assert_array_equal(saved['response_ids'],original['token_ids'][first:])
                np.testing.assert_array_equal(saved['step_ranges'],original['step_ranges']-first)
            last_margin_differences.extend(np.abs(saved['margin'][-1]-saved['final_margin']))
        with np.load(output/'scores'/(row['id']+'.npz')) as scores, np.load(previous/'scores'/(row['id']+'.npz')) as old:
            np.testing.assert_array_equal(scores['attention_base'],old['base'])
            for name in scores.files:
                assert scores[name].shape==(row['steps'],)
                assert np.isfinite(scores[name]).all()
    roles = list(partitions)
    for index,role in enumerate(roles):
        for other in roles[index+1:]:
            assert not partitions[role]&partitions[other]
    with (output/'steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('step','label','first_error'):
            row[key] = int(row[key])
        row['score'] = float(row['score'])
    evaluation = [r for r in rows if r['role']=='evaluation']
    thresholds = json.loads((output/'thresholds.json').read_text())
    by_generator = {}
    for generator in sorted({r['generator'] for r in evaluation}):
        by_generator[generator] = {name:metrics([r for r in evaluation if r['generator']==generator and r['method']==name],thresholds[name]) for name in thresholds}
    grouped = defaultdict(lambda:defaultdict(list))
    names = ('attention_base','trajectory_rejection','final_margin','final_nll','layer_js','lda_end','lda_position')
    for row in evaluation:
        if row['label']>=0 and row['method'] in names:
            grouped[row['method']][row['problem']].append((row['label'],row['score']))
    problems = sorted(grouped['attention_base'])
    rng = np.random.default_rng(429)
    differences = defaultdict(list)
    for _ in range(300):
        chosen = rng.choice(problems,len(problems),replace=True)
        aucs = {}
        for name in names:
            data = np.array([pair for problem in chosen for pair in grouped[name][problem]])
            aucs[name] = roc_auc_score(data[:,0],data[:,1])
        for name in names[1:]:
            differences[name].append(aucs[name]-aucs['attention_base'])
    write_json(output/'verification.json',dict(status='passed',answers=len(manifest['records']),
        checks=['finite full states and scores','original token IDs and step ranges','normalized conditional distributions',
                'question-disjoint splits','attention baseline identical'],
        selected_vs_full_margin_difference_quantiles=np.quantile(last_margin_differences,[0,.5,.95,1]).tolist(),
        note='BF16 selected einsum and full-vocabulary GEMM differ; exact final margin uses full logits',
        auc_delta_vs_attention_base95={name:np.quantile(values,[.025,.975]).tolist() for name,values in differences.items()},
        bootstrap='300 paired question clusters, exploratory intervals, no multiple-testing correction',
        by_generator=by_generator))
    print('verified 400 full-state records, split isolation, and unchanged attention baseline',flush=True)


if __name__=='__main__':
    main()
