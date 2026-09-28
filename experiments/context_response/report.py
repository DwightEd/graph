"""Collect all completed pilot variants and show exact token-level outcomes."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.head_state_readout.audit import browser
from experiments.role_free_flow.diagnostics import read_local_spans


def tables(outputs):
    aggregate, natural = [], []
    for version, output in enumerate(outputs, 1):
        result = read_json(output/'evaluation.json')
        main = read_json(output/'scores_frozen.json')['main']
        aggregate.extend(dict(version=version, method=name, primary=name==main, **value)
                         for name, value in result['aggregate'].items())
        natural.extend(dict(version=version, **row) for row in result['natural_pairs'])
        with (output/'token_audit.csv').open() as stream:
            tokens = list(csv.DictReader(stream))
        for status in ('FP', 'FN', 'TP'):
            with (output/(status+'_tokens.csv')).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=tokens[0])
                writer.writeheader()
                writer.writerows(row for row in tokens if row['status']==status)
    for name, rows in [('all_round_metrics.csv', aggregate), ('all_natural_metrics.csv', natural)]:
        with (outputs[-1]/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return aggregate, natural


def plot(outputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    comparisons = [(outputs[3], 'original_full_reference_fixed', 'Original full-reference baseline'),
        (outputs[0], 'signed_tail', 'Signed response only'),
        (outputs[1], 'sparse_joint_fused', 'Sparse response + small-reference baseline'),
        (outputs[2], 'conditional_sparse_fused', 'Conditional + small-reference baseline'),
        (outputs[3], 'strong_fused', 'Sparse response + original baseline'),
        (outputs[4], 'request_joint_fused', 'Request-conditioned + original baseline')]
    with (outputs[0]/'tokens.csv').open() as stream:
        tokens = list(csv.DictReader(stream))
    records = {r['key']: r for r in read_json(outputs[0]/'manifest.json')['records']}
    keys = list(dict.fromkeys(row['key'] for row in tokens))
    figure, axes = plt.subplots(len(keys), 1, figsize=(15, 16), constrained_layout=True)
    colors = ListedColormap(['#eeeeee', '#66b88a', '#e67f76', '#ad88ca'])
    for axis, key in zip(axes, keys):
        selected = [r for r in tokens if r['key']==key]
        count = max(int(r['token']) for r in selected)+1
        matrix = np.zeros((len(comparisons), count))
        for index, (output, method, _) in enumerate(comparisons):
            score = np.load(output/key/'scores.npz')[method]
            threshold = read_json(output/'thresholds.json')[records[key]['task']][method]
            for row in selected:
                position, target = int(row['token']), int(row['label'])
                alarm = score[position]>threshold
                matrix[index, position] = (1 if target else 2) if alarm else (3 if target else 0)
        axis.imshow(matrix, aspect='auto', interpolation='nearest', cmap=colors, vmin=0, vmax=3)
        axis.set_yticks(range(len(comparisons)), [r[2] for r in comparisons], fontsize=8)
        axis.set_title(key+' / '+records[key]['task'], loc='left', fontsize=10)
        axis.set_xlabel('Original answer token position (zero-based)', fontsize=8)
    figure.suptitle('Frozen thresholds: green TP | red FP | purple FN | gray TN')
    figure.savefig(outputs[-1]/'token_positions.png', dpi=150)
    figure.savefig(outputs[-1]/'token_positions.svg')
    plt.close(figure)


def natural_browser(outputs):
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    manifest = read_json(outputs[0]/'manifest.json')
    records = {r['trace']: r for r in manifest['records'] if r['role']=='natural'}
    comparisons = [('signed', outputs[0], 'signed_tail'), ('sparse', outputs[1], 'sparse_joint'),
        ('conditional', outputs[2], 'conditional_sparse'), ('request', outputs[4], 'request_joint')]
    display = []
    for trace, selected in local.groupby('trace', sort=False):
        row = records[trace]
        with np.load(Path(manifest['samples'])/trace) as saved:
            pieces = saved['token_text'][int(saved['prompt_length']):]
        arrays = {label: np.load(output/row['key']/'scores.npz')[method] for label, output, method in comparisons}
        thresholds = {label: read_json(output/'thresholds.json')['QA'][method] for label, output, method in comparisons}
        tokens = []
        for token in selected.itertuples():
            values = {label: float(array[token.position]) for label, array in arrays.items()}
            status = {label: ('TP' if token.local_label else 'FP') if value>thresholds[label]
                      else ('FN' if token.local_label else 'TN') for label, value in values.items()}
            tokens.append(dict(position=int(token.position), text=str(pieces[token.position]), label=int(token.local_label), scores=values, status=status))
        display.append(dict(key=row['key'], task='QA / reviewed local tokens only', tokens=tokens))
    directory = outputs[-1]/'natural_local'
    directory.mkdir(exist_ok=True)
    browser(directory, display, [r[0] for r in comparisons])
    page = directory/'TOKEN_AUDIT.html'
    page.write_text(page.read_text().replace('官方标签仅用于冻结后显示；“正常”表示官方未标错。', '仅31个局部token有人工核验，其余位置未知；不展示缺来源分数的融合方法。'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    outputs = [Path(args.prefix+'_v'+str(i)) for i in range(1, 6)]
    aggregate, natural = tables(outputs)
    plot(outputs)
    natural_browser(outputs)
    write_json(outputs[-1]/'report_complete.json', dict(status='complete', rounds=5,
        candidate_rows_including_reused_controls=len(aggregate), natural_rows=len(natural)))


if __name__=='__main__':
    main()
