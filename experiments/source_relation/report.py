"""Export all completed candidates and explicit per-token failure lists."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix',required=True)
    args = parser.parse_args()
    outputs = [Path(args.output_prefix+'_v'+str(i)) for i in range(1,6)]
    table,natural = [],[]
    for version,output in enumerate(outputs,1):
        evaluated = read_json(output/'evaluation.json')
        main = read_json(output/'scores_frozen.json')['main']
        for name,result in evaluated['aggregate'].items():
            table.append(dict(version=version,method=name,primary=name==main,**result))
        natural.extend(dict(version=version,**row) for row in evaluated['natural_pairs'])
        with (output/'token_audit.csv').open() as stream:
            tokens = list(csv.DictReader(stream))
        for category in ('FP','FN','TP'):
            chosen = [row for row in tokens if row['status']==category]
            with (output/(category+'_tokens.csv')).open('w') as stream:
                writer = csv.DictWriter(stream,fieldnames=tokens[0])
                writer.writeheader()
                writer.writerows(chosen)
    for name,rows in [('all_round_metrics.csv',table),('all_natural_metrics.csv',natural)]:
        with (outputs[-1]/name).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    checks = []
    for version,output in enumerate(outputs,1):
        evaluated = read_json(output/'evaluation.json')
        main = read_json(output/'scores_frozen.json')['main']
        for name,result in evaluated['aggregate'].items():
            pairs = [r for r in evaluated['natural_pairs'] if r['method']==name]
            natural_complete = all(r['scored_tokens']==r['eligible_tokens'] for r in pairs)
            local_tp = sum(r['detected_errors'] for r in pairs)
            local_fp = sum(r['false_alarms'] for r in pairs)
            recall = result['detected_errors']/result['error_tokens']
            fpr = result['false_alarms']/result['normal_tokens']
            goal = recall>=.95 and fpr<=.05 and natural_complete and local_tp/18>=.95 and local_fp/13<=.05
            checks.append(dict(version=version,method=name,primary=name==main,regression_recall=recall,
                regression_fpr=fpr,natural_complete=natural_complete,natural_detected=local_tp,
                natural_false_alarms=local_fp,goal_passed=goal))
    write_json(outputs[-1]/'goal_audit.json',dict(criteria='each official/natural subgroup: recall>=.95, FPR<=.05, complete coverage',
        rows=checks,any_candidate_passed=any(r['goal_passed'] for r in checks),posthoc_method_selection=False))
    plot(outputs)
    natural_browser(outputs)
    write_json(outputs[-1]/'report_complete.json',dict(status='complete',rounds=5,
        candidate_rows=len(table),natural_metric_rows=len(natural),goal_passed=any(r['goal_passed'] for r in checks)))


def natural_browser(outputs):
    from experiments.head_state_readout.audit import browser
    ledger = read_json(outputs[0]/'address_token_ledger.json')['rows']
    comparisons = [('old_association',outputs[0],'association'),
        ('backward_JS',outputs[0],'relation_incremental'),
        ('binding_state',outputs[1],'binding_conditional'),
        ('forward_JS_tail',outputs[3],'relation_tail'),
        ('unified_max',outputs[4],'omnibus_max'),
        ('unified_mean',outputs[4],'omnibus_mean')]
    selected = [r for r in ledger if r['role']=='natural']
    display = []
    for key in dict.fromkeys(r['key'] for r in selected):
        tokens = []
        arrays = {label:np.load(output/key/'scores.npz')[method] for label,output,method in comparisons}
        limits = {label:read_json(output/'thresholds.json')['QA'][method] for label,output,method in comparisons}
        for row in selected:
            if row['key']!=key:
                continue
            position,target = row['position'],row['label']
            scores = {label:float(value[position]) for label,value in arrays.items()}
            status = {label:('TP' if target else 'FP') if value>limits[label] else ('FN' if target else 'TN')
                      for label,value in scores.items()}
            tokens.append(dict(position=position,text=row['text'],label=target,scores=scores,status=status))
        display.append(dict(key=key,task='QA / reviewed local tokens only',tokens=tokens))
    directory = outputs[-1]/'natural_local'
    directory.mkdir(exist_ok=True)
    browser(directory,display,[r[0] for r in comparisons])
    page = directory/'TOKEN_AUDIT.html'
    page.write_text(page.read_text().replace('官方标签仅用于冻结后显示；“正常”表示官方未标错。',
        '只显示人工核验过的31个局部token；其余位置未知。'))


def plot(outputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    comparisons = [(outputs[0],'source_route_fixed','Old fixed'),
        (outputs[0],'association_fused','Old head states'),
        (outputs[0],'relation_conditional_fused','Backward JS + fixed'),
        (outputs[2],'relation_conditional_fused','Message MMD + fixed'),
        (outputs[3],'relation_conditional_fused','Forward JS + fixed'),
        (outputs[4],'omnibus_max_fused','Unified max + fixed')]
    with (outputs[0]/'tokens.csv').open() as stream:
        tokens = list(csv.DictReader(stream))
    keys = list(dict.fromkeys(row['key'] for row in tokens))
    records = {r['key']:r for r in read_json(outputs[0]/'manifest.json')['records']}
    fig,axes = plt.subplots(len(keys),1,figsize=(15,15),constrained_layout=True)
    colors = ListedColormap(['#eeeeee','#66b88a','#e67f76','#ad88ca'])
    for ax,key in zip(axes,keys):
        selected = [r for r in tokens if r['key']==key]
        count = max(int(r['token']) for r in selected)+1
        array = np.zeros((len(comparisons),count))
        for i,(output,method,label) in enumerate(comparisons):
            score = np.load(output/key/'scores.npz')[method]
            threshold = read_json(output/'thresholds.json')[records[key]['task']][method]
            for row in selected:
                position,target = int(row['token']),int(row['label'])
                alarm = score[position]>threshold
                array[i,position] = (1 if target else 2) if alarm else (3 if target else 0)
        ax.imshow(array,aspect='auto',interpolation='nearest',cmap=colors,vmin=0,vmax=3,
                  extent=(-.5,count-.5,len(comparisons)-.5,-.5))
        ax.set_yticks(range(len(comparisons)),[r[2] for r in comparisons],fontsize=8)
        ax.set_title(key+' / '+records[key]['task'],loc='left',fontsize=10)
        ax.set_xlabel('Original answer token position (0-based)',fontsize=8)
    fig.suptitle('Fixed unlabeled thresholds: green TP | red FP | purple FN | gray TN',fontsize=13)
    fig.savefig(outputs[-1]/'token_positions.png',dpi=170)
    fig.savefig(outputs[-1]/'token_positions.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
