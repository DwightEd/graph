"""Read historical frozen LDA predictions, retain original thresholds and node indices."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.gsm8k_recurrence.measure import write_json

LDA = Path('/share/home/tm902089733300000/a903202310/lys/.Trash-0/files/charm_structure_audit_qa/QA/seed_0/audit_lda_prompt')
CASES = Path('outputs/probabilistic_detection_20260928_full/cases/report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    thresholds = json.loads((LDA/'thresholds.json').read_text())
    with gzip.open(LDA/'test_scores.csv.gz','rt') as stream:
        rows = list(csv.DictReader(stream))
    summary,errors = {},[]
    for name in ('full','head_contrast','layer_mean'):
        score = np.array([float(r[name]) for r in rows])
        y = np.array([int(r['gold']) for r in rows])
        alarm = score>thresholds[name]
        summary[name] = dict(auroc=float(roc_auc_score(y,score)),threshold=thresholds[name],
            tp=int(sum(alarm&(y==1))),fp=int(sum(alarm&(y==0))),fn=int(sum(~alarm&(y==1))),
            answers=len({r['id'] for r in rows}),tokens=len(rows),positive=int(y.sum()),by_role={})
        for role in ('answer_first','later_onset','continuation','normal'):
            selected = np.array([r['role']==role for r in rows])
            summary[name]['by_role'][role] = dict(count=int(selected.sum()),alarms=int(sum(alarm&selected)))
        exit_mask = np.array([int(r['previous_gold'])==1 and int(r['gold'])==0 for r in rows])
        summary[name]['exit_normal'] = dict(count=int(exit_mask.sum()),alarms=int(sum(alarm&exit_mask)))
        for row,value,flag in zip(rows,score,alarm):
            if bool(int(row['gold']))==bool(flag):continue
            errors.append(dict(method=name,id=row['id'],source_id=row['source_id'],node=int(row['token']),
                text=row['text'],role=row['role'],previous_gold=int(row['previous_gold']),
                status='FN' if int(row['gold']) else 'FP',score=float(value),threshold=thresholds[name]))
    with (args.output/'lda_errors.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    write_json(args.output/'lda_summary.json',dict(methods=summary,
        provenance=str(LDA),scope='read-only historical QA frozen node scores, not retrained or new BPE captures',
        calibration='original independent normal calibration thresholds, strict gt'))
    lines=['# 历史监督LDA四个QA对照回答：全部FN/FP','',
           '使用原始full LDA及冻结阈值；位置是旧图节点索引，不能与新BPE索引直接混用。','']
    for key in ('11907','12015','12045','12219'):
        for status in ('FN','FP'):
            group=[r for r in errors if r['method']=='full' and r['id']==key and r['status']==status]
            lines += [f'## {key} {status} ({len(group)})','', '|节点|原文本|分数|','|---:|---|---:|']
            lines += [f"|{r['node']}|{repr(r['text'])}|{r['score']:.4f}|" for r in group]
            lines.append('')
    (args.output/'LDA_CASES.md').write_text('\n'.join(lines).rstrip()+'\n')
    report=json.loads(CASES.read_text())
    supervised=[]
    for case in report['cases']:
        for span in case.get('spans',[]):
            for method in ('selected_readout','selected_detector','trees'):
                if method not in span['methods']:continue
                supervised.append(dict(id=case['id'],partition=case['partition'],method=method,
                    text=span['text'],**span['methods'][method]))
    write_json(args.output/'supervised_known_spans.json',dict(rows=supervised,source=str(CASES),
        note='fit cases are in-sample; existing conditional/table models differ from head LDA'))
    print(summary,flush=True)


if __name__=='__main__':
    main()
