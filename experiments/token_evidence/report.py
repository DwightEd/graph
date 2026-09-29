"""Readable comparison and complete per-token audit, including unknown labels."""
import argparse
import csv
import html

import numpy as np
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json
from experiments.span_source_control.measure import MODEL


def comparison_table(summary):
    rows = ['GSM rows use step labels and step-mean scores; other rows use token labels.\n',
            '| Cohort | Method | AUROC | AP | TP | FP | First-error hits (token/step) | Normal answers with alarm |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort, values in summary.items():
        for method in ('base', 'base_matched', 'legacy_token_source', 'reset_cad_tail', 'full_cad_tail'):
            result = values[method]
            auc = f"{result['auroc']:.6f}" if result['auroc'] is not None else 'N/A'
            ap = f"{result['ap']:.6f}" if result['ap'] is not None else 'N/A'
            rows.append(f"| {cohort} | {method} | {auc} | {ap} | {result['tp']} | {result['fp']} | "
                        f"{result['first_error_hits']}/{result['error_answers']} | "
                        f"{result['normal_answer_false_alarms']}/{result['normal_answers']} |")
    return '\n'.join(rows) + '\n'


def token_page(output, records, evaluated, thresholds, tokenizer):
    body = ['<meta charset="utf-8"><title>Token evidence audit</title>',
        '<style>body{max-width:1150px;margin:30px auto;font:16px/1.9 sans-serif} '
        '.token{white-space:pre-wrap;padding:1px} .alarm{background:#ffe2a8} '
        '.error{border-bottom:3px solid #b71c1c} .unknown{color:#777} '
        'section{margin-bottom:35px} code{font-size:13px}</style>',
        '<h1>Unsupervised token evidence: complete audit</h1>',
        '<p>Orange = frozen primary alarm; red underline = known error; gray = unknown label. '
        'Hover for score and automatic alternative. GSM token truth remains unknown; its labels and '
        'calibration are step-level, so no token alarms are shown for GSM. '
        'No scores are averaged across neighboring tokens.</p>']
    for row in records:
        if row['role'] == 'dev':
            continue
        with np.load(output / row['key'] / 'scores.npz') as saved:
            scores = saved['reset_cad_tail']
        with np.load(output / row['key'] / 'candidates.npz') as saved:
            candidates = saved['reset_ids'][:, 16]
        limit = thresholds[row['task']]['reset_cad_tail']
        labels = {int(item['position']): int(item['gold']) for item in evaluated
                  if item['key'] == row['key']} if row['dataset'] == 'ragtruth' else {}
        body.append(f"<section><h2>{html.escape(row['key'])} — {row['task']} / {row['role']}</h2>")
        for index, text in enumerate(row['response']['token_text']):
            gold = labels.get(index, -1)
            classes = ['token']
            if row['dataset'] == 'ragtruth' and scores[index] > limit:
                classes.append('alarm')
            if gold == 1:
                classes.append('error')
            if gold < 0:
                classes.append('unknown')
            alternative = tokenizer.decode([int(candidates[index])])
            threshold = f'{limit:.6f}' if row['dataset'] == 'ragtruth' else 'N/A (step calibration only)'
            title = f't={index}; score={scores[index]:.6f}; threshold={threshold}; q top={alternative!r}; gold={gold}'
            body.append(f'<span class="{" ".join(classes)}" title="{html.escape(title, quote=True)}">{html.escape(text)}</span>')
        body.append('</section>')
    (output / 'TOKEN_AUDIT.html').write_text('\n'.join(body))


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'evaluation_complete.json')
    summary = read_json(args.output / 'summary.json')
    (args.output / 'COMPARISON.md').write_text(comparison_table(summary))
    records = read_json(args.output / 'manifest.json')['records']
    with (args.output / 'units.csv').open() as stream:
        evaluated = list(csv.DictReader(stream))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    token_page(args.output, records, evaluated, read_json(args.output / 'thresholds.json'), tokenizer)


if __name__ == '__main__':
    main()
