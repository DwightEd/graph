"""Post-score mechanism audit; reviewed labels only select reporting regions."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .operator_score import measurements


def summarize(output, base, record, positions):
    values, history, _ = measurements(output, base, record)
    with np.load(base / record['key'] / 'readouts.npz') as saved:
        raw = saved['measured'].reshape(1024, len(saved['token_ids']), -1)
        # Native confidence uses natural logarithms; display entropy in bits.
        confidence = saved['confidence'].copy()
        confidence[:, 0] /= np.log(2.)
        signed_prompt = (raw[:, :, 4] - raw[:, :, 5]).sum(0)
        signed_history = (raw[:, :, 6] - raw[:, :, 7]).sum(0)
    with np.load(output / record['key'] / 'scores.npz') as saved:
        score = saved['mechanism']
    rows = []
    for token in positions:
        rows.append(dict(token=int(token), entropy_bits=float(confidence[token, 0]),
            prompt_read_head_median=float(np.median(values[:, token, 0])),
            prompt_positive_deficit_head_median=float(np.median(values[:, token, 1])),
            relative_root_js_head_median=float(np.nanmedian(values[:, token, 2])),
            prompt_cancel_head_median=float(np.median(values[:, token, 3])),
            read_use_js_head_median=float(np.median(values[:, token, 4])),
            prompt_margin_derivative_sum=float(signed_prompt[token]),
            history_margin_derivative_sum=float(signed_history[token]),
            fisher_history_share=float(history[token, 0]), output_js_coefficient=float(history[token, 1]),
            score=float(score[token])))
    columns = [name for name in rows[0] if name != 'token']
    return dict(tokens=rows, mean={name: float(np.nanmean([row[name] for row in rows])) for name in columns},
                positive_prompt_margin_tokens=sum(row['prompt_margin_derivative_sum'] > 0 for row in rows),
                entropy_below_one_bit_tokens=sum(row['entropy_bits'] < 1 for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output / 'manifest.json')
    records = {row['key']: row for row in manifest['records']}
    base = Path(manifest['base'])
    evaluation = read_json(args.output / 'evaluation.json')
    main_method = read_json(args.output / 'scores_frozen.json')['main']
    report = dict(posthoc=True, labels_only_for_reporting=True, detector_changed=False,
        caveat='Head medians and signed sums below are display summaries; original head/key arrays remain intact. A positive derivative is not factual support.',
        natural=[], regression_spans=[])
    for row in evaluation['natural_local']:
        if row['method'] == main_method:
            report['natural'].append(dict(case=row['case'], side=row['side'], key=row['key'],
                **summarize(args.output, base, records[row['key']], row['positions'])))
    for row in evaluation['spans']:
        if row['method'] == main_method:
            report['regression_spans'].append(dict(key=row['key'], start=row['start'], stop=row['stop'],
                detected=row['detected'], **summarize(args.output, base, records[row['key']], list(range(row['start'], row['stop'])))))
    write_json(args.output / 'mechanism_diagnostics.json', report)


if __name__ == '__main__':
    main()
