"""Inspect frozen readouts against existing local labels; never feed labels to extraction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.path_conflict.paired_inputs import claim_trace
from .run import EVENT_CHANNELS, write_json


def evaluate(output, cases_path):
    frozen = json.loads((output / 'frozen.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    samples_path = Path(protocol['samples'])
    samples = [json.loads(line) for line in (samples_path / 'samples.jsonl').read_text().splitlines()]
    lookup = {(str(sample['source_id']), sample['seed']): sample for sample in samples}
    table = pd.read_csv(output / 'events.csv', dtype={'source_id': str})
    cases = json.loads(cases_path.read_text())
    summaries, neighborhoods = [], []
    for case in cases:
        for side in ('supported', 'unsupported'):
            record = lookup[case['source_id'], case[side]['seed']]
            _, span = claim_trace(samples_path, record, case[side], include_attention=False)
            rows = table[table.trace == record['trace']].copy()
            onset = rows[rows.position == span[0]].iloc[0].to_dict()
            onset.update(case=case['case_id'], side=side, claim=case[side]['target'],
                         span_start=span[0], span_stop=span[1])
            near = rows[(rows.position >= span[0] - 3) & (rows.position <= span[0] + 3)]
            onset['near_event_count'] = int(near.event_union.sum())
            onset['claim_event_count'] = int(rows[(rows.position >= span[0]) & (rows.position < span[1])].event_union.sum())
            onset['answer_event_fraction'] = float(rows[rows.valid].event_union.mean())
            for name in EVENT_CHANNELS:
                onset[name + '_within_answer_percentile'] = float((rows.loc[rows.valid, name] < onset[name]).mean())
            summaries.append(onset)
            near['case'] = case['case_id']
            near['side'] = side
            neighborhoods.append(near)
    pd.DataFrame(summaries).to_csv(output / 'reviewed_onsets.csv', index=False)
    pd.concat(neighborhoods).to_csv(output / 'reviewed_neighborhoods.csv', index=False)
    write_json(output / 'evaluation.json', dict(frozen=frozen, cases=2, local_claims=4,
        labels_read_after_freeze=True, feature_or_threshold_changes_after_evaluation=False,
        whole_answer_labels_available=False, detector_auroc=None,
        warning='Repeated exploratory cases, not independent validation. Other tokens are unlabeled.'))
    print(pd.DataFrame(summaries)[['case', 'side', 'entropy_nats', 'prompt_mass',
        'weighted_address_js', 'late_candidate_js', 'chosen_margin_final',
        'final_top1_settle_layer', 'event_union', 'near_event_count']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('experiments/path_conflict/paired_cases.json'))
    args = parser.parse_args()
    evaluate(args.output, args.cases)
