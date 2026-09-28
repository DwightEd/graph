"""Test necessary-condition stories on frozen readouts and reviewed local spans.

This is label-using mechanism analysis, not fitting or evaluating a new detector.
Unknown answer regions remain unknown. All local labels are read after the old
annotation-free feature freeze.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.path_conflict.paired_inputs import claim_trace
from .run import write_json


def read_local_spans(output, cases_path):
    protocol = json.loads((output / 'protocol.json').read_text())
    json.loads((output / 'frozen.json').read_text())
    samples_path = Path(protocol['samples'])
    samples = [json.loads(line) for line in (samples_path / 'samples.jsonl').read_text().splitlines()]
    lookup = {(str(row['source_id']), row['seed']): row for row in samples}
    tokens = pd.read_csv(output / 'events.csv', dtype={'source_id': str})
    tables = []
    for case in json.loads(cases_path.read_text()):
        for side in ('supported', 'unsupported'):
            record = lookup[case['source_id'], case[side]['seed']]
            _, (start, stop) = claim_trace(samples_path, record, case[side], include_attention=False)
            selected = tokens[(tokens.trace == record['trace']) & tokens.valid &
                              (tokens.position >= start) & (tokens.position < stop)].copy()
            selected['case'] = case['case_id']
            selected['side'] = side
            selected['local_label'] = int(side == 'unsupported')
            selected['offset'] = selected.position - start
            tables.append(selected)
    return pd.concat(tables, ignore_index=True)


def attach_writes(tokens, output):
    tables = []
    for path in sorted((output / 'messages').glob('*.csv')):
        layers = pd.read_csv(path)
        fields = ['prompt_net', 'history_net', 'special_net', 'mlp_direct',
                  'prompt_positive', 'prompt_negative_magnitude']
        summed = layers.groupby('position')[fields].sum().reset_index()
        summed['trace'] = path.stem + '.npz'
        tables.append(summed)
    return tokens.merge(pd.concat(tables), on=['trace', 'position'], how='left', validate='one_to_one')


def attach_native_margins(tokens, output):
    """Use original logits, not the numerically different FP32 logit lens."""
    protocol = json.loads((output / 'protocol.json').read_text())
    samples_path = Path(protocol['samples'])
    tokens = tokens.copy()
    for trace, rows in tokens.groupby('trace', sort=False):
        with np.load(samples_path / trace, allow_pickle=False) as saved:
            chosen = saved['token_ids'][int(saved['prompt_length']):]
            alternate_column = (saved['top_ids'][:, 0] == chosen).astype(int)
            alternate = saved['top_logits'][np.arange(len(chosen)), alternate_column]
            margin = saved['chosen_logit'] - alternate
        tokens.loc[rows.index, 'chosen_margin_native'] = margin[rows.position.to_numpy()]
    return tokens


def count_conditions(tokens):
    rows = []
    for (case, side), selected in tokens.groupby(['case', 'side'], sort=False):
        measured = selected[selected.prompt_net.notna()]
        row = dict(case=case, side=side, tokens=len(selected),
            native_entropy_below_one_bit=int((selected.entropy_nats < np.log(2)).sum()),
            chosen_beats_auto_alternative=int((selected.chosen_margin_native > 0).sum()),
            lens_native_sign_disagreements=int(((selected.chosen_margin_final > 0) !=
                                               (selected.chosen_margin_native > 0)).sum()),
            no_proposed_event=int((~selected.event_union).sum()),
            full_message_tokens=len(measured),
            prompt_promotes_chosen=int((measured.prompt_net > 0).sum()) if len(measured) else None,
            history_promotes_chosen=int((measured.history_net > 0).sum()) if len(measured) else None,
            entropy_bits_mean=float(selected.entropy_nats.mean() / np.log(2)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readouts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=Path, default=Path('experiments/path_conflict/paired_cases.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokens = read_local_spans(args.readouts, args.cases)
    tokens = attach_native_margins(tokens, args.readouts)
    tokens = attach_writes(tokens, args.readouts)
    tokens.to_csv(args.output / 'local_tokens.csv', index=False)
    counts = count_conditions(tokens)
    counts.to_csv(args.output / 'condition_counts.csv', index=False)
    write_json(args.output / 'complete.json', dict(status='complete',
        input=str(args.readouts.resolve()), sources=int(tokens.source_id.nunique()),
        local_claims=int(tokens.groupby(['case', 'side']).ngroups),
        known_local_tokens=len(tokens), classifier_fit=False, new_llm_forwards=0,
        margin='original pre-temperature chosen logit minus strongest nonchosen logit',
        labels_used='existing reviewed local spans, for analysis only',
        warning='Exploratory necessary-condition counterexamples; no full-answer labels or population inference.'))
    print(counts.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
