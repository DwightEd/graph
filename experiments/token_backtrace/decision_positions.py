"""Prepare gold-onset and explicitly audited operator diagnosis, not detection."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .grounded_projection_data import write_json
from .repetition_teacher import BASE


def first_matched_onsets(rows, pairs):
    selected = []
    for row in rows:
        for target, control in pairs[row['id']]:
            if target == 0 or not row['labels'][target-1]:
                selected.append(dict(id=row['id'], source_id=row['source_id'],
                    target=target, control=control, matching='exact_BPE',
                    kind='gold_onset', teacher_gap=0.))
                break
    return selected


def error_operators():
    # Official annotation/source inspection: wrong citation and wrong negation.
    return [dict(id='12297', source_id='14366', target=106, control=124,
        matching='context_only_not_BPE', kind='error_operator', rival_token=18, teacher_gap=0.),
        dict(id='12219', source_id='14353', target=223, control=242,
        matching='context_only_not_BPE', kind='error_operator', rival_token=3493, teacher_gap=0.)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--passage3', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    selected = error_operators()
    if args.passage3:
        selected = [dict(**case, source_passage=3) for case in selected]
    else:
        rows = torch.load(BASE / 'prepared.pt', weights_only=False)['test']
        pairs = json.loads((args.previous / 'matched_tokens.json').read_text())['strict']
        selected = first_matched_onsets(rows, pairs)+selected
    write_json(args.output / 'native_pairs.json', selected)
    for name in ('head_selection.json', 'teacher_tangents.npz'):
        shutil.copyfile(args.previous / name, args.output / name)


if __name__ == '__main__':
    main()
