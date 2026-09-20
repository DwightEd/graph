"""Does a marked hallucination onset have an unusual earlier look-back change?"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inputs import AuditInputs, default_observer_tokenizer
from .onset_changes import before_peaks, head_changes
from .onset_report import summarize_onsets
from .onset_windows import compare_window, normal_positions, onset_metadata, peak_location
from .run import ROOT


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'attention/llama31_8b')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--tokenizer')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--tasks', nargs='+', choices=['QA', 'Summary', 'Data2txt'])
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--heads', nargs='+', type=int)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--local-window', type=int, default=10)
    parser.add_argument('--position-gap', type=float, default=.25)
    parser.add_argument('--minimum-controls', type=int, default=20)
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--minimum-gain', type=float, default=.05)
    parser.add_argument('--output', type=Path, default=Path('outputs/pre_onset_audit'))
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def prepare(args):
    from transformers import AutoTokenizer

    args.tokenizer = args.tokenizer or default_observer_tokenizer(args.cache.resolve())
    if args.tokenizer is None:
        raise ValueError('--tokenizer must identify the observer tokenizer to exclude ALL special IDs')
    if min(args.window, args.local_window, args.minimum_controls) < 1:
        raise ValueError('window, local-window and minimum-controls must be positive')
    if not 0 < args.quantile < 1 or args.minimum_gain < 0 or not 0 < args.position_gap <= 1:
        raise ValueError('invalid quantile, minimum-gain or position-gap')
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ('resume', 'output')}
    settings['version'] = 'all-onsets-specials-excluded-v1'
    path = args.output / 'settings.json'
    if args.output.exists():
        if not args.resume or json.loads(path.read_text()) != settings:
            raise ValueError('Use a new output or --resume with unchanged settings')
    else:
        args.output.mkdir(parents=True)
        path.write_text(json.dumps(settings, indent=2) + '\n')
    return AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)


def compare(values, onset, references, args):
    return compare_window(values, onset, references, args.minimum_controls,
                          args.quantile, args.minimum_gain)


def save_answer(destination, details, metadata, references, joint, channels, args):
    detail_table = pd.DataFrame(details)
    detail_table.drop(columns='text').to_csv(destination / 'heads.csv.gz', index=False)
    rows = [dict(meta, phase=phase, channels=len(channels),
                 **compare(values, meta['onset'], controls, args))
            for phase, values in joint.items() for meta, controls in zip(metadata, references)]
    finite = detail_table[np.isfinite(detail_table.gain)]
    winners = finite.loc[finite.groupby(['phase', 'onset']).gain.idxmax()]
    columns = ['phase', 'onset', 'layer', 'head', 'peak_target', 'peak_source', 'peak_source_gain']
    result = pd.DataFrame(rows).merge(winners[columns], on=['phase', 'onset'], how='left')
    result.to_csv(destination / 'onsets.csv', index=False)


def measure_answer(inputs, answer, special, args, destination):
    references = [normal_positions(answer, span.start, args.window, args.position_gap)
                  for span in answer.spans]
    metadata = [onset_metadata(answer, number, span, args.window, controls)
                for number, (span, controls) in enumerate(zip(answer.spans, references))]
    joint = dict(strict_before=np.full(len(answer.response_ids), -np.inf),
                 onset_decision=np.full(len(answer.response_ids), -np.inf))
    details = []
    channels = []
    for channel in inputs.channels(answer, args.layers, args.heads):
        channels.append((channel.layer, channel.head))
        gain, endpoint, endpoint_gain, retained = head_changes(channel, answer, special, args.local_window)
        phases = dict(strict_before=before_peaks(gain, args.window), onset_decision=gain)
        mass = dict(strict_before=-before_peaks(-retained, args.window), onset_decision=retained)
        for phase, values in phases.items():
            joint[phase] = np.maximum(joint[phase], values)
            for meta, controls in zip(metadata, references):
                details.append(dict(meta, phase=phase, layer=channel.layer, head=channel.head,
                    minimum_content_mass=float(mass[phase][meta['onset']]),
                    **compare(values, meta['onset'], controls, args),
                    **peak_location(gain, endpoint, endpoint_gain, meta['onset'], phase, args.window)))
    if not channels:
        raise ValueError(f'{answer.response_id}: no requested attention channels')
    save_answer(destination, details, metadata, references, joint, channels, args)
    return channels


def population_coverage(inputs, args):
    rows = []
    for identity, annotation in inputs.annotations.items():
        source = inputs.sources[str(annotation['source_id'])]
        task = source.get('task_type', source.get('task'))
        if annotation['split'] not in args.splits or (args.tasks and task not in args.tasks):
            continue
        rows.append(dict(id=identity, source_id=str(annotation['source_id']), task=task,
                         split=annotation['split'], generator=annotation['model'],
                         raw_annotations=len(annotation['labels']), cached=identity in inputs.groups))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'population_coverage.csv', index=False)
    return table.loc[table.cached, 'id'].tolist()


def main(argv=None):
    args = arguments(argv)
    tokenizer = prepare(args)
    inputs = AuditInputs(args.cache, args.dataset, args.index, args.tokenizer)
    inputs.binding.tokenizers[args.tokenizer] = tokenizer
    identities = population_coverage(inputs, args)
    inventory = []
    for identity in tqdm(identities, desc='all annotated onsets'):
        answer = inputs.load_answer(identity)
        special = np.isin(answer.token_ids, tokenizer.all_special_ids)
        inventory.append(dict(id=identity, source_id=answer.source_id, spans=len(answer.spans),
                              special_tokens=int(special.sum()), response_tokens=len(answer.response_ids)))
        if not answer.spans:
            continue
        destination = args.output / 'samples' / identity
        if args.resume and (destination / 'onsets.csv').exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        measure_answer(inputs, answer, special, args, destination)
    pd.DataFrame(inventory).to_csv(args.output / 'inventory.csv', index=False)
    if any(row['spans'] for row in inventory):
        print(json.dumps(summarize_onsets(args.output), indent=2))
    else:
        raise ValueError('No annotated onsets found in the selected cache population')


if __name__ == '__main__':
    main()
