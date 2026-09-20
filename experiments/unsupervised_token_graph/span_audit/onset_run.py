"""Discover reanchor nodes independently, then audit their relation to gold spans."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inputs import AuditInputs, default_observer_tokenizer
from .onset_detection import save_detection, scan_channels
from .onset_report import summarize, write_review
from .onset_windows import link_rows, position_rows, span_rows
from .run import ROOT
from .units import marked_spans, span_mask


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'attention/llama31_8b')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--tokenizer')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--review-archive', type=Path, help='existing extracted four-prefix route archive; reviewed claims only')
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--tasks', nargs='+', choices=['QA', 'Summary', 'Data2txt'])
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--heads', nargs='+', type=int)
    parser.add_argument('--window', type=int, default=8, help='association horizon; never used to select nodes')
    parser.add_argument('--local-window', type=int, default=10)
    parser.add_argument('--baseline-steps', type=int, default=3)
    parser.add_argument('--minimum-shift', type=float, default=.1)
    parser.add_argument('--output', type=Path, default=Path('outputs/reanchor_nodes_v2'))
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def prepare(args):
    if min(args.window, args.local_window, args.baseline_steps) < 1 or not 0 < args.minimum_shift <= 1:
        raise ValueError('windows must be positive and minimum-shift must be in (0,1]')
    if args.review_archive is None:
        args.tokenizer = args.tokenizer or default_observer_tokenizer(args.cache.resolve())
        if args.tokenizer is None:
            raise ValueError('--tokenizer is required to exclude all observer special IDs')
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ('resume', 'output')}
    settings['version'] = 'label-free-local-to-old-switch-v2'
    path = args.output / 'settings.json'
    if args.output.exists():
        if not args.resume or json.loads(path.read_text()) != settings:
            raise ValueError('Use a new output or --resume with unchanged settings')
    else:
        args.output.mkdir(parents=True)
        path.write_text(json.dumps(settings, indent=2) + '\n')


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


def save_associations(answer, arrays, nodes, destination, args):
    all_positions, all_spans = [], []
    for threshold, states in zip(arrays['thresholds'], arrays['node_states']):
        all_positions.append(position_rows(answer, states, args.window).assign(threshold=threshold))
        all_spans.append(span_rows(answer, states, args.window).assign(threshold=threshold))
    positions = pd.concat(all_positions, ignore_index=True)
    positions.to_csv(destination / 'positions.csv.gz', index=False)
    pd.concat(all_spans, ignore_index=True).to_csv(destination / 'spans.csv', index=False)
    primary = positions[positions.threshold.eq(args.minimum_shift)].drop(columns='node_token')
    nodes.merge(primary, on='target').to_csv(destination / 'nodes.csv', index=False)
    link_rows(answer, nodes, args.window).to_csv(destination / 'node_span_links.csv', index=False)
    metadata = dict(id=answer.response_id, source_id=answer.source_id, task=answer.task,
                    generator=answer.generator, split=answer.split,
                    annotation_origin='official_ragtruth_complete_response',
                    response_tokens=len(answer.response_ids), annotated_spans=len(answer.spans))
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def run_cache(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    inputs = AuditInputs(args.cache, args.dataset, args.index, args.tokenizer, include_labels=False)
    inputs.binding.tokenizers[args.tokenizer] = tokenizer
    for identity in tqdm(population_coverage(inputs, args), desc='whole-answer reanchor scan'):
        destination = args.output / 'samples' / identity
        if args.resume and (destination / 'metadata.json').exists():
            continue
        answer = inputs.load_answer(identity)
        special = np.isin(answer.token_ids, tokenizer.all_special_ids)
        text = [tokenizer.decode([int(token)], clean_up_tokenization_spaces=False) for token in answer.token_ids]
        channels = inputs.channels(answer, args.layers, args.heads)
        arrays, events, sources = scan_channels(channels, answer.token_ids, answer.prompt_length, special, args)
        nodes = save_detection(destination, arrays, events, sources, answer.token_ids, text, answer.prompt_length)
        answer.spans = marked_spans(answer.offsets, inputs.annotations[identity]['labels'])
        answer.error_mask = span_mask(len(answer.response_ids), answer.spans)
        save_associations(answer, arrays, nodes, destination, args)


def main(argv=None):
    args = arguments(argv)
    prepare(args)
    if args.review_archive is None:
        run_cache(args)
    else:
        from .onset_archive import run_archive
        run_archive(args)
    result = summarize(args.output, args.minimum_shift)
    write_review(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
