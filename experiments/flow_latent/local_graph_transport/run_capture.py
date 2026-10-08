"""Collect all official QA answers without opening natural annotation files."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.probabilistic_detection.data import select_records
from .capture import capture_answer


CACHE = Path('outputs/native_support_ragtruth_all/source_first_v1')
OUTPUT = Path('outputs/supervised_local_transport_20261008')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def qa_records(cache):
    manifest = json.loads((cache / 'manifest.json').read_text())
    records = select_records(manifest, 'QA', 'train')
    records += select_records(manifest, 'QA', 'test')
    order = {'fit': 0, 'dev': 1, 'test': 2}
    records.sort(key=lambda row: (order[row['partition']], row['source_id'], row['id']))
    return manifest, records


def input_case(cache, record):
    source = json.loads((cache / record['source_file']).read_text())
    response = json.loads((cache / record['directory'] / 'response.json').read_text())
    return source, response


def capture_case(adapter, cache, record, layer, answer_limit=None, complete=False):
    source, response = input_case(cache, record)
    answer = response['answer_ids'][:answer_limit]
    return capture_answer(adapter, source['prompt_with_source'], answer,
        source['source_mask'], layer=layer, stop_after_layer=not complete)


def canaries(adapter, cache, record, layer, early):
    """Real 8B checks: selected-layer early exit and strict prefix equivalence."""
    full, full_audit = capture_case(adapter, cache, record, layer, complete=True)
    length = min(16, len(early['answer_ids']))
    prefix, prefix_audit = capture_case(adapter, cache, record, layer, answer_limit=length)
    names = ('nodes', 'query', 'local_attention', 'group_heads', 'group_mass')
    identity = {name: float(np.max(np.abs(early[name].astype(float) - full[name].astype(float))))
                for name in names}
    causal = {name: float(np.max(np.abs(early[name][:, :length + 1].astype(float) -
                                       prefix[name].astype(float)))) for name in names}
    if max(identity.values()) > .01 or max(causal.values()) > .05:
        raise ValueError(f'8B capture canary failed: {identity}, {causal}')
    return dict(early_vs_full=identity, prefix_vs_bulk=causal,
                full_audit=full_audit, prefix_audit=prefix_audit,
                extra_full_forwards=2, extra_layer_stop_forwards=2)


def save_case(output, record, arrays, audit):
    directory = output / 'capture' / record['id']
    directory.mkdir(parents=True)
    np.savez(directory / 'arrays.npz', **arrays)
    write_json(directory / 'record.json', dict(record, audit=audit))


def freeze_protocol(output, cache, manifest, records, layer):
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema='qa_selected_layer_transport_v1', layer=layer, local_width=8,
        source_cache=str(cache.resolve()), model=manifest['model'], records=records,
        collection_opens_annotation_files=False, natural_label_fits=0,
        timing='T+1 rows: prechoice[:-1], posttoken[1:]',
        scope='native blocks0..selected; original positions and complete causal text',
        historical_test_exposure=True, archival='FP16 full coordinates, not PCA')
    path = output / 'CAPTURE_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Existing capture protocol differs; preserve it and choose a new output')
    else:
        write_json(path, protocol)
        snapshot = output / 'capture_code'
        snapshot.mkdir()
        dependencies = [Path(__file__), Path(__file__).with_name('capture.py'),
            Path(__file__).with_name('messages.py'), Path('experiments/decision_risk_flow/run.py'),
            Path('experiments/decision_risk_flow/precision.py'),
            Path('experiments/flow_latent/provenance_joint_state/measure.py'),
            Path('experiments/token_backtrace/grounded_projection.py')]
        for source in dependencies:
            destination = snapshot / source.relative_to(Path.cwd()) if source.is_absolute() else snapshot / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        write_json(output / 'CAPTURE_CODE.json', {str(p): file_hash(p) for p in dependencies})
    return protocol


def collect(args):
    manifest, records = qa_records(args.cache)
    freeze_protocol(args.output, args.cache, manifest, records, args.layer)
    selected = records[:args.limit] if args.limit else records
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = ModelAdapter(load_model(manifest['model']))
    fresh = 0
    for index, record in enumerate(selected):
        directory = args.output / 'capture' / record['id']
        if (directory / 'record.json').exists():
            continue
        case_start = time.time()
        arrays, audit = capture_case(adapter, args.cache, record, args.layer)
        for world in audit['worlds']:
            if world['head_reconstruction_max_abs'] > .001 or world['state_equation_max_abs'] > .001:
                raise ValueError(f'Native layer reconstruction failed: {world}')
            if world['forbidden_attention_max_abs'] != 0:
                raise ValueError('Source mask leaked attention')
        if not (args.output / 'CANARIES.json').exists():
            write_json(args.output / 'CANARIES.json', canaries(adapter, args.cache, record, args.layer, arrays))
        audit['capture_seconds'] = time.time() - case_start
        save_case(args.output, record, arrays, audit)
        fresh += 1
        if index < 4 or (index + 1) % 30 == 0:
            print(f"CAPTURE {index + 1}/{len(selected)} new={fresh} seconds={time.time()-started:.1f} "
                  f"id={record['id']} tokens={record['tokens']} case={audit['capture_seconds']:.2f}", flush=True)
        del arrays
    ledger = dict(requested_answers=len(selected), new_answers=fresh,
        completed_answers=sum((args.output / 'capture' / r['id'] / 'record.json').exists() for r in records),
        total_answers=len(records), new_layer_stop_forwards=2 * fresh,
        wall_seconds=time.time() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        full_qa_complete=len(selected) == len(records), layer=args.layer)
    write_json(args.output / ('CAPTURE_COMPLETE.json' if not args.limit else 'CAPTURE_PILOT.json'), ledger)
    print(json.dumps(ledger), flush=True)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--limit', type=int)
    collect(parser.parse_args())


if __name__ == '__main__':
    main()
