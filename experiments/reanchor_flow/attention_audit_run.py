"""Full-cohort normal/hallucinated attention audit, one native pass per sample."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from urllib.parse import quote

import numpy as np
from tqdm.auto import tqdm

from .attention_audit import AuditConfig, SCHEMA, capture_audit, special_token_mask
from .attention_audit_stats import HORIZONS
from .attention_rhythm_run import MODEL, CACHE, SOURCE, TASKS
from .attention_rhythm_report import save_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('all', 'capture', 'analyze'), default='all')
    p.add_argument('--completed-only', action='store_true', help='analyze completed samples from an interrupted capture; keep the full resume index')
    p.add_argument('--model', type=Path, default=Path(MODEL))
    p.add_argument('--cache', type=Path, default=Path(CACHE))
    p.add_argument('--scans', type=Path)
    p.add_argument('--source-info', type=Path, default=Path(SOURCE))
    p.add_argument('--labels-from', type=Path, help='reuse v2 labels after checking the original token IDs and response start')
    p.add_argument('--output', type=Path, default=Path('experiments/reanchor_flow/outputs/attention_audit_v3'))
    p.add_argument('--split', choices=('all', 'train', 'test'), default='all')
    p.add_argument('--task', choices=(*TASKS, 'all'), default='all')
    p.add_argument('--samples-per-task', type=int, default=0, help='0 = all; no correctness-based capture selection')
    p.add_argument('--sample-id', action='append', default=[])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='bfloat16')
    p.add_argument('--query-chunk', type=int, default=32)
    p.add_argument('--top-k', type=int, default=8, help='display edges only; statistics and full history are unpruned')
    p.add_argument('--local-window', type=int, default=10)
    p.add_argument('--exclude-token-id', type=int, action='append', default=[])
    p.add_argument('--discard-states', action='store_true', help='keep signed write accounting but discard full vector states')
    p.add_argument('--full-attention', action='store_true', help='also retain every prompt-source attention row; can be very large')
    p.add_argument('--plan-only', action='store_true', help='verify all inputs and estimate storage without loading model weights')
    p.add_argument('--horizon', action='append', default=[], help='LO:HI; HI=0 means all observed future; first must be finite')
    p.add_argument('--match-window', type=int, default=32)
    p.add_argument('--onset-radius', type=int, default=8)
    p.add_argument('--plots-per-class', type=int, default=2, help='per split/task: both positive and fully negative answers')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    return p


def native_config(args):
    return AuditConfig(args.local_window, args.top_k, args.query_chunk, not args.discard_states, args.full_attention)


def prepare(args):
    """Read/validate every token input before spending the first GPU forward."""
    from transformers import AutoConfig, AutoTokenizer
    from research_dataset import open_research_dataset
    from .scan_dataset import ScanDataset
    from .subset_data import inspect_records, select_records, sample_tokens
    from .units import build_source_units

    config = native_config(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model_config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
    with args.source_info.open(encoding='utf-8') as stream:
        sources = {str(x['source_id']): x for x in map(json.loads, stream)}
    settings = {**asdict(config), 'schema': SCHEMA, 'model': str(args.model), 'dtype': args.dtype,
                'extra_special_ids': sorted(args.exclude_token_id),
                'tokenizer_special_ids': sorted(getattr(tokenizer, 'all_special_ids', []))}
    entries, sizes = [], dict(history=0, qk=0, states=0, full_attention=0, compact=0)
    groups = {}
    splits = ('train', 'test') if args.split == 'all' else (args.split,)
    tasks = TASKS if args.task == 'all' else (args.task,)
    for split in splits:
        dataset = ScanDataset(args.scans / split) if args.scans else open_research_dataset(
            args.cache / split, device='cpu', retain_embedded_labels=False)
        records = dataset.records if args.scans else inspect_records(dataset, source_info=sources)
        chosen = select_records(records, tasks=tasks, samples_per_task=args.samples_per_task,
                                seed=args.seed, sample_ids=tuple(args.sample_id))
        for record in tqdm(chosen, desc=f'preflight {split}', unit='sample'):
            if args.scans:
                scan = dataset.load(record.sample_id, fields=('token_ids',))
                ids, start = scan['token_ids'], scan.response_start
                full_count = scan.metadata['full_response_tokens']
            else:
                ids, start = sample_tokens(dataset, record.sample_id)
                ids = ids.numpy()
                full_count = len(ids) - start
            if len(ids) - start != full_count:
                raise ValueError(f'{split}/{record.sample_id}: input scan truncates response; supply full inputs')
            units = build_source_units(sources[record.source_id], tokenizer, ids, start)
            unit_id = np.full(len(ids), -1, np.int32)
            evidence, names, kinds = np.zeros(len(ids), bool), [], []
            for u, kind in enumerate(units.kind):
                if kind not in {'response', 'other_prompt'}:
                    mask = units.token_unit_id.numpy() == u
                    unit_id[:-1][mask] = len(names)
                    evidence[:-1] |= mask
                    names.append(str(units.name[u]) if hasattr(units, 'name') else f'{kind}:{u}')
                    kinds.append(kind)
            special = special_token_mask(tokenizer, ids, args.exclude_token_id)
            folder = args.output / split / record.task_type
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (quote(str(record.sample_id), safe='') + '.npz')
            inputs = dict(token_ids=ids, response_start=np.array(start), evidence_mask=evidence,
                          special_mask=special, source_unit_id=unit_id, unit_names=np.array(names),
                          unit_kinds=np.array(kinds), token_text=np.array([tokenizer.decode([int(t)]) for t in ids]),
                          sample_id=np.array(str(record.sample_id)), source_id=np.array(str(record.source_id)),
                          task_type=np.array(record.task_type), full_response_tokens=np.array(full_count),
                          generator_model=np.array(getattr(record, 'generator_model', 'unknown')))
            resumed = path.exists()
            if resumed:
                with np.load(path, allow_pickle=False) as stored:
                    if (json.loads(str(stored['settings'])) != settings
                            or any(not np.array_equal(stored[k], inputs[k]) for k in
                                   ('token_ids', 'response_start', 'evidence_mask', 'special_mask', 'source_unit_id'))):
                        raise ValueError(f'{path}: capture settings differ; use analyze or a new output directory')
                required = ['.history.npz', '.qk.npz'] + (['.states.npz'] if config.save_states else []) + (['.attention.npz'] if config.full_attention else [])
                if any(not path.with_suffix(s).exists() for s in required):
                    raise ValueError(f'{path}: required companion file is missing; restore it or use a new output')
            else:
                np.savez_compressed(path.with_suffix('.input.npz'), **inputs)
            e = dict(split=split, task_type=record.task_type, sample_id=str(record.sample_id),
                     source_id=str(record.source_id), path=str(path.relative_to(args.output)),
                     response_tokens=full_count, response_start=int(start), resumed=resumed)
            entries.append(e)
            group = split + '/' + record.task_type
            g = groups.setdefault(group, dict(samples=0, tokens=0, resumed=0))
            g['samples'] += 1; g['tokens'] += full_count; g['resumed'] += resumed
            l, h, d = model_config.num_hidden_layers, model_config.num_attention_heads, model_config.hidden_size
            r, n = full_count + 1, len(ids)
            sizes['history'] += 4*l*h*r*r
            sizes['full_attention'] += 4*l*h*r*n if config.full_attention else 0
            kv = model_config.num_key_value_heads
            sizes['qk'] += 4*l*(r*d + n*kv*(d//h))
            sizes['states'] += 4*(l*(4*r*d + n*kv*(d//h)) + r*d) if config.save_states else 0
            sizes['compact'] += 4*l*h*r*(32 + len(names) + 12*config.top_k)
    if not entries:
        raise ValueError('no samples selected')
    manifest = dict(audit_schema=SCHEMA, labels_used_for_capture=False, settings=settings,
                    config={'scans': str(args.scans) if args.scans else None, 'cache': str(args.cache)},
                    samples=entries, input_groups=groups, storage_uncompressed_bytes=sizes)
    if {e['source_id'] for e in entries if e['split']=='train'} & {e['source_id'] for e in entries if e['split']=='test'}:
        raise ValueError('train/test source IDs overlap; independent replication would be invalid')
    save_json(args.output / 'index.json', manifest)
    print('\nINPUTS VERIFIED', flush=True)
    for group, count in groups.items():
        print(f"{group:16s} samples={count['samples']} tokens={count['tokens']} resumed={count['resumed']}", flush=True)
    gib = 1024**3
    print('Storage before compression (not a runtime estimate): ' + ', '.join(f'{k}={v/gib:.1f} GiB' for k,v in sizes.items()), flush=True)
    print(f'Free disk={shutil.disk_usage(args.output).free/gib:.1f} GiB; NPZ uses compression. Full states are {"kept" if config.save_states else "discarded after signed accounting"}.', flush=True)
    print('One native forward per unfinished sample. Capture, saving, labels and analysis have separate progress.', flush=True)
    return manifest


def capture(args, manifest):
    import torch
    from transformers import AutoModelForCausalLM

    config, model = native_config(args), None
    progress = tqdm(manifest['samples'], desc='native audit capture', unit='sample')
    for e in progress:
        path = args.output / e['path']
        identity = f"{e['split']}/{e['task_type']}/{e['sample_id']}"
        if e['resumed']:
            progress.set_postfix_str(identity + ' cached')
            continue
        if model is None:
            print(f'Loading model on {args.device} ({args.dtype})', flush=True)
            model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                    torch_dtype=getattr(torch, args.dtype), attn_implementation='eager').to(args.device).eval()
        with np.load(path.with_suffix('.input.npz'), allow_pickle=False) as stored:
            inputs = dict(stored)
        def layer_progress(stage, done, total):
            progress.set_postfix_str(f'{identity} {stage} layer={done}/{total}', refresh=True)
        trace = capture_audit(model, inputs['token_ids'], int(inputs['response_start']),
                              inputs['evidence_mask'], inputs['special_mask'], inputs['source_unit_id'],
                              path, config, layer_progress)
        trace.update(inputs, settings=np.array(json.dumps(manifest['settings'])))
        progress.set_postfix_str(identity + ' saving', refresh=True)
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **trace)
        temp.replace(path)
        e['resumed'] = True
        save_json(args.output / 'index.json', manifest)
        del trace, inputs
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analysis_manifest(output, manifest, completed_only=False, required_suffixes=None, *, allow_empty=False):
    """Select committed captures without changing the index used for resuming."""
    settings = manifest['settings']
    suffixes = ['.npz', '.history.npz', '.qk.npz']
    if settings.get('save_states'):
        suffixes.append('.states.npz')
    if settings.get('full_attention'):
        suffixes.append('.attention.npz')
    if required_suffixes is not None:
        # Derived score evaluation may need only committed compact captures.
        suffixes = list(required_suffixes)
    samples, skipped, groups = [], [], {}
    for e in tqdm(manifest['samples'], desc='check completed samples', unit='sample'):
        path = output / e['path']
        missing = [str(path.with_suffix(s).relative_to(output)) for s in suffixes
                   if not path.with_suffix(s).is_file()]
        group = groups.setdefault(e['split']+'/'+e['task_type'],
                                 dict(planned_samples=0, completed_samples=0,
                                      planned_tokens=0, completed_tokens=0))
        group['planned_samples'] += 1
        group['planned_tokens'] += e['response_tokens']
        if missing:
            skipped.append(dict(path=e['path'], missing_files=missing))
        else:
            # The final NPZ is the completion marker, even if interruption
            # happened before updating the manifest's resumed flag.
            samples.append(dict(e))
            group['completed_samples'] += 1
            group['completed_tokens'] += e['response_tokens']
    for group, c in groups.items():
        print(f"{group:16s} completed={c['completed_samples']}/{c['planned_samples']} "
              f"tokens={c['completed_tokens']}/{c['planned_tokens']}", flush=True)
    if skipped and not completed_only:
        raise ValueError(f"{len(skipped)} captures are incomplete; use --phase analyze --completed-only to analyze completed samples, or resume capture")
    if not samples and not allow_empty:
        raise ValueError('no completed samples are available for analysis')
    coverage = dict(planned_samples=len(manifest['samples']), completed_samples=len(samples),
                    skipped_samples=len(skipped), partial=bool(skipped), groups=groups, skipped=skipped)
    return {**manifest, 'samples': samples, 'analysis_coverage': coverage}


def join_labels(args, manifest, *, save_index=True):
    from research_dataset import open_research_dataset
    from .scan_dataset import ScanDataset, ScanLabelStore
    config = manifest['config']
    scans_root = args.scans or (Path(config['scans']) if config.get('scans') else None)
    # Analyze defaults to the saved cache; an explicit override is installed by run().
    cache_root = args.cache
    stores = {}
    if args.labels_from is not None and not args.labels_from.is_dir():
        raise FileNotFoundError(f'--labels-from directory does not exist: {args.labels_from}')
    for e in tqdm(manifest['samples'], desc='join N/H labels', unit='sample'):
        path = args.output / e['path']
        input_path = path if path.exists() else path.with_suffix('.input.npz')
        with np.load(input_path,allow_pickle=False) as inputs:
            ids,start,special=inputs['token_ids'],int(inputs['response_start']),inputs['special_mask']
        destination = path.with_suffix('.labels.npz')
        labels=None
        if destination.exists():
            with np.load(destination, allow_pickle=False) as stored:
                labels = stored['labels']
        elif args.labels_from is not None:
            old=args.labels_from/e['path']
            if old.exists() and old.with_suffix('.labels.npz').exists():
                with np.load(old,allow_pickle=False) as stored:
                    if not np.array_equal(stored['token_ids'],ids) or int(stored['response_start'])!=start:
                        raise ValueError(f'{old}: prior labels refer to a different token sequence')
                with np.load(old.with_suffix('.labels.npz'),allow_pickle=False) as stored:
                    labels=stored['labels']
        if labels is None:
            split = e['split']
            if split not in stores:
                if scans_root:
                    ds = ScanDataset(scans_root / split)
                    stores[split] = (ds, ScanLabelStore(ds, dataset_root=cache_root / split))
                else:
                    stores[split] = (open_research_dataset(cache_root / split, device='cpu', retain_embedded_labels=True), None)
            ds, store = stores[split]
            if store is not None:
                labels = store.load(ds.load(e['sample_id'], fields=()))
            else:
                lookup = ds.prepare_evaluation_labels([e['sample_id']])
                sample = ds[e['sample_id']]
                try:
                    labels = lookup.response_labels(sample).detach().cpu().numpy()
                finally:
                    sample.release_attention()
        if len(labels) != e['response_tokens'] or not np.isin(labels, (-1, 0, 1)).all():
            raise ValueError(f'{destination}: labels must exactly cover all response tokens and use -1/0/1')
        if not destination.exists():
            np.savez_compressed(destination,labels=labels)
        ordinary=~special[start:]
        e.update(normal_tokens=int(((labels==0)&ordinary).sum()),
                 hallucinated_tokens=int(((labels==1)&ordinary).sum()),
                 unknown_tokens=int(((labels<0)&ordinary).sum()),special_targets=int((~ordinary).sum()))
    if save_index:
        save_json(args.output/'index.json',manifest)


def run(args):
    if args.completed_only and args.phase != 'analyze':
        raise ValueError('--completed-only is only valid with --phase analyze')
    if min(args.samples_per_task, args.plots_per_class, args.onset_radius) < 0 or min(args.match_window, args.cpu_threads) < 1:
        raise ValueError('invalid analysis/capture budget')
    horizons = tuple(tuple(map(int, v.split(':'))) for v in args.horizon) or HORIZONS
    if any(len(x)!=2 or x[0]<1 or (x[1] and x[1]<x[0]) for x in horizons) or not horizons[0][1]:
        raise ValueError('horizons must be LO:HI with LO>=1; the primary horizon must have a finite HI')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'analyze':
        manifest = json.loads((args.output / 'index.json').read_text())
        if manifest.get('audit_schema') != SCHEMA or manifest.get('labels_used_for_capture') is not False:
            raise ValueError('v3 analysis needs a v3 capture; old rhythm summaries cannot reconstruct missing edges')
        if args.cache == Path(CACHE) and manifest['config'].get('cache'):
            args.cache = Path(manifest['config']['cache'])
        manifest = analysis_manifest(args.output, manifest, args.completed_only)
    else:
        manifest = prepare(args)
        # Read-only label verification before any expensive model work; the
        # capture observer receives neither labels nor correctness selection.
        join_labels(args,manifest)
        for group in manifest['input_groups']:
            entries=[e for e in manifest['samples'] if e['split']+'/'+e['task_type']==group]
            print(group+' verified labels: '+', '.join(f'{k}={sum(e[k] for e in entries)}' for k in
                  ('normal_tokens','hallucinated_tokens','unknown_tokens','special_targets')),flush=True)
        if args.plan_only:
            return manifest
        capture(args, manifest)
    if args.phase == 'capture':
        return manifest
    if args.phase=='analyze':
        join_labels(args, manifest, save_index=False)
    from .attention_audit_report import summarize
    return summarize(args.output, manifest, horizons=horizons, match_window=args.match_window,
                     onset_radius=args.onset_radius, plots_per_class=args.plots_per_class,
                     cpu_threads=args.cpu_threads)


if __name__ == '__main__':
    run(parser().parse_args())
