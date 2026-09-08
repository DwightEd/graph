"""Scan every native lookback event, trace its messages, then join labels offline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from ..attention_audit_run import analysis_manifest
from ..attention_rhythm_report import save_json
from ..message_lineage import CheckpointWeights
from .cache import MASK_POLICY, NativeCache, read_trace
from .events import EventConfig, scan
from .event_trace import trace_events
from .selection import read_labels

SCHEMA = 1


def save_index(output, manifest):
    temporary = output/'index.tmp.json'
    save_json(temporary,manifest)
    temporary.replace(output/'index.json')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('experiments/reanchor_flow/outputs/attention_audit_v3'))
    p.add_argument('--output', type=Path, help='default AUDIT/lookback_events_v1')
    p.add_argument('--phase', choices=('all','scan','trace','evaluate'), default='all')
    p.add_argument('--completed-only', action='store_true', help='disclose and skip missing native captures')
    p.add_argument('--list-available', action='store_true', help='coverage only; no model loading or output writes')
    p.add_argument('--model', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--split', choices=('all','train','test'), default='all')
    p.add_argument('--task', choices=('all','QA','Summary','Data2txt'), default='all')
    p.add_argument('--sample-id', action='append', default=[])
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--gain', type=float, default=.10)
    p.add_argument('--local-floor', type=float, default=.50)
    p.add_argument('--query-chunk', type=int, default=8)
    p.add_argument('--event-batch', type=int, default=2, help='position events per GPU batch; never a selection budget')
    p.add_argument('--contrasts', type=Path, help='JSON: split/task/id -> [{target, positive_id, negative_id}]')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--bootstrap', type=int, default=200)
    return p


def atomic_npz(path, **values):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def sample_key(entry):
    return f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}"


def run(args):
    import torch
    from .event_report import evaluate
    config = EventConfig(args.window,args.gain,args.local_floor)
    if min(args.query_chunk,args.event_batch,args.cpu_threads)<1 or args.bootstrap<0:
        raise ValueError('positive chunk/thread counts and nonnegative bootstrap required')
    torch.set_num_threads(args.cpu_threads)
    output = args.output or args.audit/'lookback_events_v1'
    if output.resolve()==args.audit.resolve():
        raise ValueError('event output must differ from the native capture directory')
    if args.phase=='evaluate':
        return evaluate(output,bootstrap=args.bootstrap)
    original = json.loads((args.audit/'index.json').read_text())
    if original.get('audit_schema')!=3:
        raise ValueError('lookback events require full v3 Q/K and history captures, not scalar/top-k summaries')
    def selected(e):
        return ((args.split=='all' or e['split']==args.split)
                and (args.task=='all' or e['task_type']==args.task)
                and (not args.sample_id or str(e['sample_id']) in args.sample_id))
    requested = {**original,'samples':[e for e in original['samples'] if selected(e)]}
    suffixes = ('.npz','.qk.npz','.history.npz') + (() if args.phase=='scan' else ('.states.npz',))
    manifest = analysis_manifest(args.audit,requested,completed_only=True,
                                 required_suffixes=suffixes,allow_empty=True)
    if args.list_available: return manifest
    coverage = manifest['analysis_coverage']
    if coverage['partial'] and not args.completed_only:
        raise ValueError(f"{coverage['completed_samples']}/{coverage['planned_samples']} requested captures complete; "
                         'resume native capture or explicitly add --completed-only. See --list-available.')
    if not manifest['samples']:
        raise ValueError('no complete captures in the requested scope; --list-available reports missing files')
    model = str(args.model or original['settings'].get('model',''))
    contrasts = json.loads(args.contrasts.read_text()) if args.contrasts else {}
    settings = dict(schema=SCHEMA,event=asdict(config),mask_policy=MASK_POLICY,model=model,contrasts=contrasts)
    request = dict(split=args.split,task=args.task,sample_id=sorted(args.sample_id))
    previous_path = output/'index.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous and (previous.get('event_settings')!=settings or previous.get('request')!=request):
        raise ValueError('event definition, checkpoint, contrasts or sample scope changed; choose another --output')
    manifest.update(event_settings=settings,request=request,native_audit=str(args.audit.resolve()),
                    labels_used_for_events=False,labels_used_for_dag=False)
    prior = {e['path']:e for e in previous.get('samples',[])}
    for e in manifest['samples']:
        e.update({k:v for k,v in prior.get(e['path'],{}).items() if k.startswith('event_') or k=='read_sites'})
        e['folder'] = 'samples/'+sample_key(e)
    output.mkdir(parents=True,exist_ok=True)
    save_index(output,manifest)
    weights = None
    for e in tqdm(manifest['samples'],desc='lookback events',unit='sample'):
        path,folder = args.audit/e['path'],output/e['folder']
        folder.mkdir(parents=True,exist_ok=True)
        scan_path = folder/'scan.npz'
        started = perf_counter()
        if scan_path.exists():
            with np.load(scan_path,allow_pickle=False) as stored: found = dict(stored)
            if str(found['settings'])!=json.dumps(settings,sort_keys=True) or not np.array_equal(found['token_ids'],read_trace(path)['token_ids']):
                raise ValueError(f'{scan_path}: capture identity/configuration changed')
        else:
            with tqdm(total=original['settings'].get('num_hidden_layers',0) or None,desc=sample_key(e),unit='layer',leave=False) as bar:
                def progress(stage):
                    bar.set_postfix_str(stage,refresh=False);bar.update()
                found = scan(path,config,chunk=args.query_chunk,device=args.device,progress=progress)
            found['settings'] = np.array(json.dumps(settings,sort_keys=True))
            atomic_npz(scan_path,**found)
        sites = found['event_index']
        rows = np.unique(sites[:,2])
        e.update(event_rows=rows.tolist(),event_count=len(rows),read_sites=len(sites))
        pending = []
        for row in rows:
            destination = folder/f"event_{int(found['row_position'][row])}.npz"
            if destination.exists():
                with np.load(destination,allow_pickle=False) as saved:
                    if (str(saved['settings'])!=str(found['settings'])
                            or not np.array_equal(saved['event_sites'],sites[sites[:,2]==row])):
                        raise ValueError(f'{destination}: saved event identity/configuration changed')
            else: pending.append(int(row))
        e['event_traced'] = len(rows)-len(pending)
        save_index(output,manifest)
        print(f"{sample_key(e)}: {len(rows)} event positions / {len(sites)} read sites; "
              f"{len(pending)} untraced, no event selection budget",flush=True)
        if args.phase!='scan' and pending:
            if weights is None: weights = CheckpointWeights(model,args.device)
            with NativeCache(path,weights) as cache:
                state_bytes = 3*3*min(args.event_batch,len(pending))*cache.rows*weights.config['hidden_size']*4
                print(f'  tangent state alone {state_bytes/2**30:.3f} GiB; operators and work buffers are additional',flush=True)
                with tqdm(total=len(pending),desc=f'{sample_key(e)} DAG',unit='event',leave=False) as bar:
                    for begin in range(0,len(pending),args.event_batch):
                        batch = sites[np.isin(sites[:,2],pending[begin:begin+args.event_batch])]
                        def progress(stage): bar.set_postfix_str(stage,refresh=True)
                        results = trace_events(cache,batch,window=config.window,query_chunk=args.query_chunk,
                                               contrasts=contrasts.get(sample_key(e)),progress=progress)
                        for result in results:
                            result['settings'] = found['settings']
                            atomic_npz(folder/f"event_{int(result['event_position'])}.npz",**result)
                            e['event_traced'] += 1
                        bar.update(len(results));save_index(output,manifest)
        # Labels are joined only after this sample's label-free construction.
        if path.with_suffix('.labels.npz').exists():
            atomic_npz(folder/'labels.npz',labels=read_labels(path,found))
        e['event_last_run_seconds'] = perf_counter()-started
        save_index(output,manifest)
    if args.phase in ('all','scan'): return evaluate(output,bootstrap=args.bootstrap)
    return manifest


if __name__=='__main__':
    run(parser().parse_args())
