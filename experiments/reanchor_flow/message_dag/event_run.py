"""Scan every native lookback event, trace its messages, then join labels offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from tqdm.auto import tqdm

from ..attention_audit_run import analysis_manifest
from ..attention_rhythm_report import save_json
from ..message_lineage import CheckpointWeights
from .cache import MASK_POLICY, NativeCache, read_trace
from .events import EventConfig, scan
from .event_trace import trace_events
from .selection import read_labels
from .transport import CutRecorder, prepare_local_readout

SCHEMA = 2


@contextmanager
def atomic_path(destination):
    """Each write owns a temporary name on the destination filesystem."""
    destination = Path(destination)
    with NamedTemporaryFile(dir=destination.parent,prefix=f'.{destination.stem}.',
                            suffix=destination.suffix,delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_writer(output):
    """One writer for the whole run, including scan, resume and evaluation."""
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    # Keep the inode: unlinking a lock file permits two independent locks.
    with (output/'.event_run.lock').open('a+',encoding='utf-8') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or 'owner not yet recorded'
            raise RuntimeError(f'output already has an active writer ({owner}): {output}; '
                               'finish/stop that run or choose another --output') from exc
        handle.seek(0);handle.truncate()
        handle.write(f'host={socket.gethostname()} pid={os.getpid()}\n');handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def save_index(output, manifest):
    with atomic_path(output/'index.json') as temporary:
        save_json(temporary,manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('experiments/reanchor_flow/outputs/attention_audit_v3'))
    p.add_argument('--output', type=Path, help='default AUDIT/lookback_events_v2 (v1 with --legacy-v1)')
    p.add_argument('--legacy-v1', action='store_true', help='resume original hop-only v1 results without computing v2 edges')
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
    p.add_argument('--state-cache-gib', type=float, default=1., help='CPU tangent-state budget for sharing each layer across GPU batches; 0 disables grouping')
    p.add_argument('--profile', action='store_true', help='record synchronized stage times and CUDA peak memory per sample')
    p.add_argument('--contrasts', type=Path, help='JSON: split/task/id -> [{target, positive_id, negative_id}]')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--bootstrap', type=int, default=200)
    return p


def atomic_npz(path, **values):
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, **values)


def sample_key(entry):
    return f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}"


def event_group_size(rows, hidden_size, event_batch, state_cache_gib):
    per_event = 3*3*rows*hidden_size*4
    capacity = int(state_cache_gib*2**30)//per_event
    return max(event_batch,capacity//event_batch*event_batch)


def run(args):
    version = 1 if args.legacy_v1 else SCHEMA
    output = args.output or args.audit/f'lookback_events_v{version}'
    if output.resolve()==args.audit.resolve():
        raise ValueError('event output must differ from the native capture directory')
    if args.list_available and args.phase!='evaluate':
        return _run(args,output,version)  # read-only, no output or lock creation
    with output_writer(output):
        return _run(args,output,version)


def _run(args, output, version):
    import torch
    from .event_report import evaluate
    config = EventConfig(args.window,args.gain,args.local_floor)
    if min(args.query_chunk,args.event_batch,args.cpu_threads)<1 or args.bootstrap<0:
        raise ValueError('positive chunk/thread counts and nonnegative bootstrap required')
    if not np.isfinite(args.state_cache_gib) or args.state_cache_gib<0:
        raise ValueError('state-cache-gib must be finite and nonnegative')
    torch.set_num_threads(args.cpu_threads)
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
    settings = dict(schema=version,event=asdict(config),mask_policy=MASK_POLICY,model=model,contrasts=contrasts)
    request = dict(split=args.split,task=args.task,sample_id=sorted(args.sample_id))
    previous_path = output/'index.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous.get('event_settings',{}).get('schema')==1 and version!=1:
        raise ValueError('this output contains v1 results; add --legacy-v1 to resume them, '
                         'or use a new --output for v2 signed edges')
    if previous and (previous.get('event_settings')!=settings or previous.get('request')!=request):
        raise ValueError('event definition, checkpoint, contrasts or sample scope changed; choose another --output')
    manifest.update(event_settings=settings,request=request,native_audit=str(args.audit.resolve()),
                    labels_used_for_events=False,labels_used_for_dag=False)
    prior = {e['path']:e for e in previous.get('samples',[])}
    for e in manifest['samples']:
        e.update({k:v for k,v in prior.get(e['path'],{}).items() if k.startswith('event_') or k=='read_sites'})
        e['folder'] = 'samples/'+sample_key(e)
    save_index(output,manifest)
    print(f'lookback method v{version}: {output}'+
          ('; legacy hop-only resume, no v2 signed edges' if version==1 else '; signed V/K edges'),flush=True)
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
                if version>=2 and not (folder/f"edges_{int(found['row_position'][row])}.npz").exists():
                    pending.append(int(row))
            else: pending.append(int(row))
        e['event_traced'] = len(rows)-len(pending)
        save_index(output,manifest)
        print(f"{sample_key(e)}: {len(rows)} event positions / {len(sites)} read sites; "
              f"{len(pending)} untraced, no event selection budget",flush=True)
        if args.phase!='scan' and pending:
            if weights is None: weights = CheckpointWeights(model,args.device)
            with NativeCache(path,weights) as cache, ExitStack() as sample_stack:
                writers = sample_stack.enter_context(ThreadPoolExecutor(max_workers=args.cpu_threads))
                group_size = min(len(pending),event_group_size(cache.rows,weights.config['hidden_size'],args.event_batch,args.state_cache_gib))
                per_event = 3*3*cache.rows*weights.config['hidden_size']*4
                work_bytes = min(args.event_batch,len(pending))*per_event
                host_bytes = group_size*per_event if group_size>args.event_batch else 0
                print(f'  device={args.device}; working tangent {work_bytes/2**30:.3f} GiB; '
                      f'CPU shared states {host_bytes/2**30:.3f} GiB; group={group_size}, event batch={args.event_batch}; '
                      'operators/work buffers additional',flush=True)
                profile = {} if args.profile else None
                cuda = torch.device(args.device).type=='cuda'
                if cuda: torch.cuda.reset_peak_memory_stats(args.device)
                trace_started = perf_counter()
                readout = None
                with tqdm(total=len(pending),desc=f'{sample_key(e)} DAG',unit='event',leave=False) as bar:
                    def progress(stage): bar.set_postfix_str(stage,refresh=True)
                    if version>=2:
                        pairs = sum(max(0,cache.rows-2-row)*(cache.rows-1-row)//2 for row in pending)
                        edge_bytes = 12*cache.layers*cache.heads*pairs
                        print(f'  all physical cut edges: about {edge_bytes/2**30:.3f} GiB uncompressed V/K/attention; streamed to NPZ',flush=True)
                        temporary = sample_stack.enter_context(TemporaryDirectory(prefix='local_readout_',dir=folder))
                        readout_path = Path(temporary)/'readout.npz'
                        prepare_local_readout(cache,readout_path,query_chunk=args.query_chunk,
                                              contrasts=contrasts.get(sample_key(e)),progress=progress)
                        readout = sample_stack.enter_context(np.load(readout_path,allow_pickle=False))
                    for begin in range(0,len(pending),group_size):
                        batch = sites[np.isin(sites[:,2],pending[begin:begin+group_size])]
                        with ExitStack() as stack:
                            recorders = {}
                            if readout is not None:
                                for row in np.unique(batch[:,2]):
                                    recorders[int(row)] = CutRecorder(folder/f"edges_{int(found['row_position'][row])}.npz",cache,row)
                                    stack.callback(recorders[int(row)].close)
                            results = trace_events(cache,batch,window=config.window,query_chunk=args.query_chunk,
                                                   contrasts=contrasts.get(sample_key(e)),progress=progress,
                                                   cut_readout=readout,cut_recorders=recorders,
                                                   event_batch=args.event_batch,profile=profile)
                        save_started = perf_counter()
                        for result in results:
                            result['settings'] = found['settings']
                        def save_event(result):
                            atomic_npz(folder/f"event_{int(result['event_position'])}.npz",**result)
                        list(writers.map(save_event,results))
                        e['event_traced'] += len(results)
                        bar.update(len(results));save_index(output,manifest)
                        if profile is not None:
                            profile['checkpoint_seconds'] = profile.get('checkpoint_seconds',0.)+perf_counter()-save_started
                        del results,recorders,result
                execution = dict(device=args.device,event_batch=args.event_batch,event_group=group_size,
                                 cpu_state_gib=host_bytes/2**30,trace_seconds=perf_counter()-trace_started)
                if cuda:
                    execution['cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated(args.device)/2**30
                    execution['cuda_peak_reserved_gib'] = torch.cuda.max_memory_reserved(args.device)/2**30
                if profile is not None: execution['profile'] = profile
                e['event_execution'] = execution
                print('  execution: '+json.dumps(execution,sort_keys=True),flush=True)
        # Labels are joined only after this sample's label-free construction.
        if path.with_suffix('.labels.npz').exists():
            atomic_npz(folder/'labels.npz',labels=read_labels(path,found))
        e['event_last_run_seconds'] = perf_counter()-started
        save_index(output,manifest)
    if args.phase in ('all','scan'): return evaluate(output,bootstrap=args.bootstrap)
    return manifest


if __name__=='__main__':
    run(parser().parse_args())
