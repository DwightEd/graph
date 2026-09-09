"""Build message DAGs from completed v3 caches, then compare N/H offline."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from ..attention_audit_run import analysis_manifest
from ..attention_rhythm_report import save_json
from ..message_lineage import CheckpointWeights
from .artifacts import SOURCE_ALLOCATION, artifact_header, atomic_path
from .cache import (
    MASK_POLICY,
    NativeCache,
    read_trace,
    source_partition,
    target_positions,
)
from .graph import SCHEMA, Tape, blocks, prepare, trace_targets
from .selection import paired_plan, read_labels, target_details
from .structure import describe


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('experiments/reanchor_flow/outputs/attention_audit_v3'))
    p.add_argument('--output', type=Path, help='default AUDIT/message_dag_v2')
    p.add_argument('--phase', choices=('all','trace','evaluate'), default='all')
    p.add_argument('--completed-only', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--split', choices=('all','train','test'), default='all')
    p.add_argument('--task', choices=('all','QA','Summary','Data2txt'), default='all')
    p.add_argument('--sample-id', action='append', default=[])
    p.add_argument('--selection', choices=('uniform','paired'), default='uniform', help='paired: label-stratified content comparison, not population detection evaluation')
    p.add_argument('--samples-per-group', type=int, default=1, help='samples per split/task; paired mode needs an even total split equally N/H; 0=all eligible')
    p.add_argument('--targets-per-sample', type=int, default=4, help='target budget; paired mode uses three targets per N-H-N comparison; 0=all eligible')
    p.add_argument('--target', action='append', type=int, default=[], help='absolute token position; requires one selected sample')
    p.add_argument('--query-chunk', type=int, default=8)
    p.add_argument('--source-chunk', type=int, default=2)
    p.add_argument('--target-chunk', type=int, default=2, help='independent targets sharing one loaded layer')
    p.add_argument('--edge-budget', type=int, default=24, help='display edges per layer only; all edges participate in computation')
    p.add_argument('--mlp-rule', choices=('symmetric','up'), default='symmetric')
    p.add_argument('--scratch', type=Path, help='temporary source tape; default OUTPUT/work')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--list-available', action='store_true', help='list completed/planned full-state caches in every split/task; no model or graph output')
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--cpu-threads', type=int, default=4)
    return p


def atomic_npz(path, **values):
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, **values)


def ensure_labels(args, manifest):
    missing = [e for e in manifest['samples'] if not (args.audit/e['path']).with_suffix('.labels.npz').exists()]
    if not missing: return
    from ..attention_audit_run import join_labels
    from ..attention_audit_run import parser as audit_parser
    label_args = audit_parser().parse_args(['--phase','analyze','--output',str(args.audit)])
    if manifest.get('config',{}).get('cache'): label_args.cache=Path(manifest['config']['cache'])
    join_labels(label_args,{**manifest,'samples':missing},save_index=False)


def plan(args, original):
    # Only require files read by NativeCache, not optional full-attention dumps
    # or the old audit's derived outputs. A stale resumed flag is not a marker.
    native = analysis_manifest(args.audit, original, completed_only=True,
                               required_suffixes=('.npz', '.history.npz', '.qk.npz', '.states.npz'),
                               allow_empty=True)
    if args.list_available:
        return native

    def matches(e):
        return ((args.split == 'all' or e['split'] == args.split)
                and (args.task == 'all' or e['task_type'] == args.task)
                and (not args.sample_id or str(e['sample_id']) in args.sample_id))

    groups = defaultdict(list)
    for e in native['samples']:
        if matches(e):
            groups[e['split']+'/'+e['task_type']].append(e)
    requested = sum(matches(e) for e in original['samples'])
    completed = sum(map(len, groups.values()))
    scope = f'split={args.split}, task={args.task}, sample_ids={args.sample_id or "all"}'
    if not completed:
        available = ', '.join(f"{name}={c['completed_samples']}" for name,c in
                             native['analysis_coverage']['groups'].items() if c['completed_samples']) or 'none'
        hint = 'Use --list-available and adjust --split/--task/--sample-id to an available scope.'
        if not args.sample_id and any(args.task == 'all' or e['task_type'] == args.task for e in native['samples']):
            hint = f'Use --split all --task {args.task} to select completed samples across splits.'
        raise ValueError(f'No completed message-DAG samples match {scope} (0/{requested} planned). '
                         f'Available completed scopes: {available}. '
                         f'--completed-only still applies your split/task filters. {hint}')
    if completed < requested and not args.completed_only:
        raise ValueError(f'{scope}: only {completed}/{requested} captures are complete; '
                         'add --completed-only to use them, or resume capture.')
    if args.selection=='paired':
        if args.target: raise ValueError('--target is for explicit uniform selection; paired targets come from frozen N-H-N triples')
        entries=[e for group in groups.values() for e in group]
        ensure_labels(args,{**native,'samples':entries})
        selected,selection=paired_plan(args.audit,entries,args.samples_per_group,args.targets_per_sample)
        for e in selected:
            e['folder']=(Path('samples')/e['split']/e['task_type']/str(e['sample_id'])).as_posix()
        return {**native,'samples':selected,'selection':selection}
    selected = []
    for entries in groups.values():
        n = args.samples_per_group
        if n and len(entries)>n:
            entries = [entries[i] for i in np.linspace(0,len(entries)-1,n).round().astype(int)]
        selected.extend(entries)
    if args.target and len(selected)!=1:
        raise ValueError('--target requires exactly one selected sample')
    for e in selected:
        path=args.audit/e['path'];trace=read_trace(path)
        eligible = target_positions(trace,0)
        targets = np.asarray(args.target,int) if args.target else target_positions(trace,args.targets_per_sample)
        if not np.isin(targets,eligible).all(): raise ValueError(f"{e['sample_id']}: invalid/special target")
        e['targets'] = sorted(set(targets.tolist()))
        e['eligible_targets'] = len(eligible)
        e['folder'] = (Path('samples')/e['split']/e['task_type']/str(e['sample_id'])).as_posix()
        labels=read_labels(path,trace) if path.with_suffix('.labels.npz').exists() else None
        e['target_details']=target_details(trace,targets,labels)
        e['added_special_positions']=trace['added_special_positions'].tolist()
    return {
        **native,
        'samples': selected,
        'selection': {
            'mode': 'uniform',
            'labels_used_for_selection': False,
            'labels_used_for_graph': False,
        },
    }


def run(args):
    import torch
    if min(args.samples_per_group,args.targets_per_sample,args.edge_budget,args.bootstrap)<0 or min(args.query_chunk,args.source_chunk,args.target_chunk,args.cpu_threads)<1:
        raise ValueError('invalid budgets')
    torch.set_num_threads(args.cpu_threads)
    output = args.output or args.audit/'message_dag_v2'
    if output.resolve()==args.audit.resolve(): raise ValueError('derived output must differ from the capture directory')
    if args.phase=='evaluate':
        from .report import evaluate
        return evaluate(output, json.loads((output/'index.json').read_text()), bootstrap=args.bootstrap)
    original = json.loads((args.audit/'index.json').read_text())
    if original.get('audit_schema')!=3 or not original['settings'].get('save_states'):
        raise ValueError('message DAGs require v3 full-state caches; old scalar summaries cannot reconstruct them')
    request={k:getattr(args,k) for k in ('selection','split','task','sample_id','samples_per_group','targets_per_sample','target')}
    previous=output/'index.json'
    if args.selection=='paired' and previous.exists() and not args.list_available:
        manifest=json.loads(previous.read_text())
        if manifest.get('selection_request')!=request or manifest.get('dag_settings',{}).get('schema')!=SCHEMA:
            raise ValueError('comparison selection/configuration changed; choose another --output to keep the frozen comparison')
        print('Resume frozen sample IDs and N-H-N targets from index.json.',flush=True)
    else:
        manifest = plan(args, original)
    if args.list_available:
        print('Required: .npz + .history.npz + .qk.npz + .states.npz. '
              'Optional .attention.npz, labels and old audit results are not required for graph construction.', flush=True)
        return manifest
    model = args.model or Path(original['settings']['model'])
    cfg = json.loads((model/'config.json').read_text())
    settings = {
        'schema': SCHEMA,
        'model': str(model),
        'mlp_rule': args.mlp_rule,
        'edge_budget': args.edge_budget,
        'mask_policy': MASK_POLICY,
    }
    if previous.exists() and json.loads(previous.read_text()).get('dag_settings')!=settings:
        raise ValueError('DAG settings changed; choose another --output before replacing an existing comparison')
    manifest['dag_settings'] = settings
    manifest.update(artifact_header(SOURCE_ALLOCATION, SCHEMA))
    manifest['native_audit'] = str(args.audit.resolve())
    manifest['selection_request']=request
    peak_tape,output_bytes,layer_loads = 0,0,0
    for e in manifest['samples']:
        trace=read_trace(args.audit/e['path'])
        g = len(source_partition(trace)['names'])
        size = 2*cfg['num_hidden_layers']*g*len(trace['row_position'])*cfg['hidden_size']*4
        e['temporary_bytes'] = size
        layers,heads,rows=cfg['num_hidden_layers'],cfg['num_attention_heads'],len(trace['row_position'])
        estimate=len(e['targets'])*4*(g*(5*layers+1)*rows+4*g*layers*heads+layers*args.edge_budget*(g+10))
        e['estimated_output_bytes_uncompressed']=estimate
        output_bytes+=estimate
        layer_loads+=layers*(1+(len(e['targets'])+args.target_chunk-1)//args.target_chunk)
        peak_tape = max(peak_tape,size)
        print(f"plan {e['split']}/{e['task_type']}/{e['sample_id']}: sources={g} "
              f"targets={len(e['targets'])}/{e['eligible_targets']} temporary={size/2**30:.2f} GiB "
              f"output~{estimate/2**30:.2f} GiB before compression",flush=True)
        print(f"  controls reclassified={trace['added_special_positions'].tolist()}; targets="+
              ', '.join(f"{d['position']}:{d['text']!r}[{d['label']}]" for d in e['target_details']),flush=True)
    print(f"No LLM recapture. Peak temporary tape={peak_tape/2**30:.2f} GiB; one target adjoint per selected token. "
          f"Selection={args.selection}; labels enter audit sampling only in paired mode, never the operators.",flush=True)
    print(f'Estimated derived arrays={output_bytes/2**30:.2f} GiB before compression; layer loads={layer_loads}.',flush=True)
    if args.plan_only: return manifest
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'index.json',manifest)  # planned targets survive interruption
    if args.phase!='trace' or args.selection=='paired': ensure_labels(args,manifest)
    scratch = args.scratch or output/'work'
    scratch.mkdir(parents=True,exist_ok=True)
    weights = None
    for e in tqdm(manifest['samples'],desc='message DAG',unit='sample'):
        path, folder = args.audit/e['path'], output/e['folder']
        folder.mkdir(parents=True,exist_ok=True)
        trace=read_trace(path)
        metadata = {k:trace[k] for k in ('token_ids','token_text','special_mask','source_unit_id','evidence_mask',
                                        'row_position','response_start','predictor_logprob','capture_special_mask','added_special_positions')}
        metadata.update(layers=np.array(cfg['num_hidden_layers']),heads=np.array(cfg['num_attention_heads']),
                        sample_id=np.array(str(e['sample_id'])),generator_model=trace.get('generator_model',np.array('unknown')))
        label_file=path.with_suffix('.labels.npz')
        if label_file.exists():
            labels=read_labels(path,trace)
            if args.selection=='paired' and any(labels[d['position']-int(trace['response_start'])]!=d['label'] for d in e['target_details']):
                raise ValueError(f'{path}: labels changed since the comparison was frozen; choose another --output')
            shutil.copyfile(label_file,folder/'labels.npz')  # partial reports work even if tracing is interrupted
        missing = []
        for target in e['targets']:
            destination = folder/f'target_{target}.npz'
            if destination.exists():
                with np.load(destination) as saved:
                    if json.loads(str(saved['settings']))!=settings or not np.array_equal(saved['token_ids'],trace['token_ids']) or not np.array_equal(saved['source_token_group'],source_partition(trace)['token_group']):
                        raise ValueError(f'{destination}: cached DAG identity changed; choose another --output')
            else: missing.append(target)
        atomic_npz(folder/'meta.npz',**metadata)
        if not missing: continue
        if shutil.disk_usage(scratch).free < e['temporary_bytes']+2**20:
            raise OSError(f"{scratch}: insufficient space for {e['temporary_bytes']/2**30:.2f} GiB source tape")
        if weights is None: weights=CheckpointWeights(model,args.device)
        started = perf_counter()
        progress = tqdm(total=cfg['num_hidden_layers']*(len(missing)+1),desc=str(e['sample_id']),unit='layer',leave=False)
        def update(stage, progress_bar=progress, sample_started=started):
            progress_bar.set_postfix_str(
                f'{stage}; elapsed={perf_counter()-sample_started:.1f}s', refresh=False
            )
            progress_bar.update(1)
        with NativeCache(path,weights) as cache, TemporaryDirectory(prefix='dag-',dir=scratch) as temporary:
            tape = Tape(cache,temporary)
            try:
                prepare(cache,tape,source_chunk=args.source_chunk,query_chunk=args.query_chunk,rule=args.mlp_rule,progress=update)
                for a,b in blocks(len(missing),args.target_chunk):
                    target_started = perf_counter()
                    graphs = trace_targets(cache,tape,missing[a:b],source_chunk=args.source_chunk,query_chunk=args.query_chunk,
                                           rule=args.mlp_rule,edge_budget=args.edge_budget,progress=update)
                    duration=perf_counter()-target_started
                    for target,graph in zip(missing[a:b],graphs):
                        graph.update(describe(graph,trace))
                        graph.update(settings=np.array(json.dumps(settings)), token_ids=trace['token_ids'],
                                     source_token_group=tape.partition['token_group'], seconds_per_target_in_block=np.array(duration/(b-a)))
                        atomic_npz(folder/f'target_{target}.npz',**graph)
                        tqdm.write(f"saved {e['sample_id']} t={target}: block {duration:.1f}s/{b-a} targets; "
                                   f"balance={float(graph['balance_error'].max()):.2g}; display coverage={float(graph['edge_display_coverage']):.1%}")
            finally: tape.close()
        progress.close()
    del weights
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.phase=='trace': return manifest
    from .report import evaluate
    return evaluate(output,manifest,bootstrap=args.bootstrap)


if __name__=='__main__':
    run(parser().parse_args())
