"""One frozen FIT/pilot coupling falsification, with labels opened only last."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_likelihood_calibration import array_hash, native_likelihood
from .run_source_null_calibration import source_coordinates, verify as verify_previous
from .run_state_graph import snapshot_code
from .run_unlabeled import answer_inputs, weighted_reference_threshold
from .source_coupling import (COORDINATES, SHRINKAGE, answer_coordinates,
    coupling_rank, dependence_ratio, fit_normal_model, fit_risk_reference)
from .source_route_refine_eval import (answer_masks, alarm_counts, method_report,
                                      transitions)
from .unlabeled import equal_source_weights, graph_fields, local_weights


SHARED = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_null_calibration_20261009')
PLAN = SHARED / 'COUPLING_DETECTOR_PLAN.md'
GATE = SHARED / 'COPULA_FIT_CHECK.json'
PREVIOUS = Path('outputs/source_null_calibration_20261009')
DEFAULT_OUTPUT = Path('outputs/source_coupling_20261009')
METHODS = ('old_native', 'likelihood_native', 'null_native', 'coupling_unary', 'coupling_native')
BASELINES = METHODS[:3]
PRIMARY = 'coupling_native'
PACK_FIELDS = ('token_id', 'target', 'source_index', 'answer_index', 'unit_index')
KEYS = {'quilt_not': ('12219',223), 'citation_2': ('12297',106),
        'pension_private': ('15604',96), 'pension_normal': ('15604',103)}


def previous_phase(phase):
    """Validate old frozen inputs without reading their evaluation or other phases."""
    if phase not in ('fit','pilot'):
        raise ValueError('This experiment has no DEV/test stage')
    verify_previous(PREVIOUS,phase)
    with np.load(PREVIOUS/f'{phase}_pack.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(PACK_FIELDS):
            raise ValueError('Scoring requires a pure identity pack')
        pack={name:saved[name].copy() for name in PACK_FIELDS}
    metadata=json.loads((PREVIOUS/f'{phase}_metadata.json').read_text())
    with np.load(PREVIOUS/f'{phase}_scores.npz',allow_pickle=False) as saved:
        baseline={name:saved[name].copy() for name in BASELINES}
    wanted=(612083,4026,671) if phase=='fit' else (1139,8,4)
    actual=(len(pack['target']),len(metadata['records']),len(np.unique(pack['source_index'])))
    if actual!=wanted:
        raise ValueError(f'Complete {phase} population differs: {actual}/{wanted}')
    return pack,metadata,baseline


def initialize(directory):
    """Bind executed core, evaluator, tests, reference lineage and GT bytes pre-FIT."""
    directory.mkdir(parents=True,exist_ok=False)
    dependencies=snapshot_code(directory)
    test=Path(__file__).with_name('test_source_coupling.py').resolve()
    target=directory/'code_snapshot'/test.relative_to(Path.cwd().resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(test,target)
    dependencies[str(test)]=file_hash(test)
    # Bind every copied executable independently, before fitting/scoring.
    # Original-source hashes alone do not certify the archived snapshot bytes.
    snapshot_root=directory/'code_snapshot'
    for copied in sorted(snapshot_root.rglob('*.py')):
        original=(Path.cwd()/copied.relative_to(snapshot_root)).resolve()
        actual=file_hash(copied)
        if dependencies.get(str(original))!=actual:
            raise ValueError(f'Snapshot/source bytes differ before FIT: {copied}')
        dependencies[str(copied.resolve())]=actual
    manifest=json.loads((CACHE/'manifest.json').read_text())
    truth=Path(manifest['dataset'])/'response.jsonl'
    if not (OUTPUT/'CAPTURE_PROTOCOL.json').is_file():
        raise FileNotFoundError('The recorded original capture protocol is required')
    files=[PLAN,GATE,SHARED/'COPULA_FIT_CHECK_PLAN.md',SHARED/'COPULA_FIT_CHECK.py',
        SHARED/'COPULA_FIT_CHECK_WITNESS.json',CACHE/'manifest.json',truth,
        OUTPUT/'CAPTURE_PROTOCOL.json',Path('.aris/compute/env-spec.json')]
    files += [PREVIOUS/name for name in ('fit_FREEZE.json','fit_pack.npz','fit_ranks.npz',
        'fit_metadata.json','global_reference.npz','reference.npz','DEPENDENCIES.json')]
    dependencies.update({str(path.resolve()):file_hash(path) for path in files})
    write_json(directory/'DEPENDENCIES.json',dependencies)
    shutil.copy2(PLAN,directory/'EXPERIMENT_PLAN.md')
    for source,destination in (('global_reference.npz','global_reference.npz'),
                               ('reference.npz','likelihood_reference.npz')):
        shutil.copy2(PREVIOUS/source,directory/destination)
    write_json(directory/'PRE_FIT_BINDING.json',dict(dependencies=dependencies,
        ground_truth_bytes_bound_but_not_parsed=True,methods=METHODS,primary=PRIMARY,
        prior_coupling_gate_sha256=file_hash(GATE),source_null_experiment_unchanged=True))


def load_npz(path):
    with np.load(path,allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


def assert_gate_reproduction(model,reference,diagnostics):
    gate=json.loads(GATE.read_text())
    if gate['status']!='PASS_LIMITED_COUPLING_ONLY' or not all(gate['gates'].values()):
        raise ValueError('Frozen FIT statistical qualification did not pass')
    comparison={'mean':gate['weighted_mean'],'raw_covariance':gate['raw_covariance'],
        'covariance':gate['shrunk_covariance'],'conditioner':gate['conditioner'],
        'conditional_covariance':gate['conditional_covariance']}
    comparison.update({name:gate['conditional_model'][name] for name in ('A','b','d','beta','v')})
    errors={name:float(np.max(np.abs(model[name]-np.asarray(value))))
            for name,value in comparison.items()}
    if max(errors.values())>1e-12 or float(model['shrinkage'])!=gate['shrinkage']:
        raise ValueError(f'FIT Gaussian parameters differ from pre-label check: {errors}')
    hashes={}
    for report in gate['transform_reports']:
        name=report['name']
        for suffix,key in (('_rank','rank_sha256'),('_normal','normal_sha256')):
            actual=array_hash(diagnostics[name+suffix])
            if actual!=report[key]:
                raise ValueError(f'Frozen normal map differs: {name}{suffix}')
            hashes[name+suffix]=actual
        if name in COORDINATES[3:]:
            for suffix,key in (('_values','reference_values_sha256'),
                               ('_cumulative','reference_cumulative_sha256')):
                if array_hash(reference[name+suffix])!=report[key]:
                    raise ValueError(f'Native empirical reference differs: {name}{suffix}')
    return dict(max_parameter_differences=errors,normal_map_sha256=hashes,
                statistical_gate_sha256=file_hash(GATE),tolerance=1e-12)


def fit_model(directory,pack):
    with np.load(PREVIOUS/'fit_ranks.npz',allow_pickle=False) as saved:
        fit_ranks={name:saved[name].copy() for name in COORDINATES}
    global_reference=load_npz(directory/'global_reference.npz')
    model,native_reference,z,diagnostics=fit_normal_model(fit_ranks,pack['source_index'],global_reference)
    reproduction=assert_gate_reproduction(model,native_reference,diagnostics)
    D,_=dependence_ratio(z,model)
    risk_reference=fit_risk_reference(D,pack['source_index'])
    np.savez(directory/'model.npz',**model)
    np.savez(directory/'native_reference.npz',**native_reference)
    np.savez(directory/'risk_reference.npz',**risk_reference)
    write_json(directory/'MODEL_REPRODUCTION.json',reproduction)
    return model,native_reference,risk_reference,D


def answer_fields(global_reference,likelihood_reference,native_reference,risk_reference,
                  model,values,likelihood,attention):
    """Only five score fields; graph operates on every original answer node."""
    coordinates,old_diagnostics=source_coordinates(global_reference,likelihood_reference,values,likelihood)
    graph=dict(native=local_weights(attention))
    scores={}
    for name,coordinate in (('old_native','old'),('likelihood_native','likelihood'),('null_native','null')):
        source=coordinates[coordinate]
        unary=.375*(source['local']+source['full'])+.25*old_diagnostics['raw_route']
        fields,_=graph_fields(unary,graph)
        scores[name]=fields['native_huber']
    z,diagnostics=answer_coordinates(global_reference,native_reference,values,likelihood)
    D,residual=dependence_ratio(z,model)
    diagnostics.update(residual)
    unary=coupling_rank(risk_reference,D)
    fields,_=graph_fields(unary,graph)
    scores.update(coupling_unary=unary,coupling_native=fields['native_huber'])
    diagnostics.update(source_local_effect=np.asarray(values['source_local']),
        source_full_effect=np.asarray(values['source_full']),raw_route_value=np.asarray(values['raw_route']),
        local_native_logp=np.asarray(likelihood['local']),full_native_logp=np.asarray(likelihood['full']))
    if tuple(scores)!=METHODS or not all(np.isfinite(value).all() for value in (*scores.values(),*diagnostics.values())):
        raise ValueError('Exactly five complete finite score fields are required')
    return scores,diagnostics


def score_records(directory,phase,pack,metadata,baseline,model,native_reference,risk_reference,fit_D=None):
    global_reference=load_npz(directory/'global_reference.npz')
    likelihood_reference=load_npz(directory/'likelihood_reference.npz')
    scores={name:np.full(len(pack['target']),np.nan) for name in METHODS}
    diagnostics,files,capture_fields,full_nodes={},{},{},{}
    differences={name:0. for name in BASELINES}
    difference_D=0.
    full_directory=directory/'full_nodes'/phase
    full_directory.mkdir(parents=True,exist_ok=False)
    for index,record in enumerate(metadata['records']):
        values,attention,targets,region=answer_inputs(CACHE,OUTPUT/'capture',record,pack)
        likelihood=native_likelihood(record,values['token_id'])
        result,details=answer_fields(global_reference,likelihood_reference,native_reference,risk_reference,
                                     model,values,likelihood,attention)
        for name in METHODS:
            scores[name][region]=result[name][targets]
        for name,value in details.items():
            diagnostics.setdefault(name,np.empty(len(pack['target'])))[region]=value[targets]
        for name in BASELINES:
            differences[name]=max(differences[name],float(np.abs(result[name][targets]-baseline[name][region]).max()))
        if max(differences.values())>1e-8:
            raise ValueError(f"{record['id']}: archived baseline changed {differences}")
        if fit_D is not None:
            difference_D=max(difference_D,float(np.abs(details['D'][targets]-fit_D[region]).max()))
            if difference_D>1e-12:
                raise ValueError('Full-answer valid FIT D differs from its frozen model reference')
        destination=full_directory/f"{record['id']}.npz"
        np.savez(destination,token_id=values['token_id'],target=np.arange(len(values['token_id'])),
                 **result,**details)
        full_nodes[str(destination.resolve())]=file_hash(destination)
        for name in ('observations.npz','with_source.npz','response.json'):
            path=CACHE/record['directory']/name
            files[str(path.resolve())]=file_hash(path)
        path=OUTPUT/'capture'/record['id']/'arrays.npz'
        capture_fields[str(path.resolve())]=dict(attention=array_hash(attention),answer_ids=array_hash(values['token_id']),
            recipe='local_attention[0,1:] and answer_ids; consumed field bytes only')
        if (index+1)%300==0:
            print(f'COUPLING {phase} {index+1}/{len(metadata["records"])}',flush=True)
    if not all(np.isfinite(value).all() for value in (*scores.values(),*diagnostics.values())):
        raise ValueError('Complete valid-token predictions/diagnostics are required')
    return scores,diagnostics,dict(files=files,capture_fields=capture_fields,full_node_artifacts=full_nodes,
        baseline_max_differences=differences,fit_valid_D_max_difference=difference_D)


def verify(directory,phase):
    freeze=json.loads((directory/f'{phase}_FREEZE.json').read_text())
    if freeze['status']!='all_predictions_frozen' or tuple(freeze['methods'])!=METHODS:
        raise ValueError('Declared fields were not completely frozen')
    for category in ('artifacts','dependencies'):
        for path,expected in freeze[category].items():
            if file_hash(Path(path))!=expected:
                raise ValueError(f'Changed frozen {category}: {path}')
    inputs=json.loads((directory/f'{phase}_inputs.json').read_text())
    for category in ('files','full_node_artifacts'):
        for path,expected in inputs[category].items():
            if file_hash(Path(path))!=expected:
                raise ValueError(f'Changed consumed/exported bytes: {path}')
    for path,expected in inputs['capture_fields'].items():
        with np.load(path,allow_pickle=False) as saved:
            actual=dict(attention=array_hash(saved['local_attention'][0,1:]),answer_ids=array_hash(saved['answer_ids']))
        if any(actual[name]!=expected[name] for name in actual):
            raise ValueError(f'Changed consumed capture field: {path}')
    return freeze


def run_score(directory,phase):
    started=time.time()
    if phase=='fit':
        if directory.exists():
            raise FileExistsError('Use the new frozen output directory exactly once')
    else:
        verify(directory,'fit')
        if any(directory.glob('pilot_*')) or (directory/'full_nodes'/'pilot').exists():
            raise FileExistsError('Preserve partial/completed pilot; no overwrite')
    pack,metadata,baseline=previous_phase(phase)
    if phase=='fit':
        initialize(directory)
        model,native_reference,risk_reference,fit_D=fit_model(directory,pack)
    else:
        model=load_npz(directory/'model.npz')
        native_reference=load_npz(directory/'native_reference.npz')
        risk_reference=load_npz(directory/'risk_reference.npz')
        fit_D=None
    scores,diagnostics,inputs=score_records(directory,phase,pack,metadata,baseline,model,native_reference,risk_reference,fit_D)
    np.savez(directory/f'{phase}_scores.npz',**scores)
    np.savez(directory/f'{phase}_diagnostics.npz',**diagnostics)
    np.savez(directory/f'{phase}_pack.npz',**pack)
    write_json(directory/f'{phase}_metadata.json',metadata)
    write_json(directory/f'{phase}_inputs.json',inputs)
    if phase=='fit':
        weights=equal_source_weights(pack['source_index'])
        thresholds={name:weighted_reference_threshold(value,weights) for name,value in scores.items()}
        write_json(directory/'CONFIG.json',dict(primary=PRIMARY,methods=METHODS,thresholds=thresholds,
            shrinkage=SHRINKAGE,coordinates=COORDINATES,natural_label_fits=0,new_llm_forwards=0,
            timing='offline observed-token, original post-token lag8/head-mean graph',
            threshold_rule='source-equal unknown-label FIT mixture95; not normal FPR',
            source_null_control_promotion=False,graph_changed=False,factual_probability=False))
    dependencies=json.loads((directory/'DEPENDENCIES.json').read_text())
    artifacts=[directory/f'{phase}_{suffix}' for suffix in ('scores.npz','diagnostics.npz','pack.npz','metadata.json','inputs.json')]
    artifacts += [directory/name for name in ('model.npz','native_reference.npz','risk_reference.npz','global_reference.npz',
        'likelihood_reference.npz','MODEL_REPRODUCTION.json','CONFIG.json','DEPENDENCIES.json','PRE_FIT_BINDING.json','EXPERIMENT_PLAN.md')]
    artifacts += [PREVIOUS/f'{phase}_{suffix}' for suffix in ('FREEZE.json','scores.npz','pack.npz','metadata.json')]
    for path,expected in dependencies.items():
        if file_hash(Path(path))!=expected:
            raise ValueError(f'Pre-FIT scientific dependency changed: {path}')
    write_json(directory/f'{phase}_FREEZE.json',dict(status='all_predictions_frozen',methods=METHODS,
        artifacts={str(path.resolve()):file_hash(path) for path in artifacts},dependencies=dependencies,
        baseline_max_differences=inputs['baseline_max_differences'],natural_labels_opened=False,
        command=sys.argv,python=sys.version,numpy=np.__version__,torch=torch.__version__,seconds=time.time()-started))
    print(f'COUPLING {phase} FROZEN seconds={time.time()-started:.2f}',flush=True)


def key_positions(pack,metadata):
    positions={}
    for name,(identity,target) in KEYS.items():
        matches=[]
        for record in metadata['records']:
            if record['id']==identity:
                start,stop=record['packed_start'],record['packed_stop']
                matches.extend((start+np.flatnonzero(pack['target'][start:stop]==target)).tolist())
        if len(matches)!=1:
            raise ValueError(f'Fixed key identity missing/duplicated: {name}')
        positions[name]=matches[0]
    return positions


def export_tokens(directory,pack,metadata,scores,diagnostics,methods,thresholds):
    policies=('fit_mixture95','normal_token_budget')
    columns=['id','source_id','generator','target','token_id','word','context_left','context_right','label','onset','first']
    for name in METHODS:
        columns.append(name)
        for policy in policies:
            for suffix in ('alarm', 'outcome'):
                columns.append(f'{name}__{policy}__{suffix}')
    columns += list(diagnostics)+['changed_fit95','changed_matched57']
    paths=[directory/name for name in ('pilot_tokens.csv','pilot_failures.csv','pilot_changes.csv')]
    streams=[path.open('x',newline='',encoding='utf-8') for path in paths]
    try:
        writers=[csv.DictWriter(stream,fieldnames=columns) for stream in streams]
        for writer in writers:
            writer.writeheader()
        for record in metadata['records']:
            response=json.loads((CACHE/record['directory']/'response.json').read_text())
            text=response['token_text']
            for index in range(record['packed_start'],record['packed_stop']):
                target=int(pack['target'][index])
                label=int(pack['labels'][index])
                if int(response['answer_ids'][target])!=int(pack['token_id'][index]):
                    raise ValueError('Post-score token export identity differs')
                row=dict(id=record['id'],source_id=record['source_id'],generator=record['generator'],target=target,
                    token_id=int(pack['token_id'][index]),word=text[target],context_left=''.join(text[max(0,target-12):target]),
                    context_right=''.join(text[target+1:target+13]),label=label,onset=int(pack['onsets'][index]),first=int(pack['firsts'][index]))
                alarms={}
                for name in METHODS:
                    row[name]=float(scores[name][index])
                    for policy in policies:
                        alarm=bool(scores[name][index]>methods[name][policy]['threshold'])
                        alarms[name,policy]=alarm
                        row[f'{name}__{policy}__alarm']=int(alarm)
                        row[f'{name}__{policy}__outcome']=('TP' if label else 'FP') if alarm else ('FN' if label else 'TN')
                row.update({name:float(value[index]) for name,value in diagnostics.items()})
                row['changed_fit95']=int(alarms[PRIMARY,'fit_mixture95']!=alarms['old_native','fit_mixture95'])
                row['changed_matched57']=int(alarms[PRIMARY,'normal_token_budget']!=alarms['old_native','normal_token_budget'])
                writers[0].writerow(row)
                if alarms[PRIMARY,'normal_token_budget']!=bool(label):
                    writers[1].writerow(row)
                if row['changed_fit95'] or row['changed_matched57']:
                    writers[2].writerow(row)
    finally:
        for stream in streams:
            stream.close()
    return {str(path.resolve()):file_hash(path) for path in paths}


def run_evaluate(directory):
    if any((directory/name).exists() for name in ('pilot_evaluation.json','pilot_tokens.csv','pilot_failures.csv','pilot_changes.csv')):
        raise FileExistsError('Preserve previous/partial evaluation; no overwrite')
    for phase in ('fit','pilot'):
        verify(directory,phase)
    pack=load_npz(directory/'pilot_pack.npz')
    scores=load_npz(directory/'pilot_scores.npz')
    diagnostics=load_npz(directory/'pilot_diagnostics.npz')
    metadata=json.loads((directory/'pilot_metadata.json').read_text())
    config=json.loads((directory/'CONFIG.json').read_text())
    thresholds=config['thresholds']
    if set(scores)!=set(METHODS) or tuple(config['methods'])!=METHODS:
        raise ValueError('The five fixed score fields changed')
    pack.update(evaluation_labels(CACHE,pack,metadata))
    if int(pack['labels'].sum())!=77 or len(pack['labels'])!=1139:
        raise ValueError('Official fixed pilot population differs')
    clean,first,answers=answer_masks(pack,metadata['records'])
    old=alarm_counts(pack,scores['old_native'],thresholds['old_native'],clean,first,answers)
    if (old['tp'],old['fp'],old['normal_answer_alarms'],old['normal_answers'])!=(7,57,3,5):
        raise ValueError('Fixed old pilot operating point differs from the predeclared gate')
    budgets=dict(normal_token_budget=57,normal_answer_budget=3)
    methods={name:method_report(pack,scores[name],thresholds[name],clean,first,answers,budgets) for name in METHODS}
    old_auc=methods['old_native']['all_tokens']['auroc']
    if abs(old_auc-.430271235356)>1e-12:
        raise ValueError('Archived old pilot AUROC differs')
    operating=methods[PRIMARY]['normal_token_budget']
    cutoff=operating['threshold']
    positions=key_positions(pack,metadata)
    key_results={name:dict(id=KEYS[name][0],target=KEYS[name][1],label=int(pack['labels'][position]),
        score=float(scores[PRIMARY][position]),alarm=bool(scores[PRIMARY][position]>cutoff)) for name,position in positions.items()}
    conditions=dict(auroc_above_old=methods[PRIMARY]['all_tokens']['auroc']>old_auc,
        more_than_seven_tp_same57fp=operating['tp']>7 and operating['fp']<=57,
        no_more_than_three_normal_answer_alarms=operating['normal_answer_alarms']<=3,
        at_least_one_clean_first=operating['first_errors_without_prior_alarm']>=1,
        retain_literal_not=key_results['quilt_not']['alarm'],
        recover_one_previously_missed_key=any(key_results[name]['alarm'] for name in ('citation_2','pension_private','pension_normal')))
    comparisons={name:source_bootstrap(pack,scores[PRIMARY],scores[name],repeats=300) for name in BASELINES}
    changed={policy:transitions(pack['labels'],scores['old_native'],scores[PRIMARY],
                               methods['old_native'][policy]['threshold'],methods[PRIMARY][policy]['threshold'])
             for policy in ('fit_mixture95','normal_token_budget','normal_answer_budget')}
    token_exports=export_tokens(directory,pack,metadata,scores,diagnostics,methods,thresholds)
    fit_meta=json.loads((directory/'fit_metadata.json').read_text())
    overlap=sorted({row['source_id'] for row in fit_meta['records']} & {row['source_id'] for row in metadata['records']})
    report=dict(primary=PRIMARY,status='PILOT_PASS_LIMITED_ONLY' if all(conditions.values()) else 'PILOT_FAIL_STOP',
        gate=conditions,key_operating_point=key_results,methods=methods,source_bootstrap=comparisons,
        baseline_to_primary=changed,counts=dict(tokens=1139,errors=77,answers=8,sources=4),
        token_exports_sha256=token_exports,fit_source_overlap=overlap,historical_exposure=True,
        natural_label_fits=0,oracle_operating_points_not_deployment_calibration=True,
        annotation_reader_scope='official JSONL records parsed before ID filtering only after fit+pilot freezes',
        timing='offline observed token, inherited post-token lag8/head-mean graph',
        freeze_sha256={phase:file_hash(directory/f'{phase}_FREEZE.json') for phase in ('fit','pilot')},
        no_control_promotion=True,no_dev_or_test_stage=True)
    write_json(directory/'pilot_evaluation.json',report)
    print(json.dumps(dict(status=report['status'],gate=conditions,primary=methods[PRIMARY]['all_tokens'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('scorefit','scorepilot','evaluatepilot'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='evaluatepilot':
        run_evaluate(DEFAULT_OUTPUT)
    else:run_score(DEFAULT_OUTPUT,'fit' if args.stage=='scorefit' else 'pilot')


if __name__=='__main__':
    main()
