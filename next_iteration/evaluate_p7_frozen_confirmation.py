"""One-shot CPU runner for the documented frozen P7 confirmation commands."""
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path
from .reasoned_confirmation_freeze_check import check, ROOT, INPUT_SHA, sha
from .confirmation_assess_scoped import frozen_records
from .reasoned_confirmation_assess import check_boundary

BASE=ROOT.parents[1]
PRED=ROOT/'outputs/p7_reasoned_confirmation_20260914_v1'
REF=ROOT/'outputs/p7_population_reference_confirmation_20260914_v1'
EXECUTION=ROOT/'outputs/P7_CONFIRMATION_EVALUATION_EXECUTION_20260914.json'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def readiness():
    freeze_path=ROOT/'outputs/P7_CONFIRMATION_FREEZE_20260914.json'
    freeze=check(freeze_path)
    launch_path=ROOT/'outputs/P7_CONFIRMATION_LAUNCH_20260914.json'
    launch=json.loads(launch_path.read_text())
    if launch.get('actual_subprocess_returncode')!=0:
        raise ValueError('GPU command not completed successfully; no annotations opened')
    m,records=frozen_records(PRED)
    check_boundary(m,records,freeze)
    return dict(freeze_sha256=sha(freeze_path),prediction_manifest_sha256=sha(PRED/'manifest.json'),
                launch_record_sha256=sha(launch_path),responses=len(records),sources=64,
                tokens=sum(len(r['token_ids']) for r in records))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    identity=readiness()
    if not args.execute:
        print(json.dumps({'ready':True,**identity}));return
    logs=[ROOT/'runs/p7_confirmation_reference_20260914_v1.log',
          ROOT/'runs/p7_confirmation_evaluation_20260914_v1.log']
    if any(path.exists() for path in [REF,PRED/'evaluation.json',EXECUTION,*logs]):
        raise FileExistsError('one-shot CPU outputs already exist; inspect/preserve, do not overwrite')
    commands=[
        [sys.executable,'-m','next_iteration.reasoned_confirmation_reference_export',
         '--inputs',str(ROOT/'outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl'),
         '--population',str(BASE/'research/reanchor/outputs/ragtruth_population_20260912'),
         '--phase','validation','--output',str(REF)],
        [sys.executable,'-m','next_iteration.reasoned_confirmation_assess',
         '--predictions',str(PRED),'--reference-predictions',str(REF),
         '--annotations',str(BASE/'data/RAGTruth/dataset/response.jsonl'),
         '--primary','reasoned_source_risk','--output',str(PRED/'evaluation.json')],
    ]
    state=dict(status='running',started_utc=utc(),pid=os.getpid(),identity=identity,
               executed_code_sha256=sha(__file__),commands=commands,completed=[])
    with EXECUTION.open('x') as f:json.dump(state,f,indent=2)
    EXECUTION.with_suffix('.executed_code.py').write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for cmd,log in zip(commands,logs):
        start=utc()
        with log.open('x') as f:run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        state['completed'].append(dict(command=cmd,started_utc=start,completed_utc=utc(),actual_returncode=run.returncode,
                                       log=str(log),log_sha256=sha(log)))
        if run.returncode:state['status']='failed'
        elif len(state['completed'])==2:
            state.update(status='complete',evaluation_sha256=sha(PRED/'evaluation.json'))
        EXECUTION.write_text(json.dumps(state,indent=2)+'\n')
        if run.returncode:raise SystemExit(run.returncode)
    print(json.dumps(state))

if __name__=='__main__':main()
