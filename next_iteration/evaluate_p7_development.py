"""One-shot CPU P7 development evaluation; refuses incomplete GPU output."""
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path
from .development_assess_scoped import frozen_records, DEV_IDS
from .grounding_contrast_evaluate import sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
PRED=ROOT/'outputs/p7_reasoned_development_20260914_v1'
LAUNCH=ROOT/'outputs/P7_DEVELOPMENT_LAUNCH_20260914.json'
OUT=ROOT/'outputs/P7_DEVELOPMENT_EVALUATION_EXECUTION_20260914.json'
LOG=ROOT/'runs/p7_development_evaluation_20260914_v1.log'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def readiness():
    launch=json.loads(LAUNCH.read_text())
    if launch.get('actual_subprocess_returncode')!=0:
        raise ValueError('GPU command not finished successfully; no labels opened')
    m,rows=frozen_records(PRED)
    if len(rows)!=32 or {str(r['id']) for r in rows}!=DEV_IDS or m['settings']['phase']!='development':
        raise ValueError('not complete original development32')
    if m['code_sha256']!='f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f':
        raise ValueError('P7 scorer identity')
    if sha(ROOT/'next_iteration/development_assess_scoped.py')!='9627dba28c3882a7c2de869af1c406ae0a873991dff9c357e93cfcd333c9d68b':
        raise ValueError('scoped evaluator changed')
    if sum(len(r['token_ids']) for r in rows)!=5170:raise ValueError('full token denominator')
    return dict(launch_sha256=sha(LAUNCH),prediction_manifest_sha256=sha(PRED/'manifest.json'),responses=32,tokens=5170)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    identity=readiness()
    if not args.execute:
        print(json.dumps({'ready':True,**identity}));return
    if any(p.exists() for p in [OUT,LOG,PRED/'evaluation.json']):
        raise FileExistsError('one-shot evaluation already exists, do not overwrite')
    cmd=[sys.executable,'-m','next_iteration.development_assess_scoped',
         '--predictions',str(PRED),'--annotations',str(BASE/'data/RAGTruth/dataset/response.jsonl'),
         '--primary','reasoned_source_risk','--output',str(PRED/'evaluation.json')]
    for name in ['p6_review_development_20260914_v1','p5_local_development_20260914_v3_fp32',
                 'p4_read_commit_development_20260914_v1','p5_population_reference_development_20260914_v1']:
        cmd.extend(['--reference-predictions',str(ROOT/'outputs'/name)])
    state=dict(started_utc=utc(),status='running',pid=os.getpid(),code_sha256=sha(__file__),
               command=cmd,identity=identity,actual_returncode=None)
    with OUT.open('x') as f:json.dump(state,f,indent=2)
    OUT.with_suffix('.executed_code.py').write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with LOG.open('x') as log:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state.update(completed_utc=utc(),actual_returncode=r.returncode,log=str(LOG),log_sha256=sha(LOG),
                 status='complete' if r.returncode==0 else 'failed')
    if r.returncode==0:state['evaluation_sha256']=sha(PRED/'evaluation.json')
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state))
    if r.returncode:raise SystemExit(r.returncode)

if __name__=='__main__':main()
