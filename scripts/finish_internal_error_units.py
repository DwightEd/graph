"""Wait for the already launched capture, then execute one frozen CPU evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/share/home/tm902089733300000/a903202310/lys/research/graph')
BASE=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    p=Path(f'/proc/{pid}/stat')
    try:
        fields=p.read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-pid',type=int,required=True)
    args=parser.parse_args();os.chdir(ROOT)
    status=ROOT/'outputs/N9_CPU_FOLLOWUP_20260914.json'
    if status.exists():raise RuntimeError('one-shot followup already recorded')
    evaluator=ROOT/'next_iteration/internal_error_units_evaluate.py'
    stamp=identity(args.capture_pid)
    if stamp is None:raise RuntimeError('capture process absent at launch')
    hashes={str(p):sha(p) for p in [evaluator,ROOT/'next_iteration/internal_error_units.py',
        ROOT/'next_iteration/grounding_contrast_evaluate.py',ROOT/'next_iteration/confirmation_assess_scoped.py',
        ROOT/'next_iteration/development_assess_scoped.py']}
    state=dict(pid=os.getpid(),capture_pid=args.capture_pid,capture_start=stamp,state='waiting_capture',
               evaluator_dependencies_sha256=hashes,started=time.time())
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
    save()
    try:
        while identity(args.capture_pid)==stamp:
            if time.time()-state['started']>7200:raise RuntimeError('capture wait budget exceeded')
            time.sleep(10)
        capture=ROOT/'outputs/n9_internal_error_units_full_20260914_v1'
        manifest=json.loads((capture/'manifest.json').read_text())
        if not manifest.get('complete') or manifest.get('error') or len(manifest['completed_ids'])!=800:
            raise RuntimeError('capture did not finish successfully; no evaluation launched')
        if any(sha(p)!=h for p,h in hashes.items()):raise RuntimeError('evaluator dependencies changed after scheduling')
        command=[str(BASE/'conda_envs/research/bin/python'),'-m','next_iteration.internal_error_units_evaluate',
            '--inputs','outputs/n9_internal_error_units_inputs_20260914_v3/inputs.jsonl',
            '--capture',str(capture),'--output','outputs/n9_internal_error_units_evaluation_20260914_v1',
            '--annotations',str(BASE/'data/RAGTruth/dataset/response.jsonl')]
        state.update(state='evaluating',command=command,capture_manifest_sha256=sha(capture/'manifest.json'));save()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (ROOT/'outputs/n9_internal_error_units_evaluation_20260914_v1.log').open('x') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        state.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,finished=time.time());save()
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),finished=time.time());save();raise

if __name__=='__main__':main()
