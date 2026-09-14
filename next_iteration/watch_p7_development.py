"""Wait for actual P7 child completion, then run only scoped CPU postprocessing."""
import datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/P7_DEVELOPMENT_POSTPROCESS_20260914.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    state=dict(status='waiting_for_actual_gpu_exit',started_utc=utc(),pid=os.getpid(),
               executed_code_sha256=sha(__file__),completed=[])
    with STATE.open('x') as f:json.dump(state,f,indent=2)
    STATE.with_suffix('.executed_code.py').write_bytes(Path(__file__).read_bytes())
    deadline=time.monotonic()+7200
    try:
        while True:
            launch=json.loads((ROOT/'outputs/P7_DEVELOPMENT_LAUNCH_20260914.json').read_text())
            rc=launch.get('actual_subprocess_returncode')
            if rc is not None:
                if rc!=0:raise RuntimeError('GPU child failed; no annotation access')
                break
            if launch.get('status') in ['failed','supervisor_error']:
                raise RuntimeError('GPU supervisor failure; no annotation access')
            if time.monotonic()>deadline:raise TimeoutError('GPU completion not observed within two hours')
            time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        commands=[
            [sys.executable,'-m','next_iteration.evaluate_p7_development','--execute'],
            [sys.executable,'-m','next_iteration.reasoned_development_diagnostics','--output',
             str(ROOT/'outputs/p7_reasoned_development_20260914_v1/diagnostics.json')],
        ]
        state['status']='cpu_postprocessing'
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        for cmd in commands:
            started=utc()
            run=subprocess.run(cmd,cwd=ROOT,env=env)
            state['completed'].append(dict(command=cmd,started_utc=started,completed_utc=utc(),actual_returncode=run.returncode))
            STATE.write_text(json.dumps(state,indent=2)+'\n')
            if run.returncode:raise RuntimeError('CPU postprocess failed, outputs preserved')
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        state['ended_utc']=utc()
        STATE.write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
