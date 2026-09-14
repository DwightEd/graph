"""CPU-only preparation checks; NEVER freeze, score or open annotations."""
import datetime, hashlib, json, os, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/P7_CONFIRMATION_INTERFACE_CHECK_20260914_v2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    if (ROOT/'outputs/P7_CONFIRMATION_FREEZE_20260914.json').exists():
        raise ValueError('preparation only, freeze already exists')
    target=ROOT/'outputs/p7_reasoned_confirmation_20260914_v1'
    if target.exists():raise FileExistsError(target)
    import sys
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4')
    steps=[]
    for cmd,expected in [
        ([sys.executable,'-m','unittest','discover','-s','tests','-p','test_reasoned_confirmation.py','-v'],0),
        (['bash','-n','scripts/run_reasoned_confirmation.sh'],0),
        (['bash','scripts/run_reasoned_confirmation.sh',str(target)],2),
    ]:
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        run=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
        steps.append(dict(command=cmd,started_utc=started,returncode=run.returncode,
                          expected_returncode=expected,stdout=run.stdout,stderr=run.stderr))
        if run.returncode!=expected:break
    from next_iteration.reasoned_confirmation_freeze_check import FILES
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PASS' if len(steps)==3 and all(s['returncode']==s['expected_returncode'] for s in steps) and not target.exists() else 'FAIL',
        steps=steps,confirmation_output_absent=not target.exists(),
        freeze_absent=not (ROOT/'outputs/P7_CONFIRMATION_FREEZE_20260914.json').exists(),
        annotations_opened=0,model_forwards=0,executed_code_sha256=sha(__file__),
        existing_file_sha256={p:sha(ROOT/p) for p in FILES if (ROOT/p).is_file()},
        future_freeze_artifacts_pending=[p for p in FILES if not (ROOT/p).is_file()])
    with OUT.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
