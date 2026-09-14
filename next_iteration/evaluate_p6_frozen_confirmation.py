"""One-shot CPU orchestration of the already-frozen P6 evaluation commands.

No new metric/scoring code. Refuses incomplete GPU execution and existing outputs.
Default is read-only readiness checking; --execute runs the two pinned commands.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .confirmation_freeze_check import check, sha
from .confirmation_assess_scoped import check_scope_addendum, frozen_records

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
PRED = ROOT / 'outputs/p6_review_confirmation_20260914_v1'
REF = ROOT / 'outputs/p6_population_reference_confirmation_20260914_v1'
SCOPE = ROOT / 'outputs/P6_EVALUATION_SCOPE_ADDENDUM_20260914.json'
EXECUTION = ROOT / 'outputs/P6_CONFIRMATION_EVALUATION_EXECUTION_20260914.json'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ready():
    freeze = check(ROOT / 'outputs/P6_CONFIRMATION_FREEZE_20260914.json')
    if freeze['freeze_sha256'] != 'ffe0607ed0be030310d12c32c4145de6e13d56a12d7d381f2d68b1c09df307d5':
        raise ValueError('parent freeze identity')
    if sha(SCOPE) != '8713a47047aa69c754df8dbba52404d6129d843269c1dc44b113d57ff2af60f9':
        raise ValueError('scope addendum identity')
    check_scope_addendum(SCOPE)
    launch = json.loads((ROOT / 'outputs/P6_CONFIRMATION_LAUNCH_20260914.json').read_text())
    if launch.get('actual_subprocess_returncode') != 0:
        raise ValueError('GPU command has not finished successfully; no annotation access')
    manifest, records = frozen_records(PRED)
    if len(records) != 64 or len({str(r['source_id']) for r in records}) != 32:
        raise ValueError('fixed64 denominator')
    if manifest.get('input_sha256') != 'bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101':
        raise ValueError('input identity')
    if manifest['settings']['phase'] != 'validation':
        raise ValueError('confirmation phase')
    return {'freeze': freeze, 'prediction_manifest_sha256': sha(PRED / 'manifest.json'),
            'launch_record_sha256': sha(ROOT / 'outputs/P6_CONFIRMATION_LAUNCH_20260914.json'),
            'scope_addendum_sha256': sha(SCOPE), 'responses': len(records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    identity = ready()
    if not args.execute:
        print(json.dumps({'ready': True, **identity}))
        return
    logs = [ROOT / 'runs/p6_confirmation_reference_20260914_v1.log',
            ROOT / 'runs/p6_confirmation_evaluation_20260914_v1.log']
    forbidden = [REF, PRED / 'evaluation.json', EXECUTION, *logs]
    if any(path.exists() for path in forbidden):
        raise FileExistsError('One-shot evaluation already exists; inspect and preserve it, do not overwrite')
    commands = [
        [sys.executable, '-m', 'next_iteration.confirmation_reference_export',
         '--inputs', str(ROOT / 'outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl'),
         '--population', str(BASE / 'research/reanchor/outputs/ragtruth_population_20260912'),
         '--phase', 'validation', '--output', str(REF)],
        [sys.executable, '-m', 'next_iteration.confirmation_assess_scoped',
         '--scope-addendum', str(SCOPE), '--predictions', str(PRED),
         '--reference-predictions', str(REF), '--annotations', str(BASE / 'data/RAGTruth/dataset/response.jsonl'),
         '--primary', 'reviewed_source_risk', '--output', str(PRED / 'evaluation.json')],
    ]
    state = dict(started_utc=utc(), pid=os.getpid(), identity=identity,
                 code_sha256=sha(__file__), commands=commands, completed=[], status='running')
    with EXECUTION.open('x') as f:
        json.dump(state, f, indent=2)
    EXECUTION.with_suffix('.executed_code.py').write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for command, log in zip(commands, logs):
        start = utc()
        with log.open('x') as output:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        state['completed'].append(dict(command=command, started_utc=start, ended_utc=utc(),
                                      actual_returncode=result.returncode, log=str(log), log_sha256=sha(log)))
        if result.returncode:
            state['status'] = 'failed'
        elif len(state['completed']) == 2:
            state['status'] = 'complete'
            state['evaluation_sha256'] = sha(PRED / 'evaluation.json')
        EXECUTION.write_text(json.dumps(state, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
