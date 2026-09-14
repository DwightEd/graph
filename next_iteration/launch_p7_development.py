import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path('/share/home/tm902089733300000/a903202310/lys')
G = B / 'research/graph'
OUT = G / 'outputs/p7_reasoned_development_20260914_v1'
LOG = G / 'runs/p7_reasoned_development_20260914_v1.log'
RECORD = G / 'outputs/P7_DEVELOPMENT_LAUNCH_20260914.json'
CMD = ['bash', str(G / 'scripts/run_local_grounding_reasoned.sh'), 'development', str(OUT)]
EXPECTED_ENV = '03909e02bb7b57dbd24bb883979ab51f068d06d57875c3dfd0bba49e1d1f4c4a'
EXPECTED_METADATA = {
    'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'configuration.json': 'f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9',
    'model.safetensors.index.json': 'f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'config.json': 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert str(Path(sys.executable)) == str(B / 'conda_envs/research/bin/python'), sys.executable
pilot=G/'outputs/p7_reasoned_pilot_20260914_v1'
pm=json.loads((pilot/'manifest.json').read_text())
pl=json.loads((G/'outputs/P7_PILOT_LAUNCH_20260914.json').read_text())
assert pl['actual_subprocess_returncode']==0 and pm['complete'] and not pm.get('error')
assert pm['completed_ids']==pm['planned_ids'] and len(pm['completed_ids'])==2
assert sha(G/'next_iteration/local_grounding_reasoned.py')==pm['code_sha256']=='f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f'
assert sha(G/'docs/P7_REASONED_AUDIT_PROTOCOL.md')==pm['protocol_sha256']=='d54fa4b6c6e91f6868d37e6eb65eb59db5db5840af54b21ee17d7313d2798a5e'
for filename,expected in {**pm['output_sha256'],**pm['engineering_artifact_sha256']}.items():
    assert sha(pilot/filename)==expected, filename
canaries=json.loads((pilot/'canaries.json').read_text())
assert len(canaries)==8
for case in canaries:
    assert 'AB'[max(range(2),key=case['result']['logits'].__getitem__)]==case['expected'], case['name']
spec = json.loads((G / '.aris/compute/env-spec.json').read_text())
env_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert env_hash == EXPECTED_ENV, env_hash
assert 'research@03909e02' in (G / '.aris/compute/local.md').read_text()
for name, wanted in spec['packages'].items():
    assert importlib.metadata.version(name) == wanted, (name, importlib.metadata.version(name), wanted)
for path in [OUT, LOG, RECORD]:
    assert not path.exists(), f'pre-existing artifact; no launch: {path}'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
assert len(gpu.splitlines()) == 1 and int(gpu.split(',')[2].strip()) < 500, gpu
assert not apps, apps
input_path = B / 'research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl'
assert sha(input_path) == '3258b79c7886dc2f6f93ff6c6e9e38c695a773b7dab2ddfc4cba271a3baac38f'
protocol = (G / 'docs/P7_DEVELOPMENT_RUN_20260914.md').read_text()
assert ' '.join(CMD) in protocol
model_metadata = {p.name: sha(p) for p in (B / 'models/Qwen3-8B').glob('*.json')}
assert model_metadata == EXPECTED_METADATA, 'model metadata changed from P6 weight witness'
preflight = {
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'canonical_env_spec_sha256': env_hash,
    'environment_identity': 'research@03909e02',
    'environment_identity_kind': 'normalized environment spec hash, not Git SHA',
    'python': sys.executable,
    'versions': {name: importlib.metadata.version(name) for name in spec['packages']},
    'gpu': gpu, 'compute_apps': apps, 'output_log_launch_absent': True,
    'input_sha256': sha(input_path),
    'frozen_files_sha256': {str(p.relative_to(G)): sha(p) for p in [
        G / 'docs/P7_REASONED_AUDIT_PROTOCOL.md', G / 'scripts/run_local_grounding_reasoned.sh',
        G / 'docs/P7_DEVELOPMENT_RUN_20260914.md', G / 'next_iteration/development_assess_scoped.py',
        G / 'next_iteration/local_grounding_reasoned.py', G / 'next_iteration/grounding_contrast.py']},
    'pilot_manifest_sha256':sha(pilot/'manifest.json'),
    'launcher_sha256':sha(Path(__file__)),
    'model_metadata_sha256': model_metadata,
}
record = dict(command=CMD, command_verbatim=' '.join(CMD), cwd=str(G), output=str(OUT), log=str(LOG),
    attempt=1, status='launching', actual_subprocess_returncode=None, preflight=preflight,
    scope='main-executed full development32 after fresh pilot engineering gate; no annotations or natural efficacy evaluation in runner')
with RECORD.open('x') as handle:
    json.dump(record, handle, indent=2)
supervisor_source = '''
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
record_path = Path(RECORD_PATH)
record = json.loads(record_path.read_text())
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
    temporary = record_path.with_suffix('.writing')
    temporary.write_text(json.dumps(record, indent=2) + '\\n')
    os.replace(temporary, record_path)
record['supervisor_pid'] = os.getpid()
record['launch_utc'] = utc()
start = time.monotonic()
try:
    with open(record['log'], 'xb', buffering=0) as log:
        child = subprocess.Popen(record['command'], cwd=record['cwd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record.update(pid=child.pid, pid_observed_utc=utc(), status='running')
        save()
        code = child.wait()
    record.update(actual_subprocess_returncode=code, completed_utc=utc(), status='completed' if code == 0 else 'failed',
        subprocess_wall_seconds=time.monotonic()-start,
        log_sha256=hashlib.sha256(Path(record['log']).read_bytes()).hexdigest())
except BaseException as exc:
    record.update(status='supervisor_error', supervisor_error=repr(exc), completed_utc=utc())
save()
'''
supervisor_source = 'RECORD_PATH = ' + repr(str(RECORD)) + '\n' + supervisor_source
supervisor = subprocess.Popen([sys.executable, '-c', supervisor_source], cwd=G,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(json.dumps(dict(supervisor_pid=supervisor.pid, record=str(RECORD), preflight=preflight), indent=2), flush=True)
