"""Regression checks for competing writers and interrupted checkpoint writes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import numpy as np
import pytest

from experiments.reanchor_flow.message_dag import event_run


@pytest.mark.parametrize('kind',['json','npz'])
def test_overlapping_writes_never_share_a_temporary_name(tmp_path,monkeypatch,kind):
    barrier = Barrier(2)
    seen = []
    owner,name = (event_run,'save_json') if kind=='json' else (np,'savez_compressed')
    original = getattr(owner,name)
    def write(path,*args,**kwargs):
        original(path,*args,**kwargs)
        seen.append(path)
        barrier.wait(timeout=10)  # both files exist before either replacement
    monkeypatch.setattr(owner,name,write)
    payloads = [dict(writer=i,values=np.full(100,i)) for i in range(2)]
    target = tmp_path/('index.json' if kind=='json' else 'event_7.npz')
    def save(value):
        if kind=='json': event_run.save_index(tmp_path,value)
        else: event_run.atomic_npz(target,**value)
    with ThreadPoolExecutor(2) as pool: list(pool.map(save,payloads))
    assert len(set(seen))==2 and not any(path.exists() for path in seen)
    if kind=='json': result = json.loads(target.read_text())
    else:
        with np.load(target,allow_pickle=False) as saved: result = dict(saved)
    assert int(result['writer']) in (0,1)
    np.testing.assert_array_equal(result['values'],np.full(100,int(result['writer'])))


@pytest.mark.parametrize('kind',['json','npz'])
def test_failed_write_keeps_previous_checkpoint_and_cleans_only_its_temp(tmp_path,monkeypatch,kind):
    target = tmp_path/('index.json' if kind=='json' else 'event_7.npz')
    save = (lambda: event_run.save_index(tmp_path,dict(done=2))) if kind=='json' else (
        lambda: event_run.atomic_npz(target,done=np.array(2)))
    save();before = target.read_bytes()
    unrelated = tmp_path/'unrelated.tmp';unrelated.write_text('keep')
    def interrupted(path,*args,**kwargs):
        Path(path).write_bytes(b'partial')
        raise OSError('simulated write failure')
    owner,name = (event_run,'save_json') if kind=='json' else (np,'savez_compressed')
    monkeypatch.setattr(owner,name,interrupted)
    with pytest.raises(OSError,match='simulated write failure'): save()
    assert target.read_bytes()==before
    assert set(tmp_path.iterdir())=={target,unrelated}


def test_second_process_is_rejected_before_work_and_lock_releases_after_exception(tmp_path):
    output = tmp_path/'events'
    code = """
import sys
from experiments.reanchor_flow.message_dag import event_run
def must_not_run(*args):
    raise AssertionError('second writer entered computation')
event_run._run = must_not_run
args = event_run.parser().parse_args(['--output',sys.argv[1]])
try:
    event_run.run(args)
except RuntimeError as error:
    assert 'active writer' in str(error)
    print('blocked before computation')
else:
    raise AssertionError('writer was not blocked')
"""
    with pytest.raises(ValueError,match='interrupted run'):
        with event_run.output_writer(output):
            result = subprocess.run([sys.executable,'-c',code,str(output)],capture_output=True,text=True,timeout=30)
            assert result.returncode==0,result.stdout+result.stderr
            assert 'blocked before computation' in result.stdout
            # Separate output directories remain independent.
            with event_run.output_writer(tmp_path/'other'): pass
            raise ValueError('interrupted run')
    inode = (output/'.event_run.lock').stat().st_ino
    with event_run.output_writer(output):
        assert (output/'.event_run.lock').stat().st_ino==inode


def test_list_available_does_not_create_an_output_or_lock(tmp_path,monkeypatch):
    audit = tmp_path/'audit';audit.mkdir()
    (audit/'index.json').write_text(json.dumps(dict(audit_schema=3,settings={},samples=[])))
    def must_not_lock(*args): raise AssertionError('read-only coverage acquired output lock')
    monkeypatch.setattr(event_run,'output_writer',must_not_lock)
    result = event_run.run(event_run.parser().parse_args(['--audit',str(audit),'--list-available']))
    assert result['samples']==[]
    assert not (audit/'lookback_events_v2').exists()
