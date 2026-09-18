"""Check the real export entry and its exact payload; no trained model is needed."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from experiments.charm_structure_audit.export_circuit_inputs import FIELDS, export_circuit_inputs


def inputs(tmp_path):
    root = tmp_path / 'seed with spaces'
    audit = root / 'audit_whitebox'
    (audit / 'captures').mkdir(parents=True)
    checkpoint = root / 'actual_node.pt'
    checkpoint.write_bytes(b'opaque checkpoint bytes; never unpickled')
    pairs = []
    arrays = dict(error_x=np.arange(12).reshape(3, 4), normal_x=np.zeros((3, 4)),
                  logits=np.array([[1., 2., 3.], [-1., 0., 1.]]),
                  baseline_score=np.full((2, 3), .5), tokens=np.arange(6).reshape(2, 3),
                  text=np.array([['one', 'two', 'three'], ['a', 'b', 'c']]), threshold=np.array(.8))
    for index in range(2):
        pairs.append(dict(id=str(index), source_id='source', file=f'{index:06d}.npz'))
        # Loading unused edge members would require pickle and fail.
        np.savez_compressed(audit / 'captures' / pairs[-1]['file'], **arrays,
                            edge_attr=np.array([object()], dtype=object),
                            embedding=np.zeros((3, 1024)))
    manifest = dict(checkpoint=[str(checkpoint), checkpoint.stat().st_size, 0], pairs=pairs)
    (audit / 'manifest.json').write_text(json.dumps(manifest))
    (audit / 'geometry.json').write_text(json.dumps(dict(layers=2, heads=2)))
    return root, arrays, manifest


def test_export_exact_inputs_and_checkpoint_without_touching_originals(tmp_path):
    root, arrays, manifest = inputs(tmp_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in root.rglob('*') if path.is_file()}
    destination = tmp_path / 'out' / 'inputs.tar.gz'
    export_circuit_inputs(root, destination)
    with tarfile.open(destination) as archive:
        assert archive.getnames() == ['node_only/checkpoint.pt', 'manifest.json', 'geometry.json',
                                      'inputs/000000.npz', 'inputs/000001.npz']
        assert archive.extractfile('node_only/checkpoint.pt').read() == Path(manifest['checkpoint'][0]).read_bytes()
        assert json.load(archive.extractfile('manifest.json')) == manifest
        with np.load(io.BytesIO(archive.extractfile('inputs/000000.npz').read()), allow_pickle=False) as saved:
            assert set(saved.files) == set(FIELDS)
            for name in FIELDS:
                np.testing.assert_array_equal(saved[name], arrays[name])
    for path, state in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state


def test_actual_main_entry_uses_default_output_without_importing_torch(tmp_path):
    root, _, _ = inputs(tmp_path)
    code = (
        "import runpy, sys; sys.argv=['main']+sys.argv[1:]; "
        "runpy.run_module('experiments.charm_structure_audit.main', run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, '-c', code, '--mode', 'export_circuit', '--root', str(root)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert 'Exported 2 paired inputs' in completed.stdout
    assert (root / 'audit_circuit_export' / 'charm_circuit_inputs.tar.gz').exists()


def test_missing_capture_is_not_silently_skipped_or_published(tmp_path):
    root, _, _ = inputs(tmp_path)
    destination = tmp_path / 'complete.tar.gz'
    export_circuit_inputs(root, destination)
    original = destination.read_bytes()
    (root / 'audit_whitebox/captures/000001.npz').unlink()
    with pytest.raises(FileNotFoundError):
        export_circuit_inputs(root, destination)
    assert destination.read_bytes() == original
