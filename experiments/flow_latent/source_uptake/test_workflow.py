"""The runner's automatic freeze must satisfy the annotation gate unchanged."""
import json

from .evaluate import verify_evaluation_protocol, verify_freeze
from .run import freeze_outputs
from .test_evaluate import synthetic_run


def test_runner_freeze_binds_evaluator_without_manual_manifest_edits(tmp_path):
    roster = synthetic_run(tmp_path)
    old = json.loads((tmp_path / 'FREEZE.json').read_text())
    (tmp_path / 'FREEZE.json').unlink()
    (tmp_path / 'EVALUATION_PROTOCOL.json').unlink()
    files = list(tmp_path.glob('answers/*/data.json')) + list(tmp_path.glob('answers/*/scores.npz'))
    freeze_outputs(tmp_path, roster, {'checkpoints': old['checkpoints']}, files)
    freeze, _, verified = verify_freeze(tmp_path, roster)
    protocol = verify_evaluation_protocol(tmp_path, freeze, verified, draws=2000, seed=73)
    assert protocol['primary_key'] == 'ordered_seed42_gap'
    assert protocol['natural_annotations_opened'] is False
