"""Fail-closed P7 confirmation identities; usable by launcher, scorer and evaluator."""
import argparse
import json
from pathlib import Path
from .grounding_contrast_evaluate import sha

ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA = 'cb16edbc11ad1d5cf637481d038d2ac2bec0f72136df97fc461103b9f06d196e'
FILES = [
    'next_iteration/local_grounding_reasoned.py',
    'next_iteration/reasoned_confirmation.py',
    'next_iteration/reasoned_confirmation_assess.py',
    'next_iteration/reasoned_confirmation_reference_export.py',
    'next_iteration/reasoned_confirmation_freeze_check.py',
    'next_iteration/reasoned_confirmation_contract.py',
    'next_iteration/freeze_p7_confirmation.py',
    'next_iteration/reasoned_confirmation_roster.py',
    'next_iteration/development_assess_scoped.py',
    'next_iteration/grounding_contrast.py',
    'next_iteration/grounding_contrast_evaluate.py',
    'next_iteration/confirmation_assess_scoped.py',
    'scripts/run_reasoned_confirmation.sh',
    'tests/test_reasoned_confirmation.py',
    'docs/P7_REASONED_AUDIT_PROTOCOL.md',
    'docs/P7_CONFIRMATION_ROSTER_PROTOCOL.md',
    'docs/P7_CONFIRMATION_RUN_20260914.md',
    'outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl',
    'outputs/p7_confirmation_roster_20260914_v1/manifest.json',
    'outputs/p7_confirmation_roster_20260914_v1/roster.json',
    'outputs/P5_QWEN3_WEIGHT_IDENTITY_20260914.json',
    'outputs/p7_reasoned_pilot_20260914_v1/manifest.json',
    'outputs/p7_reasoned_development_20260914_v1/manifest.json',
    'outputs/p7_reasoned_development_20260914_v1/evaluation.json',
]


def check(path):
    freeze = json.loads(Path(path).read_text())
    if freeze.get('status') != 'candidate_frozen_before_confirmation':
        raise ValueError('P7 candidate not frozen')
    if freeze.get('input_sha256') != INPUT_SHA or freeze.get('responses') != 128 or freeze.get('sources') != 64:
        raise ValueError('fixed128 roster changed')
    if freeze.get('primary_score') != 'reasoned_source_risk' or freeze.get('endpoint') != 'all-token AUROC > 0.8':
        raise ValueError('primary endpoint changed')
    if set(freeze.get('file_sha256', {})) != set(FILES):
        raise ValueError('incomplete frozen identities')
    for relative in FILES:
        if sha(ROOT / relative) != freeze['file_sha256'][relative]:
            raise ValueError('frozen file changed: ' + relative)
    for key in ['integrity_record','main_decision']:
        if sha(freeze[key]) != freeze[key+'_sha256']:
            raise ValueError('frozen decision/review receipt changed: '+key)
    from .reasoned_confirmation_contract import validate_main_decision
    validate_main_decision(json.loads(Path(freeze['main_decision']).read_text()),
        freeze['integrity_record_sha256'],
        freeze['file_sha256']['outputs/p7_reasoned_development_20260914_v1/manifest.json'],
        freeze['file_sha256']['outputs/p7_reasoned_development_20260914_v1/evaluation.json'])
    if sha(ROOT / 'outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl') != INPUT_SHA:
        raise ValueError('input identity mismatch')
    return freeze


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', required=True)
    args = p.parse_args()
    result = check(args.freeze)
    print(json.dumps({'verified': len(FILES), 'responses': 128, 'freeze_sha256': sha(args.freeze)}))
