"""Fail-closed confirmation gate: immutable candidate, inputs, evaluation and scope."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA = 'bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(freeze_path):
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get('status') != 'candidate_frozen_before_confirmation':
        raise ValueError('candidate not frozen')
    if freeze.get('input_sha256') != INPUT_SHA or freeze.get('responses') != 64:
        raise ValueError('confirmation roster changed')
    if freeze.get('primary_score') != 'reviewed_source_risk' or freeze.get('endpoint') != 'all-token AUROC > 0.8':
        raise ValueError('frozen primary endpoint changed')
    required = ['next_iteration/local_grounding_review.py',
                'next_iteration/local_grounding_confirmation.py',
                'next_iteration/grounding_contrast.py',
                'next_iteration/grounding_contrast_evaluate.py',
                'next_iteration/confirmation_assess.py',
                'next_iteration/confirmation_reference_export.py',
                'next_iteration/confirmation_freeze_check.py',
                'scripts/run_local_grounding_confirmation.sh',
                'docs/P6_CONFIRMATION_RUN_20260914.md',
                'outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl',
                'outputs/P5_QWEN3_WEIGHT_IDENTITY_20260914.json',
                'outputs/p6_review_development_20260914_v1/manifest.json',
                'outputs/p6_review_development_20260914_v1/evaluation.json']
    if set(freeze.get('file_sha256', {})) != set(required):
        raise ValueError('incomplete freeze file identities')
    for relative in required:
        if sha(ROOT/relative) != freeze['file_sha256'][relative]:
            raise ValueError('frozen file changed: '+relative)
    if sha(ROOT/'outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl') != INPUT_SHA:
        raise ValueError('input identity differs from preregistered roster')
    return {'freeze_sha256':sha(freeze_path), 'verified':len(required), 'responses':64}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', required=True)
    args = p.parse_args()
    print(json.dumps(check(args.freeze)))


if __name__ == '__main__':
    main()
