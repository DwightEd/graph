"""One-shot P6 candidate freeze AFTER dev completion, BEFORE new64 scoring."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/share/home/tm902089733300000/a903202310/lys')
G = B/'research/graph'
OUT = G/'outputs/P6_CONFIRMATION_FREEZE_20260914.json'
P = G/'outputs/p6_review_development_20260914_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if OUT.exists() or (G/'outputs/p6_review_confirmation_20260914_v1').exists():
    raise FileExistsError('freeze or confirmation output already exists')
m=json.loads((P/'manifest.json').read_text())
e=json.loads((P/'evaluation.json').read_text())
assert m['complete'] and not m.get('error') and len(m['planned_ids'])==32 and m['completed_ids']==m['planned_ids']
assert e['prediction_manifest_sha256']==sha(P/'manifest.json')
assert m['code_sha256']=='167d7199508cc2e55dad878e1e2f173b562ea95692ab835175fd58ede03b5b38'
assert e['tokens']==5170 and e['primary_score']=='reviewed_source_risk'
assert e['all_token']['reviewed_source_risk']['auroc']>e['all_token']['localized_source_risk']['auroc']
for name,h in m['output_sha256'].items():
    assert sha(P/name)==h
files=['next_iteration/local_grounding_review.py',
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
record=dict(status='candidate_frozen_before_confirmation',created_utc=datetime.now(timezone.utc).isoformat(),
    responses=64,sources=32,official_split='test',
    input_sha256='bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101',
    primary_score='reviewed_source_risk',endpoint='all-token AUROC > 0.8',
    timing='retrospective full-source/full-answer; not online or original-generator causality',
    method='same frozen Qwen3 generates fallible evidence audit then identical word risk; no natural-label training or score blend',
    precision=m['precision'],label_ids=m['label_ids'],model_metadata_sha256=m['model_metadata_sha256'],
    source_scope='fixed32 official-test sources disjoint from R04 by ID/content; prior population measurements exist; no globally pristine claim',
    development_primary=e['all_token']['reviewed_source_risk'],
    development_p5=e['all_token']['localized_source_risk'],
    development_within_answer=e['within_answer']['reviewed_source_risk'],
    development_bootstrap=e['bootstrap']['primary_absolute_95_ci'],
    selection_disclosure='Candidate retained after improved dev point AUROC/AP and within-answer; paired gain CI crosses zero. Confirmation is required; no success claim yet.',
    old_validation='P5 R04 validation0.747435856 failed; not reused to verify P6. Only aggregate P5 validation failure observed for this revision.',
    confirmation_rule='One primary64-response denominator, complete hash-frozen predictions before any label join; no dropping, flipping, task selection, or tuning on confirmation.',
    output='outputs/p6_review_confirmation_20260914_v1',
    failure_policy='Retain failure and do not reinterpret or repeatedly reuse this cohort as untouched.',
    file_sha256={name:sha(G/name) for name in files},
    freeze_builder_sha256=sha(Path(__file__)))
assert record['file_sha256']['outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl']==record['input_sha256']
with OUT.open('x') as f:
    json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps({'freeze':str(OUT),'sha256':sha(OUT),'created_utc':record['created_utc'],'development':record['development_primary']}))
