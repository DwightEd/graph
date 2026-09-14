"""Additive annotation-scope correction proof; never alter original candidate freeze."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .confirmation_assess_scoped import selected_annotations, check_scope_addendum

B = Path('/share/home/tm902089733300000/a903202310/lys')
G = B/'research/graph'
D = G/'outputs/p6_review_development_20260914_v1'
OUT = G/'outputs/P6_EVALUATION_SCOPE_ADDENDUM_20260914.json'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if OUT.exists() or (G/'outputs/p6_review_confirmation_20260914_v1/evaluation.json').exists():
    raise FileExistsError('scope freeze already exists or confirmation evaluated')
m=json.loads((D/'manifest.json').read_text())
assert m['complete'] and m['planned_ids']==m['completed_ids'] and len(m['planned_ids'])==32
gt=selected_annotations(B/'data/RAGTruth/dataset/response.jsonl', set(m['planned_ids']))
y=[]; risk=[]
for rid in m['planned_ids']:
    path=D/f'response_{rid}.json'
    assert sha(path)==m['output_sha256'][path.name]
    r=json.loads(path.read_text());a=gt[rid]
    assert hashlib.sha256(a['response'].encode()).hexdigest()==r['response_sha256']
    for lo,hi in r['offsets']:
        y.append(int(any(lo<s['end'] and hi>s['start'] for s in a['labels'])))
    risk.extend(r['scores']['reviewed_source_risk'])
metrics=dict(auroc=roc_auc_score(y,risk),auprc=average_precision_score(y,risk))
e=json.loads((D/'evaluation.json').read_text())['all_token']['reviewed_source_risk']
assert max(abs(metrics[k]-e[k]) for k in metrics)<1e-12
note=dict(created_utc=datetime.now(timezone.utc).isoformat(),
    change_scope='annotation_ID_first_parsing_only_no_metric_or_score_change',
    parent_freeze_sha256=sha(G/'outputs/P6_CONFIRMATION_FREEZE_20260914.json'),
    original_evaluator_sha256=sha(G/'next_iteration/confirmation_assess.py'),
    scoped_evaluator_sha256=sha(G/'next_iteration/confirmation_assess_scoped.py'),
    tests_sha256=sha(G/'tests/test_scoped_annotations.py'),
    protocol_sha256=sha(G/'docs/P6_EVALUATION_SCOPE_ADDENDUM_20260914.md'),
    builder_sha256=sha(__file__),
    regression=dict(scope='original development32 only; no confirmation labels',
        selected_annotation_rows=len(gt),tokens=len(y),positives=sum(y),metrics=metrics,
        max_metric_difference=max(abs(metrics[k]-e[k]) for k in metrics)),
    tests='20 targeted CPU tests passed; entire post-label numerical-metric AST identical; unselected invalid-label JSON never decoded',
    access_disclosure='Original evaluator decoded then discarded unrelated JSON rows; no unrelated labels entered scores, tuning, output or metrics. New code avoids unnecessary decoding.',
    original_freeze_and_scoring_unchanged=True,confirmation_labels_joined=False,
    scoring_action='running full64 scoring continues unchanged, no rerun',
    replacement_module='next_iteration.confirmation_assess_scoped')
with OUT.open('x') as f:
    json.dump(note,f,indent=2,allow_nan=False)
check_scope_addendum(OUT)
print(json.dumps({'addendum':str(OUT),'sha256':sha(OUT),'regression':note['regression']}))
