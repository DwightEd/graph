"""Prepare immutable P7 confirmation only after complete development and review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .reasoned_confirmation_freeze_check import ROOT, INPUT_SHA, FILES, sha
from .confirmation_assess_scoped import frozen_records
from .reasoned_confirmation_contract import (validate_development, validate_main_decision,
    POPULATION, POPULATION_SETTINGS_SHA, POPULATION_COMPLETE_SHA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integrity-record', required=True,
                   help='Existing independent development review metadata, read by main before this command')
    p.add_argument('--main-decision',required=True,
                   help='Separate explicit completed-audit decision receipt, not automatic review approval')
    args = p.parse_args()
    out = ROOT/'outputs/P7_CONFIRMATION_FREEZE_20260914.json'
    if out.exists() or (ROOT/'outputs/p7_reasoned_confirmation_20260914_v1').exists():
        raise FileExistsError('freeze or confirmation already exists')
    dev = ROOT/'outputs/p7_reasoned_development_20260914_v1'
    manifest, records = frozen_records(dev)
    evaluation = json.loads((dev/'evaluation.json').read_text())
    validate_development(manifest,records,evaluation,sha(dev/'manifest.json'))
    if len(records)!=32 or evaluation['tokens']!=5170 or manifest['settings']['phase']!='development':
        raise ValueError('complete original development32 required')
    if evaluation['primary_score']!='reasoned_source_risk' or evaluation['prediction_manifest_sha256']!=sha(dev/'manifest.json'):
        raise ValueError('development evaluation identity')
    primary=evaluation['all_token']['reasoned_source_risk']
    if primary['auroc'] is None or primary['auroc']<=0.8:
        raise ValueError('prespecified development selection gate not passed; no confirmation launch')
    if manifest['code_sha256']!='f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f':
        raise ValueError('P7 scorer changed from pilot')
    audit_path=Path(args.integrity_record).resolve()
    audit=json.loads(audit_path.read_text())
    if not isinstance(audit,dict) or not audit:
        raise ValueError('missing independent review metadata')
    decision_path=Path(args.main_decision).resolve()
    validate_main_decision(json.loads(decision_path.read_text()),sha(audit_path),sha(dev/'manifest.json'),sha(dev/'evaluation.json'))
    if sha(POPULATION/'settings.json')!=POPULATION_SETTINGS_SHA or sha(POPULATION/'COMPLETE')!=POPULATION_COMPLETE_SHA:
        raise ValueError('population parent identity changed')
    confirm_rows=[json.loads(line) for line in (ROOT/'outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl').read_text().splitlines()]
    population=dict(root=str(POPULATION),settings_sha256=POPULATION_SETTINGS_SHA,complete_sha256=POPULATION_COMPLETE_SHA,
                    parent_manifest_sha256={str(r['id']):sha(POPULATION/'responses'/str(r['id'])/'manifest.json') for r in confirm_rows})
    result=dict(status='candidate_frozen_before_confirmation',created_utc=datetime.now(timezone.utc).isoformat(),
        sources=64,responses=128,official_split='test',input_sha256=INPUT_SHA,
        primary_score='reasoned_source_risk',endpoint='all-token AUROC > 0.8',
        method='unchanged P7 single sampled budgeted reasoning audit then raw word grounding; zero additional training',
        timing='retrospective full-source/full-answer external Qwen3; not online or original-generator causality',
        source_scope='64official-test sources disjoint byID/exactsource from R04 andspentP6; prior population measurements exist; pretraining contamination unknown',
        precision=manifest['precision'],label_ids=manifest['label_ids'],model_metadata_sha256=manifest['model_metadata_sha256'],
        settings=dict(batch_size=4,thinking_budget=1024,final_budget=384,
                      sampling=dict(do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.0),
                      one_sample=True,bootstrap=500,bootstrap_seed=20260914),
        development_all_token=evaluation['all_token'],development_within_answer=evaluation['within_answer'],
        development_bootstrap=evaluation['bootstrap'],
        selection_disclosure='Development original32 used for selection; full-token AUC>0.8 gate specified before P7 development results; this is NOT goal success. Main must read the independent review before executing this builder.',
        integrity_record=str(audit_path),integrity_record_sha256=sha(audit_path),
        main_decision=str(decision_path),main_decision_sha256=sha(decision_path),population=population,
        old_failures='P5 validation0.7474358560 and P6 full64 confirmation0.7816538805 retained; both are spent and not reused as unseen confirmation',
        confirmation_policy='all128 one endpoint; all predictions hash-frozen before selected128 annotation parse; no partial metrics, tuning, sign flip, filters or sample replacement',
        output='outputs/p7_reasoned_confirmation_20260914_v1',
        file_sha256={name:sha(ROOT/name) for name in FILES},freeze_builder_sha256=sha(__file__))
    if result['file_sha256']['outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl']!=INPUT_SHA:
        raise ValueError('confirmation inputs changed')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'freeze':str(out),'sha256':sha(out),'development_primary':primary}))


if __name__=='__main__':main()
