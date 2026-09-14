"""Pre-confirmation decision and baseline identity contracts; no annotation I/O."""
from pathlib import Path
from .development_assess_scoped import DEV_IDS
from .grounding_contrast_evaluate import sha

ROOT=Path(__file__).resolve().parents[1]
ANNOTATION_SHA='e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073'
DEV_EVALUATOR_SHA='9627dba28c3882a7c2de869af1c406ae0a873991dff9c357e93cfcd333c9d68b'
POPULATION=ROOT.parents[1]/'research/reanchor/outputs/ragtruth_population_20260912'
POPULATION_SETTINGS_SHA='b5ce554885534f89b52a71aae717fed0daf9713a80ea0f7862915551135eb7c8'
POPULATION_COMPLETE_SHA='5ba0d7c503f9dcf7b301332207781fafa46551e34f72bb0d51f28b6a7fe9fc1d'
EXPECTED_POPULATION_CONTROLS={
    'population_native_entropy','population_native_nll','population_negative_margin',
    'population_source_small_js','population_history_small_js','population_source_support',
    'population_history_support','population_history_minus_source_support',
    'population_source_permute_js','population_mlp_small_js',
}

def validate_development(manifest,records,evaluation,manifest_sha):
    if len(records)!=32 or {str(r['id']) for r in records}!=DEV_IDS:
        raise ValueError('exact original development32 IDs required')
    if len({str(r['source_id']) for r in records})!=16 or sum(len(r['token_ids']) for r in records)!=5170:
        raise ValueError('original development16sources/5170tokens required')
    if manifest['settings']['phase']!='development' or set(map(str,manifest['planned_ids']))!=DEV_IDS:
        raise ValueError('development phase/manifest IDs')
    if (evaluation.get('responses'),evaluation.get('sources'),evaluation.get('tokens'))!=(32,16,5170):
        raise ValueError('development evaluation denominator')
    if evaluation.get('executed_code_sha256')!=DEV_EVALUATOR_SHA or evaluation.get('annotation_sha256')!=ANNOTATION_SHA:
        raise ValueError('scoped evaluator/annotation provenance')
    if evaluation.get('prediction_manifest_sha256')!=manifest_sha or evaluation.get('primary_score')!='reasoned_source_risk':
        raise ValueError('evaluation/manifest/primary binding')
    if {str(r['id']) for r in evaluation.get('per_response',[])}!=DEV_IDS:
        raise ValueError('evaluation per-response scope')

def validate_main_decision(decision,audit_sha,manifest_sha,evaluation_sha):
    # A review is advisory. This separate, explicit main-agent receipt records
    # reading/resolution of the completed review; no LLM verdict auto-approves.
    expected=dict(decision='authorize_confirmation',actor='/root',stage='complete_development_audit',
                  responses=32,sources=16,tokens=5170,independent_audit_sha256=audit_sha,
                  prediction_manifest_sha256=manifest_sha,evaluation_sha256=evaluation_sha)
    if any(decision.get(k)!=v for k,v in expected.items()):
        raise ValueError('explicit completed main decision identity required')
    if set(decision.get('development_ids',[]))!=DEV_IDS:
        raise ValueError('main decision exact original32 scope')
    if decision.get('review_read_by_main') is not True or decision.get('independent_recomputation_complete') is not True:
        raise ValueError('completed independent audit/read receipt required')
    if decision.get('unresolved_material_findings')!=[]:
        raise ValueError('unresolved material findings')
    if not isinstance(decision.get('finding_disposition'),dict) or not decision['finding_disposition']:
        raise ValueError('explicit finding resolutions required')
    response=Path(decision.get('independent_response_path',''))
    if not response.is_file() or sha(response)!=decision.get('independent_response_sha256'):
        raise ValueError('completed independent response artifact binding')

def validate_reference_manifest(manifest,freeze,input_sha):
    expected=freeze['population']
    if manifest.get('input_sha256')!=input_sha or manifest.get('code_sha256')!=freeze['file_sha256']['next_iteration/reasoned_confirmation_reference_export.py']:
        raise ValueError('reference input/exporter provenance')
    if manifest.get('model_forwards')!=0 or manifest['settings'].get('phase')!='validation':
        raise ValueError('reference phase/zero-forward provenance')
    if Path(manifest['settings'].get('population','')).resolve()!=POPULATION.resolve():
        raise ValueError('reference population root')
    if manifest.get('parent_settings_sha256')!=expected['settings_sha256'] or manifest.get('parent_complete_sha256')!=expected['complete_sha256']:
        raise ValueError('reference population identity')
    if manifest.get('parent_manifests')!=expected['parent_manifest_sha256']:
        raise ValueError('reference exact parent manifest coverage/identity')
    if set(map(str,manifest.get('planned_ids',[])))!=set(expected['parent_manifest_sha256']):
        raise ValueError('reference parent scope')
