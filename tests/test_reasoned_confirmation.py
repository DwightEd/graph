import ast
import json
from pathlib import Path
import tempfile
import unittest
from next_iteration import local_grounding_reasoned as dev
from next_iteration import reasoned_confirmation as conf
from next_iteration import reasoned_confirmation_assess as assess
from next_iteration import reasoned_confirmation_freeze_check as gate
from next_iteration import reasoned_confirmation_contract as contract


class ReasonedConfirmationTests(unittest.TestCase):
    def test_all_scoring_helpers_and_model_execution_are_identical(self):
        trees=[ast.parse(Path(m.__file__).read_text()) for m in (dev,conf)]
        functions=['audit_seed','final_context','units','local_sentence','messages','audit_messages',
                   'add_audit','common_prefix','align_scores']
        for name in functions:
            nodes=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]),name)
        mains=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main') for t in trees]
        blocks=[next(n for n in m.body if isinstance(n,ast.Try)) for m in mains]
        self.assertEqual(ast.dump(blocks[0]),ast.dump(blocks[1]))
        self.assertEqual(dev.SYSTEM,conf.SYSTEM)
        self.assertEqual(dev.AUDIT_SYSTEM,conf.AUDIT_SYSTEM)
        self.assertEqual(dev.SAMPLING,conf.SAMPLING)

    def test_missing_freeze_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):gate.check(Path(d)/'absent.json')

    def test_wrong_denominator_refused_before_labels(self):
        manifest=dict(input_sha256=gate.INPUT_SHA,settings={'phase':'validation'})
        with self.assertRaisesRegex(ValueError,'fixed128'):
            assess.check_boundary(manifest,[{'source_id':str(i//2)} for i in range(64)],{})

    def test_started_before_freeze_refused(self):
        manifest=dict(input_sha256=gate.INPUT_SHA,settings={'phase':'validation'},started_unix=1)
        with self.assertRaisesRegex(ValueError,'after candidate freeze'):
            assess.check_boundary(manifest,[{'source_id':str(i//2)} for i in range(128)],
                                  {'created_utc':'2026-09-14T00:00:00+00:00'})

    def test_fixed_input_identity_differs_from_spent_cohorts(self):
        self.assertEqual(conf.INPUT_SHA,gate.INPUT_SHA)
        self.assertEqual(len(gate.INPUT_SHA),64)
        self.assertNotEqual(gate.INPUT_SHA,dev.INPUT_SHA)
        self.assertNotEqual(gate.INPUT_SHA,'bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101')

    def test_metric_computation_unchanged_from_scoped_p6(self):
        from next_iteration import confirmation_assess_scoped as parent
        trees=[ast.parse(Path(m.__file__).read_text()) for m in (parent,assess)]
        mains=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main') for t in trees]
        blocks=[]
        for main in mains:
            start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                       and any(isinstance(v,ast.Name) and v.id=='labels' for v in ast.walk(n.targets[0])))
            end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                     and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='report')
            blocks.append(ast.dump(ast.Module(body=main.body[start:end],type_ignores=[])))
        self.assertEqual(*blocks)

    def development_fixture(self):
        ids=sorted(contract.DEV_IDS)
        rows=[dict(id=rid,source_id=str(i//2),token_ids=[0]*(160 if i<31 else 210)) for i,rid in enumerate(ids)]
        manifest=dict(settings={'phase':'development'},planned_ids=ids)
        evaluation=dict(responses=32,sources=16,tokens=5170,executed_code_sha256=contract.DEV_EVALUATOR_SHA,
            annotation_sha256=contract.ANNOTATION_SHA,prediction_manifest_sha256='manifest',
            primary_score='reasoned_source_risk',per_response=[{'id':rid} for rid in ids])
        return manifest,rows,evaluation

    def test_development_contract_requires_exact_ids_and_evaluator(self):
        m,rows,e=self.development_fixture()
        contract.validate_development(m,rows,e,'manifest')
        rows[0]['id']='substituted'
        with self.assertRaisesRegex(ValueError,'exact original'):contract.validate_development(m,rows,e,'manifest')
        m,rows,e=self.development_fixture();e['executed_code_sha256']='other'
        with self.assertRaisesRegex(ValueError,'evaluator'):contract.validate_development(m,rows,e,'manifest')

    def test_nonempty_or_pending_review_does_not_authorize(self):
        for decision in [{'x':1},{'decision':'authorize_confirmation','stage':'pending'}]:
            with self.assertRaisesRegex(ValueError,'completed main decision'):
                contract.validate_main_decision(decision,'audit','manifest','evaluation')

    def test_explicit_main_receipt_and_unresolved_findings(self):
        with tempfile.TemporaryDirectory() as d:
            response=Path(d)/'response.md';response.write_text('Independent completed audit fixture')
            decision=dict(decision='authorize_confirmation',actor='/root',stage='complete_development_audit',
                responses=32,sources=16,tokens=5170,independent_audit_sha256='audit',
                prediction_manifest_sha256='manifest',evaluation_sha256='evaluation',
                development_ids=sorted(contract.DEV_IDS),review_read_by_main=True,
                independent_recomputation_complete=True,unresolved_material_findings=[],
                finding_disposition={'fixture':'resolved'},independent_response_path=str(response),
                independent_response_sha256=contract.sha(response))
            contract.validate_main_decision(decision,'audit','manifest','evaluation')
            decision['unresolved_material_findings']=['F1']
            with self.assertRaisesRegex(ValueError,'unresolved material'):
                contract.validate_main_decision(decision,'audit','manifest','evaluation')
            decision['unresolved_material_findings']=[];decision['review_read_by_main']=False
            with self.assertRaisesRegex(ValueError,'audit/read receipt'):
                contract.validate_main_decision(decision,'audit','manifest','evaluation')

    def test_reference_provenance_and_exact_schema_contract(self):
        self.assertEqual(len(contract.EXPECTED_POPULATION_CONTROLS),10)
        population=dict(settings_sha256='settings',complete_sha256='complete',parent_manifest_sha256={'1':'parent'})
        freeze=dict(population=population,file_sha256={'next_iteration/reasoned_confirmation_reference_export.py':'exporter'})
        reference=dict(input_sha256=gate.INPUT_SHA,code_sha256='exporter',model_forwards=0,
            settings=dict(phase='validation',population=str(contract.POPULATION)),
            parent_settings_sha256='settings',parent_complete_sha256='complete',
            parent_manifests={'1':'parent'},planned_ids=['1'])
        contract.validate_reference_manifest(reference,freeze,gate.INPUT_SHA)
        for key,bad in [('code_sha256','other'),('parent_manifests',{'1':'different'}),('model_forwards',1)]:
            changed=dict(reference);changed[key]=bad
            with self.assertRaises(ValueError):contract.validate_reference_manifest(changed,freeze,gate.INPUT_SHA)


if __name__=='__main__':unittest.main()
