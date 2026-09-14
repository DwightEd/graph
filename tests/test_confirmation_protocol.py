import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from next_iteration import local_grounding_confirmation as confirmation
from next_iteration import local_grounding_review as development
from next_iteration import confirmation_assess
from next_iteration import confirmation_freeze_check


def tree(module):
    return ast.parse(Path(module.__file__).read_text())


def function(node, name):
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)


class TestConfirmationProtocol(unittest.TestCase):
    def test_parent_is_frozen(self):
        self.assertEqual(hashlib.sha256(Path(development.__file__).read_bytes()).hexdigest(),
                         confirmation.PARENT_CODE_SHA)

    def test_all_scoring_functions_and_model_execution_identical(self):
        parent, child = tree(development), tree(confirmation)
        for node in parent.body:
            if isinstance(node, ast.FunctionDef) and node.name != 'main':
                self.assertEqual(ast.dump(node), ast.dump(function(child, node.name)), node.name)
        # Complete try block includes model loading, all nested scoring/generation
        # functions, canaries, every natural forward, alignment and output writing.
        ptry = next(n for n in function(parent, 'main').body if isinstance(n, ast.Try))
        ctry = next(n for n in function(child, 'main').body if isinstance(n, ast.Try))
        self.assertEqual(ast.dump(ptry), ast.dump(ctry))
        for name in ('SYSTEM', 'AUDIT_SYSTEM', 'CANARIES'):
            self.assertEqual(getattr(development, name), getattr(confirmation, name))

    def test_evaluator_refuses_other_rosters_before_labels(self):
        with patch.object(confirmation_assess, 'frozen_records', return_value=(
                {'input_sha256':'old-roster'}, [])):
            with patch('sys.argv', ['assess','--predictions','unused',
                       '--annotations','not-read','--output','uncreated-confirmation-report.json']):
                with self.assertRaisesRegex(ValueError, 'not fixed confirmation'):
                    confirmation_assess.main()

    def test_evaluator_refuses_partial64_before_labels(self):
        m = dict(input_sha256=confirmation.INPUT_SHA, settings={'phase':'validation'})
        with patch.object(confirmation_assess, 'frozen_records', return_value=(m, [])):
            with patch('sys.argv', ['assess','--predictions','unused',
                       '--annotations','not-read','--output','uncreated-confirmation-report.json']):
                with self.assertRaisesRegex(ValueError, 'incomplete fixed64'):
                    confirmation_assess.main()

    def test_unfrozen_candidate_cannot_launch(self):
        with patch.object(Path, 'read_text', return_value='{"status":"not_frozen"}'):
            with self.assertRaisesRegex(ValueError, 'candidate not frozen'):
                confirmation_freeze_check.check('unused.json')

    def test_frozen_claim_with_wrong_denominator_cannot_launch(self):
        text = '{"status":"candidate_frozen_before_confirmation","input_sha256":"old","responses":32}'
        with patch.object(Path, 'read_text', return_value=text):
            with self.assertRaisesRegex(ValueError, 'confirmation roster changed'):
                confirmation_freeze_check.check('unused.json')


if __name__ == '__main__':
    unittest.main()
