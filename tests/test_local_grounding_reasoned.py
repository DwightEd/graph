import ast
from pathlib import Path
import unittest
from next_iteration import local_grounding_reasoned as p7
from next_iteration import local_grounding_review as p6
from next_iteration.development_assess_scoped import DEV_IDS


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class ReasonedAuditTests(unittest.TestCase):
    def test_natural_close_is_retained_without_injection(self):
        ids, meta = p7.final_context(FakeTokenizer(), [1, 2], [3, 9], [0], 9)
        self.assertEqual(ids, [1, 2, 3, 9, 10, 10])
        self.assertFalse(meta['forced_close'])
        self.assertEqual(meta['raw_reasoning_ids'], [3, 9])

    def test_budget_close_keeps_generated_tokens_and_records_injection(self):
        ids, meta = p7.final_context(FakeTokenizer(), [1, 2], [3, 4], [0], 9)
        self.assertTrue(meta['forced_close'])
        self.assertEqual(meta['raw_reasoning_ids'], [3, 4])
        self.assertEqual(ids[:4], [1, 2, 3, 4])
        self.assertIn('</think>', ''.join(map(chr, meta['appended_context_ids'])))

    def test_premature_eos_removal_is_only_context_not_raw_record(self):
        ids, meta = p7.final_context(FakeTokenizer(), [1], [3, 0], [0], 9)
        self.assertEqual(meta['raw_reasoning_ids'], [3, 0])
        self.assertEqual(meta['retained_reasoning_ids'], [3])
        self.assertEqual(ids[:2], [1, 3])
        self.assertTrue(meta['forced_close'])

    def test_order_independent_one_sample_seed(self):
        a = p7.audit_seed('source', 'instruction', 'answer')
        p7.audit_seed('another', 'task', 'response')
        self.assertEqual(a, p7.audit_seed('source', 'instruction', 'answer'))
        self.assertNotEqual(a, p7.audit_seed('different', 'instruction', 'answer'))
        self.assertLess(a, 2**31)

    def test_localized_score_and_prompts_unchanged(self):
        self.assertEqual(p7.SYSTEM, p6.SYSTEM)
        self.assertEqual(p7.AUDIT_SYSTEM, p6.AUDIT_SYSTEM)
        a, b = [ast.parse(Path(x.__file__).read_text()) for x in (p7, p6)]
        functions = ['units', 'local_sentence', 'messages', 'audit_messages', 'add_audit',
                     'common_prefix', 'align_scores', 'verify', 'summarize', 'fp32_linear', 'tokenize']
        for name in functions:
            na = next(n for n in ast.walk(a) if isinstance(n, ast.FunctionDef) and n.name == name)
            nb = next(n for n in ast.walk(b) if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(na), ast.dump(nb), name)

    def test_fixed_budgets_and_development_scope(self):
        self.assertEqual((p7.THINK_BUDGET, p7.FINAL_BUDGET), (1024, 384))
        self.assertEqual(len(DEV_IDS), 32)
        self.assertIn('17019', DEV_IDS)
        self.assertNotIn('14325', DEV_IDS)
        self.assertEqual(p7.SAMPLING, dict(do_sample=True, temperature=.6, top_p=.95, top_k=20, min_p=0.0))


if __name__ == '__main__':
    unittest.main()
