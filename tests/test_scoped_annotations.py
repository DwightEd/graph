import ast
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from next_iteration import confirmation_assess as original
from next_iteration import confirmation_assess_scoped as scoped


class TestScopedAnnotations(unittest.TestCase):
    def parse(self, content, wanted):
        with patch.object(Path, 'open', return_value=io.BytesIO(content)):
            return scoped.selected_annotations('unused', set(wanted))

    def test_only_selected_row_decoded(self):
        content = b'{"id":"2","labels": NOT_VALID_JSON}\n{"id":"1","labels":[]}\n'
        real_loads = json.loads
        with patch.object(scoped.json, 'loads', wraps=real_loads) as loads:
            out = self.parse(content, ['1'])
        self.assertEqual(out, {'1':{'id':'1','labels':[]}})
        self.assertEqual(loads.call_count, 1)
        self.assertIn(b'"id":"1"', loads.call_args.args[0])

    def test_wrong_schema_rejected_without_parsing(self):
        with patch.object(scoped.json, 'loads', side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(ValueError, 'ID-first schema'):
                self.parse(b'{"response":"id", "id":"1"}\n', ['1'])

    def test_numeric_id(self):
        self.assertEqual(self.parse(b' {"id":12,"labels":[]}\n', ['12'])['12']['id'], 12)

    def test_duplicate_and_missing(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.parse(b'{"id":"1"}\n{"id":"1"}\n', ['1'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.parse(b'{"id":"2"}\n', ['1'])

    def test_entire_metric_computation_ast_unchanged(self):
        def tail(module):
            tree = ast.parse(Path(module.__file__).read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
            start = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                         and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='names')
            return [ast.dump(n) for n in main.body[start:]]
        self.assertEqual(tail(original), tail(scoped))


if __name__=='__main__':
    unittest.main()
