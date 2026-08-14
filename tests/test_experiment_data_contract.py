import ast
import unittest
from pathlib import Path


class ExperimentDataContractTests(unittest.TestCase):
    def test_production_experiments_do_not_import_raw_attention_loaders(self):
        repository = Path(__file__).resolve().parents[1]
        roots = [
            repository / "experiments",
            repository / "attention_multiplex" / "attention_multiplex",
        ]
        violations = []
        forbidden_modules = {"cache", "formal_cache"}
        forbidden_names = {
            "load_attention_sample",
            "load_formal_sample",
            "read_formal_manifest",
        }
        for root in roots:
            for path in root.rglob("*.py"):
                if "tests" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module in forbidden_modules:
                            violations.append(f"{path}: imports raw module {module}")
                        for alias in node.names:
                            if alias.name in forbidden_names:
                                violations.append(f"{path}: imports {alias.name}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in forbidden_modules:
                                violations.append(
                                    f"{path}: imports raw module {alias.name}"
                                )
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
