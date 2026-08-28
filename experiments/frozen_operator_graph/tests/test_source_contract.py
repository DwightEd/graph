import ast
from pathlib import Path

from ..config import GraphConstructionConfig


def test_production_source_has_no_label_api_calls_or_trainable_graph_layers():
    root = Path(__file__).resolve().parents[1]
    forbidden_label_calls = {"labels", "prepare_evaluation_labels", "response_labels"}
    trainable_constructors = {
        "Linear",
        "Embedding",
        "GRU",
        "GRUCell",
        "GraphConv",
        "GATConv",
        "Parameter",
    }
    violations = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute):
                name = function.attr
            elif isinstance(function, ast.Name):
                name = function.id
            else:
                continue
            if name in forbidden_label_calls:
                violations.append(f"{path.name}:{node.lineno}:label call {name}")
            if name in trainable_constructors:
                violations.append(f"{path.name}:{node.lineno}:trainable layer {name}")
    assert violations == []


def test_default_graph_is_the_complete_causal_graph():
    config = GraphConstructionConfig()
    assert config.route_mass_retention == 1.0
    assert config.value_energy_retention == 1.0
