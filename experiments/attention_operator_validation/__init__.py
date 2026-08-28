"""Pair-specific attention operator-code mechanism validation."""

from importlib import import_module

__all__ = [
    "OPERATOR_MODES",
    "OperatorGeometry",
    "apply_factorized_operator",
    "load_factorized_basis",
    "PairCodeField",
    "build_pair_code_field",
    "extract_answer_features",
    "load_operator_geometry",
]


_EXPORT_MODULE = {
    "OPERATOR_MODES": ".features",
    "extract_answer_features": ".features",
    "OperatorGeometry": ".operators",
    "apply_factorized_operator": ".operators",
    "load_factorized_basis": ".operators",
    "load_operator_geometry": ".operators",
    "PairCodeField": ".pair_codes",
    "build_pair_code_field": ".pair_codes",
}


def __getattr__(name: str):
    """Keep artifact/evaluation imports usable without importing PyTorch."""

    if name not in _EXPORT_MODULE:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORT_MODULE[name], __name__), name)
    globals()[name] = value
    return value
