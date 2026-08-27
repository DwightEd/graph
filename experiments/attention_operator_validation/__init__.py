"""Pair-specific attention operator-code mechanism validation."""

from .features import OPERATOR_MODES, extract_answer_features
from .operators import (
    OperatorGeometry,
    apply_factorized_operator,
    load_factorized_basis,
    load_operator_geometry,
)
from .pair_codes import PairCodeField, build_pair_code_field

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
