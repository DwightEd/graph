"""Reviewed corrections applied before public pipeline modules are imported.

The initial feature implementation used the wrong parenthesization for the
head-code effective number and grouped those summaries into the routing-only
probe. Keeping the correction in a small module preserves the frozen feature
schema while avoiding duplicated mechanism extraction code.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from . import evaluate as _evaluate
from . import features as _features
from .pair_codes import PAIR_RETAINED, PairCodeField


_BASE_EXTRACT = _features.extract_answer_features


def _role_effective_heads(graph, field: PairCodeField, prompt: bool) -> float:
    retained = field.kind == PAIR_RETAINED
    role = field.source < graph.response_start
    selected = retained & (role if prompt else ~role)
    row = field.layer * graph.response_count + (field.target - graph.response_start)
    output = torch.zeros((graph.layer_count, graph.response_count))
    for group in torch.unique(row[selected]).tolist():
        current = selected & (row == int(group)) & (field.magnitude > 0)
        weight = field.magnitude[current]
        if not len(weight) or float(weight.sum().item()) <= 0:
            continue
        probability = weight / weight.sum()
        code = field.direction[current]
        effective = (
            -(code * code.clamp_min(1e-12).log()).sum(dim=1)
        ).exp()
        layer = int(group) // graph.response_count
        response = int(group) % graph.response_count
        output[layer, response] = (probability * effective).sum()
    return float(output.mean().item())


def extract_answer_features(*args, **kwargs):
    graph = args[0] if len(args) > 0 else kwargs["graph"]
    field = args[1] if len(args) > 1 else kwargs["field"]
    result = _BASE_EXTRACT(*args, **kwargs)
    result["prompt_code_effective_heads_mean"] = _role_effective_heads(
        graph, field, True
    )
    result["history_code_effective_heads_mean"] = _role_effective_heads(
        graph, field, False
    )
    return result


def column_groups(feature_names: tuple[str, ...]) -> OrderedDict[str, list[int]]:
    raw_head_summary_names = {
        "prompt_code_effective_heads_mean",
        "history_code_effective_heads_mean",
    }
    routing = [
        index
        for index, name in enumerate(feature_names)
        if not name.startswith(_evaluate.MODE_PREFIXES)
        and name not in raw_head_summary_names
    ]
    raw_head_summary = [
        index
        for index, name in enumerate(feature_names)
        if name in raw_head_summary_names
    ]
    identity = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("identity_")
    ]
    raw = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_raw_")
    ]
    normalized = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_normalized_")
    ]
    permuted = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_permuted_")
    ]
    return OrderedDict(
        (
            ("routing_only", routing),
            ("routing_plus_head_entropy", routing + raw_head_summary),
            ("routing_plus_raw_head_code", routing + identity),
            ("routing_plus_operator_raw", routing + raw),
            ("routing_plus_operator_normalized", routing + normalized),
            ("routing_plus_operator_permuted", routing + permuted),
            ("operator_normalized_only", normalized),
        )
    )


def install() -> None:
    _features.extract_answer_features = extract_answer_features
    _evaluate._column_groups = column_groups
