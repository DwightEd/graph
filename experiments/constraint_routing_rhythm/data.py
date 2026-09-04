"""Compatibility exports for the shared RAGTruth alignment module."""

from experiments.common.ragtruth_alignment import (
    HISTORICAL_SYSTEM_PROMPT,
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
    render_historical_prompt,
    token_array,
)

__all__ = [
    "HISTORICAL_SYSTEM_PROMPT",
    "TASK_TYPES",
    "build_evidence_mask",
    "canonical_task_type",
    "load_source_info",
    "render_historical_prompt",
    "token_array",
]
