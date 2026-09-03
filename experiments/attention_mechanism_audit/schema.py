"""Shared scientific axes for the attention mechanism audit.

This module contains only fixed coordinate systems. Keeping these definitions in
one place prevents capture, graph exposure, scoring, and evaluation from
silently disagreeing about branch, register, role, or stage order.
"""

ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
EVIDENCE, OTHER_PROMPT, HISTORY, SELF = range(len(ROLE_NAMES))

BRANCH_NAMES = ("full", "no_evidence", "no_history", "no_evidence_history")
BRANCH_REMOVALS = (None, "evidence", "history", "both")
FULL, NO_EVIDENCE, NO_HISTORY, NO_EVIDENCE_HISTORY = range(len(BRANCH_NAMES))

REGISTER_NAMES = ("evidence_adoption", "autonomous_history")
REGISTER_BRANCH_PAIRS = ((FULL, NO_EVIDENCE), (NO_EVIDENCE, NO_EVIDENCE_HISTORY))
EVIDENCE_ADOPTION, AUTONOMOUS_HISTORY = range(len(REGISTER_NAMES))

REGISTER_STAGE_NAMES = (
    "input_state",
    "attention_write",
    "mlp_write",
    "output_state",
)
INPUT_STATE, ATTENTION_WRITE, MLP_WRITE, OUTPUT_STATE = range(
    len(REGISTER_STAGE_NAMES)
)

# Residual-space vectors used by the shortcut-route audit.  The first five are
# observed branch-defined quantities.  The final two are an endpoint-rewiring
# control that swaps adjacent response carriers while keeping the target row,
# head-specific coefficients, and source-value multiset fixed.
SHORTCUT_VECTOR_NAMES = (
    "full_history_write",
    "direct_evidence_write",
    "evidence_relay_carrier",
    "evidence_relay_gate",
    "autonomous_history_write",
    "rewired_evidence_relay_carrier",
    "rewired_evidence_relay_gate",
)
(
    FULL_HISTORY_WRITE,
    DIRECT_EVIDENCE_WRITE,
    EVIDENCE_RELAY_CARRIER,
    EVIDENCE_RELAY_GATE,
    AUTONOMOUS_HISTORY_WRITE,
    REWIRED_EVIDENCE_RELAY_CARRIER,
    REWIRED_EVIDENCE_RELAY_GATE,
) = range(len(SHORTCUT_VECTOR_NAMES))

SHORTCUT_REWIRE = "adjacent_response_endpoint_swap"
