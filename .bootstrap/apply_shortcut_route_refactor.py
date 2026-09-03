from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# capture.py: shared schema + raw shortcut-route Gram capture
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/capture.py"
replace_once(
    path,
    '''from torch.nn import functional as F

ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
EVIDENCE, OTHER_PROMPT, HISTORY, SELF = range(len(ROLE_NAMES))

BRANCH_NAMES = ("full", "no_evidence", "no_history", "no_evidence_history")
BRANCH_REMOVALS = (None, "evidence", "history", "both")
REGISTER_NAMES = ("evidence_adoption", "autonomous_history")
REGISTER_BRANCH_PAIRS = ((0, 1), (1, 3))
REGISTER_STAGE_NAMES = (
    "input_state",
    "attention_write",
    "mlp_write",
    "output_state",
)
''',
    '''from torch.nn import functional as F

from .schema import (
    BRANCH_NAMES,
    BRANCH_REMOVALS,
    EVIDENCE,
    HISTORY,
    OTHER_PROMPT,
    REGISTER_BRANCH_PAIRS,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    SELF,
    SHORTCUT_VECTOR_NAMES,
)
from .shortcut import capture_shortcut_geometry
''',
)
replace_once(
    path,
    '''            "final_register_norm": torch.zeros(
                1,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
        }
''',
    '''            "final_register_norm": torch.zeros(
                1,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_route_gram": torch.zeros(
                layers,
                response_tokens,
                len(SHORTCUT_VECTOR_NAMES),
                len(SHORTCUT_VECTOR_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_head_gram": torch.zeros(
                layers,
                response_tokens,
                self.heads,
                len(SHORTCUT_VECTOR_NAMES),
                len(SHORTCUT_VECTOR_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_rewire_valid": torch.zeros(
                layers, response_tokens, dtype=torch.bool
            ),
        }
''',
)
replace_once(
    path,
    '''                    roles = (
                        evidence[None] & ~self_source,
                        (source[None] < response_start)
                        & ~evidence[None]
                        & ~self_source,
                        (source[None] >= response_start)
                        & (source[None] < query[:, None]),
                        self_source,
                    )
                    register_route = self._register_routes(
''',
    '''                    roles = (
                        evidence[None] & ~self_source,
                        (source[None] < response_start)
                        & ~evidence[None]
                        & ~self_source,
                        (source[None] >= response_start)
                        & (source[None] < query[:, None]),
                        self_source,
                    )
                    shortcut = capture_shortcut_geometry(
                        actual_attention,
                        values[index][:, :query_stop],
                        roles,
                        q_to_kv=self.q_to_kv,
                        output_weight=self.layers[index].self_attn.o_proj.weight,
                        output_gram=self.output_grams[index],
                    )
                    for name, value in shortcut.items():
                        target = trace[f"shortcut_{name}"][
                            index, row_start:row_stop
                        ]
                        target.copy_(
                            value.detach().to(dtype=target.dtype, device="cpu")
                        )
                    register_route = self._register_routes(
''',
)

# ---------------------------------------------------------------------------
# collect.py: one schema source and fresh v9 artifacts
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/collect.py"
replace_once(
    path,
    '''from .capture import (
    BRANCH_NAMES,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    FunctionalTraceReplay,
)
''',
    '''from .capture import FunctionalTraceReplay
from .schema import (
    BRANCH_NAMES,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    SHORTCUT_REWIRE,
    SHORTCUT_VECTOR_NAMES,
)
''',
)
replace_once(path, 'VERSION = 8\nSTATE_DIRECTORY = "dual_register_state"\n', 'VERSION = 9\nSTATE_DIRECTORY = "shortcut_route_state"\n')
replace_once(
    path,
    '''        "source_roles": list(ROLE_NAMES),
        "route_cover_mass": float(route_cover_mass),
        "top_k": int(top_k),
''',
    '''        "source_roles": list(ROLE_NAMES),
        "shortcut_vectors": list(SHORTCUT_VECTOR_NAMES),
        "shortcut_rewire": SHORTCUT_REWIRE,
        "route_cover_mass": float(route_cover_mass),
        "top_k": int(top_k),
''',
)

# ---------------------------------------------------------------------------
# graph.py and detect.py: remove duplicated coordinate definitions
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/graph.py"
replace_once(
    path,
    '''import numpy as np

REGISTER_NAMES = ("evidence_adoption", "autonomous_history")
STAGE_NAMES = ("input_state", "attention_write", "mlp_write", "output_state")
ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
''',
    '''import numpy as np

from .schema import REGISTER_NAMES, REGISTER_STAGE_NAMES, ROLE_NAMES

STAGE_NAMES = REGISTER_STAGE_NAMES
''',
)

path = "experiments/attention_mechanism_audit/detect.py"
replace_once(
    path,
    '''import numpy as np

REGISTER_NAMES = ("evidence_adoption", "autonomous_history")
REGISTER_STAGE_NAMES = (
    "input_state",
    "attention_write",
    "mlp_write",
    "output_state",
)
''',
    '''import numpy as np

from .schema import REGISTER_NAMES, REGISTER_STAGE_NAMES
''',
)

# ---------------------------------------------------------------------------
# evaluate.py: isolate shortcut measurement and add fixed post-hoc detection
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/evaluate.py"
replace_once(
    path,
    '''from .capture import (
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
)
''',
    '''from .schema import REGISTER_NAMES, REGISTER_STAGE_NAMES, ROLE_NAMES
''',
)
replace_once(
    path,
    '''from .graph import build_graph
from .visualize import plot_population, plot_sample_dashboard
''',
    '''from .graph import build_graph
from .shortcut import (
    SHORTCUT_SCORE_DEFINITIONS,
    SHORTCUT_SCORE_NAMES,
    shortcut_token_metrics,
)
from .visualize import plot_population, plot_sample_dashboard
''',
)
replace_once(
    path,
    '''    *(f"register_{register}_step_principal_energy" for register in REGISTER_NAMES),
}
''',
    '''    *(f"register_{register}_step_principal_energy" for register in REGISTER_NAMES),
    "shortcut_relay_completion_mean",
    "shortcut_route_completion_mean",
    "shortcut_route_incompleteness_mean",
    "shortcut_endpoint_rewire_gap_mean",
    "shortcut_autonomous_residual_alignment_mean",
    "shortcut_route_candidate_mean",
}
''',
)
replace_once(
    path,
    '''    metrics.update(
        {
            "causal_evidence_support": full - no_evidence,
            "causal_history_support": full - no_history,
            "causal_interaction": contrasts[:, 2],
            "raw_evidence_bypass": no_evidence - full,
            "raw_history_after_cut": no_evidence - no_neither,
            "raw_old_symmetric": no_evidence - no_history,
            "raw_takeover": 2 * no_evidence - full - no_neither,
            "raw_interaction": contrasts[:, 2],
        }
    )
    return {
''',
    '''    metrics.update(
        {
            "causal_evidence_support": full - no_evidence,
            "causal_history_support": full - no_history,
            "causal_interaction": contrasts[:, 2],
            "raw_evidence_bypass": no_evidence - full,
            "raw_history_after_cut": no_evidence - no_neither,
            "raw_old_symmetric": no_evidence - no_history,
            "raw_takeover": 2 * no_evidence - full - no_neither,
            "raw_interaction": contrasts[:, 2],
        }
    )
    metrics.update(shortcut_token_metrics(artifact))
    return {
''',
)
replace_once(
    path,
    '''def _position_match_design(
''',
    '''def shortcut_detection_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate preregistered shortcut candidates with their own validity masks."""

    result: dict[str, dict[str, Any]] = {}
    for offset, name in enumerate(SHORTCUT_SCORE_NAMES):
        validity_name = f"{name}__valid"
        valid = np.asarray(arrays[validity_name], dtype=bool)
        valid &= np.isfinite(arrays[name])
        label = arrays["label"][valid]
        score = arrays[name][valid]
        source = arrays["source_id"][valid]
        current: dict[str, Any] = {
            "valid_tokens": int(valid.sum()),
            "valid_sources": int(np.unique(source).size),
            "auroc": None,
            "average_precision": None,
            "ap_lift": None,
            "auroc_ci95": [None, None],
            "average_precision_ci95": [None, None],
        }
        if len(label) and np.unique(label).size == 2:
            current.update(_binary_metrics(label, score))
            if bootstrap:
                interval = _source_bootstrap(
                    label,
                    score,
                    source,
                    replicates=bootstrap,
                    seed=seed + offset,
                )
                current.update(
                    {
                        "auroc_ci95": [
                            interval["auroc_low"],
                            interval["auroc_high"],
                        ],
                        "average_precision_ci95": [
                            interval["average_precision_low"],
                            interval["average_precision_high"],
                        ],
                        "bootstrap_replicates": interval["replicates"],
                    }
                )
        result[name] = current
    return result


def _position_match_design(
''',
)
replace_once(
    path,
    '''        for marker in ("history", "autonomous", "interaction", "takeover", "symmetric")
''',
    '''        for marker in (
            "history",
            "autonomous",
            "interaction",
            "takeover",
            "symmetric",
            "shortcut",
            "relay",
            "rewire",
            "completion",
        )
''',
)
replace_once(
    path,
    '''        "detection": detection,
        "detector": dict(detector),
''',
    '''        "detection": detection,
        "shortcut_route_detection": shortcut_detection_summary(
            arrays, bootstrap=bootstrap, seed=seed + len(SCORE_ORDER)
        ),
        "shortcut_score_definitions": SHORTCUT_SCORE_DEFINITIONS,
        "detector": dict(detector),
''',
)
replace_once(
    path,
    '''            "label-free finite-difference evidence-adoption versus autonomous-"
            "history registers with raw causal controls and posthoc mechanism "
            "audit by task"
''',
    '''            "label-free finite-difference evidence-adoption versus autonomous-"
            "history registers, exact response-relay Gram geometry, endpoint-"
            "rewiring controls, and posthoc mechanism audit by task"
''',
)

# ---------------------------------------------------------------------------
# run.py: expose the new fixed audit without changing the locked primary score
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/run.py"
replace_once(
    path,
    '''from .evaluate import SCORE_ORDER, evaluate_all, plot_saved_sample
''',
    '''from .evaluate import SCORE_ORDER, evaluate_all, plot_saved_sample
from .shortcut import SHORTCUT_SCORE_NAMES
''',
)
replace_once(
    path,
    '''    "register_autonomous_history_response_history_effective_routes_mean",
)
''',
    '''    "register_autonomous_history_response_history_effective_routes_mean",
    "shortcut_history_write_norm_mean",
    "shortcut_direct_evidence_write_norm_mean",
    "shortcut_evidence_relay_write_norm_mean",
    "shortcut_autonomous_history_write_norm_mean",
    "shortcut_relay_completion_mean",
    "shortcut_route_completion_mean",
    "shortcut_route_incompleteness_mean",
    "shortcut_rewired_route_completion_mean",
    "shortcut_endpoint_rewire_gap_mean",
    "shortcut_autonomous_residual_alignment_mean",
    "shortcut_route_candidate_mean",
    "shortcut_route_rewired_control_mean",
)
''',
)
replace_once(path, 'REPORT_DIRECTORY = "dual_register_v8"\n', 'REPORT_DIRECTORY = "shortcut_route_v9"\n')
replace_once(
    path,
    '''    print("POST-HOC matched hallucinated - correct token differences")
''',
    '''    shortcut = report.get("shortcut_route_detection", {})
    if shortcut:
        print("POST-HOC fixed shortcut-route candidates")
        for name in SHORTCUT_SCORE_NAMES:
            result = shortcut[name]
            if result["auroc"] is None:
                print(f"audit-AUC {name:38s} AUROC=n/a AP=n/a")
                continue
            print(
                f"audit-AUC {name:38s} "
                f"AUROC={result['auroc']:.6f} CI={ci(result['auroc_ci95'])} "
                f"AP={result['average_precision']:.6f} "
                f"CI={ci(result['average_precision_ci95'])}"
            )
    print("POST-HOC matched hallucinated - correct token differences")
''',
)

# ---------------------------------------------------------------------------
# Tests: fixtures, contracts, and route geometry
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/tests/test_capture.py"
replace_once(
    path,
    '''from experiments.attention_mechanism_audit.capture import (
''',
    '''from experiments.attention_mechanism_audit.capture import (
''',
)
# The no-op above intentionally asserts that the expected import anchor exists.
replace_once(
    path,
    '''    assert trace["final_register_norm"].shape == (1, 4, 2)
    assert ROLE_NAMES == (
''',
    '''    assert trace["final_register_norm"].shape == (1, 4, 2)
    assert trace["shortcut_route_gram"].shape == (2, 4, 7, 7)
    assert trace["shortcut_head_gram"].shape == (2, 4, 4, 7, 7)
    assert trace["shortcut_rewire_valid"].shape == (2, 4)
    assert torch.allclose(
        trace["shortcut_route_gram"],
        trace["shortcut_route_gram"].transpose(-1, -2),
    )
    assert torch.allclose(
        trace["shortcut_head_gram"],
        trace["shortcut_head_gram"].transpose(-1, -2),
    )
    assert ROLE_NAMES == (
''',
)

path = "experiments/attention_mechanism_audit/tests/test_collect.py"
replace_once(
    path,
    '''        "final_register_norm": torch.zeros(1, response_tokens, registers),
    }
''',
    '''        "final_register_norm": torch.zeros(1, response_tokens, registers),
        "shortcut_route_gram": torch.zeros(layers, response_tokens, 7, 7),
        "shortcut_head_gram": torch.zeros(layers, response_tokens, heads, 7, 7),
        "shortcut_rewire_valid": torch.zeros(
            layers, response_tokens, dtype=torch.bool
        ),
    }
''',
)
replace_once(path, 'assert first["version"] == collect_module.VERSION == 8\n', 'assert first["version"] == collect_module.VERSION == 9\n')
replace_once(
    path,
    '''        "source_roles": [
            "evidence",
            "other_prompt",
            "response_history",
            "predictor_self",
        ],
        "route_cover_mass": 0.8,
''',
    '''        "source_roles": [
            "evidence",
            "other_prompt",
            "response_history",
            "predictor_self",
        ],
        "shortcut_vectors": [
            "full_history_write",
            "direct_evidence_write",
            "evidence_relay_carrier",
            "evidence_relay_gate",
            "autonomous_history_write",
            "rewired_evidence_relay_carrier",
            "rewired_evidence_relay_gate",
        ],
        "shortcut_rewire": "adjacent_response_endpoint_swap",
        "route_cover_mass": 0.8,
''',
)
replace_once(path, 'assert first_rows[0]["artifact_contract"]["version"] == 8\n', 'assert first_rows[0]["artifact_contract"]["version"] == 9\n')
replace_once(
    path,
    'assert collect_module.STATE_DIRECTORY == "dual_register_state"\n',
    'assert collect_module.STATE_DIRECTORY == "shortcut_route_state"\n',
)

path = "experiments/attention_mechanism_audit/tests/test_evaluate.py"
replace_once(
    path,
    '''    trace = {
''',
    '''    shortcut_vectors = torch.zeros(layers, tokens, 7, 3)
    shortcut_vectors[..., 0, 0] = 1.0
    shortcut_vectors[..., 2, 0] = 1.0
    shortcut_vectors[..., 4, 1] = 1.0
    shortcut_vectors[..., 5, 1] = 1.0
    shortcut_route_gram = torch.einsum(
        "ltkd,ltmd->ltkm", shortcut_vectors, shortcut_vectors
    )
    trace = {
''',
)
replace_once(
    path,
    '''        "final_register_norm": torch.tensor([5.0, 7.0])
        .expand(1, tokens, registers)
        .clone(),
    }
''',
    '''        "final_register_norm": torch.tensor([5.0, 7.0])
        .expand(1, tokens, registers)
        .clone(),
        "shortcut_route_gram": shortcut_route_gram,
        "shortcut_head_gram": shortcut_route_gram[:, :, None].expand(
            layers, tokens, heads, 7, 7
        ).clone(),
        "shortcut_rewire_valid": torch.ones(layers, tokens, dtype=torch.bool),
    }
''',
)
replace_once(
    path,
    '''    assert "prompt_edge_log_volume_layer_shift" in audit
    assert not any("pathway_" in name for name in audit)
''',
    '''    assert "prompt_edge_log_volume_layer_shift" in audit
    np.testing.assert_allclose(audit["shortcut_relay_completion_mean"], 1.0)
    np.testing.assert_allclose(audit["shortcut_route_incompleteness_mean"], 0.0)
    assert audit["shortcut_route_candidate_mean__valid"].all()
    assert not any("pathway_" in name for name in audit)
''',
)

# ---------------------------------------------------------------------------
# Documentation and formal claim boundaries
# ---------------------------------------------------------------------------
path = "experiments/attention_mechanism_audit/AGENTS.md"
replace_once(
    path,
    '''- Maintain one implementation of the **dual-register attention mechanism
  audit**. Do not restore the retired incidence-graph framing or make
  `unsupported_history_takeover` the fixed primary detector.
''',
    '''- Maintain one implementation of the **dual-register attention mechanism
  audit with shortcut-route validation**. Do not restore the retired incidence-
  graph framing or make `unsupported_history_takeover` or a new shortcut
  statistic the fixed primary detector before full-data evaluation.
''',
)
replace_once(
    path,
    '''- Keep the established full-prompt collapse measurements only as a historical
  QA audit. They are not the current detector and must not be generalized from
  the earlier QA result to Summary or Data2txt.
''',
    '''- Capture, for every `(layer,target)`, the residual-space Gram of the full
  strict-history write, direct-evidence write, evidence-conditioned history
  carrier and gate writes, autonomous-history write, and an adjacent-endpoint
  rewiring control. Preserve the matching per-head Gram. These are the raw
  audit objects; scalar completion scores are derived later and never replace
  them.
- The route-completion hypothesis is specific: a supported response relay has
  a full history write explained by direct evidence plus evidence-conditioned
  carrier/gate writes. A shortcut is the residual history write that remains
  aligned with autonomous history after this evidence-support subspace is
  removed. Compare observed endpoints with the fixed adjacent-endpoint rewire.
- Keep the established full-prompt collapse measurements only as a historical
  QA audit. They are not the current detector and must not be generalized from
  the earlier QA result to Summary or Data2txt.
''',
)
replace_once(
    path,
    '''- Schema 8 requires a fresh capture under `dual_register_state/train/` and
  `dual_register_state/test/`. Do not adapt old artifacts in place and do not
  delete them; historical output directories remain preserved. Write v8 task
  reports under `dual_register_v8/{qa,summary,data2txt}/` instead of replacing
  earlier reports.
''',
    '''- Schema 9 requires a fresh capture under `shortcut_route_state/train/` and
  `shortcut_route_state/test/`. Do not adapt old artifacts in place and do not
  delete them; historical output directories remain preserved. Write v9 task
  reports under `shortcut_route_v9/{qa,summary,data2txt}/` instead of replacing
  earlier reports.
''',
)
replace_once(
    path,
    '''- Test predictor alignment, branch removals, role partitioning, layer closure,
  register Gram construction, global sparse selection and exact tails, raw
  score equations, validity masks, and label sealing. Synthetic correctness is
  not empirical validation.
''',
    '''- Test predictor alignment, branch removals, role partitioning, layer closure,
  register Gram construction, shortcut-route Gram geometry, endpoint rewiring,
  global sparse selection and exact tails, raw score equations, validity masks,
  and label sealing. Synthetic correctness is not empirical validation.
''',
)

path = "experiments/attention_mechanism_audit/README.md"
text = Path(path).read_text(encoding="utf-8")
anchor = "## Pipeline\n"
if text.count(anchor) != 1:
    raise RuntimeError("README pipeline anchor changed")
insert = '''## Shortcut-route hypothesis\n\nA current token may legitimately read a previous response token. The audit\ntherefore does not call response attention a shortcut by itself. It tests\nwhether the strict-history write is supported by the evidence-conditioned\nstate of those response carriers. For every layer and prediction event it\nstores the residual-space Gram of:\n\n```text\nfull history write\ndirect evidence write\nevidence relay: mean(A) delta(V)\nevidence-conditioned gate: delta(A) mean(V)\nautonomous history write after the evidence cut\nadjacent-endpoint rewired relay and gate controls\n```\n\nThe observed route is complete when the full history write lies in the span of\ndirect evidence and evidence-conditioned relay/gate writes. The shortcut\ncandidate is the remaining history component when it is aligned with the\nautonomous-history write. The adjacent swap preserves target rows, heads,\ncoefficient values, and the response-value multiset while breaking the exact\ncarrier endpoint. All scalar measurements are post-capture views of the saved\nGram; they do not replace the raw geometry.\n\n'''
Path(path).write_text(text.replace(anchor, insert + anchor), encoding="utf-8")

path = "experiments/attention_mechanism_audit/METHOD.md"
text = Path(path).read_text(encoding="utf-8")
section = '''\n\n## Shortcut-route completeness audit\n\nFor prediction position `q`, let `H_q` denote strict response-history sources.\nThe full response-history write is\n\n\\[\nh_q^l = W_O^l\\operatorname{concat}_a\\sum_{j\\in H_q}\nA_{F,qj}^{l,a}V_{F,j}^{l,\\kappa(a)}.\n\\]\n\nDeleting direct evidence gives the exact midpoint decomposition over response\ncarriers\n\n\\[\ne_{\\mathrm{carrier},q}^l = W_O^l\\operatorname{concat}_a\n\\sum_{j\\in H_q}\\frac{A_F+A_{noE}}{2}\n(V_F-V_{noE}),\n\\]\n\n\\[\ne_{\\mathrm{gate},q}^l = W_O^l\\operatorname{concat}_a\n\\sum_{j\\in H_q}(A_F-A_{noE})\n\\frac{V_F+V_{noE}}{2}.\n\\]\n\nThe direct evidence write and the exact history-root write for `noE - noEH`\ncomplete the observed vector set. Capture stores their aggregate and per-head\nGram matrices. It also swaps adjacent response value endpoints before the two\nrelay calculations. This control keeps the coefficient and value multisets but\nbreaks the observed endpoint pairing with at most one-token displacement inside\neach pair.\n\nLet `S=[direct evidence, carrier, gate]`. Route completion is the fraction of\nthe full-history energy projected onto `span(S)`. The shortcut candidate is the\nunexplained energy fraction multiplied by the positive cosine between the\nresidualized full-history and autonomous-history writes. These directions are\nfrozen before labels are opened. They remain mechanism-audit candidates; the\nlocked primary detector is unchanged until full QA, Summary, and Data2txt\nevaluation supports replacement.\n'''
if "## Shortcut-route completeness audit" in text:
    raise RuntimeError("METHOD shortcut section already exists")
Path(path).write_text(text.rstrip() + section + "\n", encoding="utf-8")
