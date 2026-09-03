from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


path = "experiments/attention_mechanism_audit/shortcut.py"
replace_once(
    path,
    '''    "shortcut_autonomous_residual_alignment_mean",
''',
    '''    "shortcut_autonomous_support_mean",
''',
)
replace_once(
    path,
    '''    "shortcut_autonomous_residual_alignment_mean": (
        "alignment of full-history and autonomous-history writes after removing "
        "their shared evidence-support subspace"
    ),
''',
    '''    "shortcut_autonomous_support_mean": (
        "signed contribution of the no-evidence history write to the full "
        "history-write direction"
    ),
''',
)
replace_once(
    path,
    '''def _residual_cosine(
    gram: np.ndarray,
    left: int,
    right: int,
    support: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    inverse = _support_inverse(gram, support)
    left_support = np.take(gram[..., left, :], support, axis=-1)
    right_support = np.take(gram[..., right, :], support, axis=-1)
    left_energy = gram[..., left, left] - np.einsum(
        "...i,...ij,...j->...", left_support, inverse, left_support
    )
    right_energy = gram[..., right, right] - np.einsum(
        "...i,...ij,...j->...", right_support, inverse, right_support
    )
    cross = gram[..., left, right] - np.einsum(
        "...i,...ij,...j->...", left_support, inverse, right_support
    )
    left_energy = np.maximum(left_energy, 0.0)
    right_energy = np.maximum(right_energy, 0.0)
    denominator = np.sqrt(left_energy * right_energy)
    valid = denominator > _EPS
    cosine = np.zeros_like(denominator)
    cosine[valid] = np.clip(cross[valid] / denominator[valid], -1.0, 1.0)
    return cosine, valid


''',
    '''def _signed_support(
    gram: np.ndarray, source: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project additive source writes onto the full-history direction.

    For an exact decomposition ``h = r + a``, the returned signed supports
    ``<h,r>/||h||²`` and ``<h,a>/||h||²`` sum to one.  Unlike residualizing
    both ``h`` and ``a`` against ``r``, this quantity is not an algebraic
    identity equal to one; it retains reinforcement and cancellation.
    """

    energy = np.maximum(gram[..., FULL_HISTORY_WRITE, FULL_HISTORY_WRITE], 0.0)
    cross = np.take(gram[..., FULL_HISTORY_WRITE, :], source, axis=-1).sum(-1)
    valid = energy > _EPS
    support = np.zeros_like(energy)
    support[valid] = cross[valid] / energy[valid]
    return support, valid


''',
)
replace_once(
    path,
    '''    autonomous_alignment, autonomous_valid = _residual_cosine(
        gram,
        FULL_HISTORY_WRITE,
        AUTONOMOUS_HISTORY_WRITE,
        evidence_support,
    )
    rewired_alignment, rewired_autonomous_valid = _residual_cosine(
        gram,
        FULL_HISTORY_WRITE,
        AUTONOMOUS_HISTORY_WRITE,
        rewired_support,
    )

    route_incompleteness = 1.0 - route_completion
    relay_incompleteness = 1.0 - relay_completion
    endpoint_rewire_gap = rewired_completion - route_completion
    shortcut_candidate = route_incompleteness * np.maximum(autonomous_alignment, 0.0)
    rewired_candidate = (1.0 - rewired_completion) * np.maximum(
        rewired_alignment, 0.0
    )
''',
    '''    evidence_relay_support, support_valid = _signed_support(
        gram, (EVIDENCE_RELAY_CARRIER, EVIDENCE_RELAY_GATE)
    )
    autonomous_support, autonomous_valid = _signed_support(
        gram, (AUTONOMOUS_HISTORY_WRITE,)
    )
    additive_support_error = np.abs(
        evidence_relay_support + autonomous_support - 1.0
    )

    route_incompleteness = 1.0 - route_completion
    endpoint_rewire_gap = rewired_completion - route_completion
    shortcut_candidate = route_incompleteness * np.maximum(autonomous_support, 0.0)
    rewired_candidate = (1.0 - rewired_completion) * np.maximum(
        autonomous_support, 0.0
    )
''',
)
replace_once(
    path,
    '''        "shortcut_endpoint_rewire_gap": endpoint_rewire_gap,
        "shortcut_autonomous_residual_alignment": autonomous_alignment,
        "shortcut_route_candidate": shortcut_candidate,
''',
    '''        "shortcut_endpoint_rewire_gap": endpoint_rewire_gap,
        "shortcut_evidence_relay_support": evidence_relay_support,
        "shortcut_autonomous_support": autonomous_support,
        "shortcut_additive_support_error": additive_support_error,
        "shortcut_route_candidate": shortcut_candidate,
''',
)
replace_once(
    path,
    '''        "shortcut_endpoint_rewire_gap": valid_rewire,
        "shortcut_autonomous_residual_alignment": history_valid & autonomous_valid,
        "shortcut_route_candidate": history_valid & autonomous_valid,
        "shortcut_route_rewired_control": (
            valid_rewire & rewired_autonomous_valid
        ),
''',
    '''        "shortcut_endpoint_rewire_gap": valid_rewire,
        "shortcut_evidence_relay_support": history_valid & support_valid,
        "shortcut_autonomous_support": history_valid & autonomous_valid,
        "shortcut_additive_support_error": history_valid & support_valid & autonomous_valid,
        "shortcut_route_candidate": history_valid & autonomous_valid,
        "shortcut_route_rewired_control": valid_rewire & autonomous_valid,
''',
)

path = "experiments/attention_mechanism_audit/evaluate.py"
replace_once(
    path,
    '''    "shortcut_autonomous_residual_alignment_mean",
''',
    '''    "shortcut_autonomous_support_mean",
''',
)

path = "experiments/attention_mechanism_audit/run.py"
replace_once(
    path,
    '''    "shortcut_autonomous_residual_alignment_mean",
''',
    '''    "shortcut_evidence_relay_support_mean",
    "shortcut_autonomous_support_mean",
    "shortcut_additive_support_error_mean",
''',
)
replace_once(path, 'REPORT_DIRECTORY = "shortcut_route_v9"\n', 'REPORT_DIRECTORY = "shortcut_route_v10"\n')

path = "experiments/attention_mechanism_audit/collect.py"
replace_once(path, 'VERSION = 9\nSTATE_DIRECTORY = "shortcut_route_state"\n', 'VERSION = 10\nSTATE_DIRECTORY = "shortcut_route_state_v10"\n')

path = "experiments/attention_mechanism_audit/AGENTS.md"
replace_once(
    path,
    '''- Schema 9 requires a fresh capture under `shortcut_route_state/train/` and
  `shortcut_route_state/test/`. Do not adapt old artifacts in place and do not
  delete them; historical output directories remain preserved. Write v9 task
  reports under `shortcut_route_v9/{qa,summary,data2txt}/` instead of replacing
  earlier reports.
''',
    '''- Schema 10 requires a fresh capture under `shortcut_route_state_v10/train/`
  and `shortcut_route_state_v10/test/`. Do not adapt old artifacts in place and
  do not delete them; historical output directories remain preserved. Write v10
  task reports under `shortcut_route_v10/{qa,summary,data2txt}/` instead of
  replacing earlier reports.
''',
)

path = "experiments/attention_mechanism_audit/README.md"
replace_once(
    path,
    '''The observed route is complete when the full history write lies in the span of
direct evidence and evidence-conditioned relay/gate writes. The shortcut
candidate is the remaining history component when it is aligned with the
autonomous-history write. The adjacent swap preserves target rows, heads,
''',
    '''The observed route is complete when the full history write lies in the span of
direct evidence and evidence-conditioned relay/gate writes. The shortcut
candidate multiplies unexplained history energy by the positive signed
contribution of the no-evidence history write to the full-history direction.
This avoids the degenerate operation of residualizing two vectors whose
residuals are algebraically identical. The adjacent swap preserves target rows, heads,
''',
)
replace_once(
    path,
    '''Schema 9 must be recaptured into
`outputs/<observer-model>/shortcut_route_state/{train,test}/`. Older capture
directories are preserved as historical artifacts and are not adapted or
deleted. New reports are written under
`outputs/<observer-model>/shortcut_route_v9/{qa,summary,data2txt}/`, so the
earlier task reports are not overwritten.
''',
    '''Schema 10 must be recaptured into
`outputs/<observer-model>/shortcut_route_state_v10/{train,test}/`. Older capture
directories are preserved as historical artifacts and are not adapted or
deleted. New reports are written under
`outputs/<observer-model>/shortcut_route_v10/{qa,summary,data2txt}/`, so the
earlier task reports are not overwritten.
''',
)

path = "experiments/attention_mechanism_audit/METHOD.md"
replace_once(
    path,
    '''Let `S=[direct evidence, carrier, gate]`. Route completion is the fraction of
the full-history energy projected onto `span(S)`. The shortcut candidate is the
unexplained energy fraction multiplied by the positive cosine between the
residualized full-history and autonomous-history writes. These directions are
frozen before labels are opened. They remain mechanism-audit candidates; the
locked primary detector is unchanged until full QA, Summary, and Data2txt
evaluation supports replacement.
''',
    '''Let `S=[direct evidence, carrier, gate]`. Route completion is the fraction of
the full-history energy projected onto `span(S)`. Since
`full_history = evidence_relay + autonomous_history`, residualizing the first
and third terms against the relay would make their residuals identical and the
cosine trivially one. The audit therefore uses the non-degenerate signed
support

\\[
c_{auto}=\\frac{\\langle h_{full},h_{auto}\\rangle}
{\\lVert h_{full}\\rVert^2},
\\]

and verifies that the corresponding relay support sums with it to one. The
shortcut candidate is route incompleteness times `max(c_auto, 0)`. These
directions are frozen before labels are opened. They remain mechanism-audit
candidates; the locked primary detector is unchanged until full QA, Summary,
and Data2txt evaluation supports replacement.
''',
)

path = "experiments/attention_mechanism_audit/tests/test_collect.py"
replace_once(path, 'assert first["version"] == collect_module.VERSION == 9\n', 'assert first["version"] == collect_module.VERSION == 10\n')
replace_once(path, 'assert first_rows[0]["artifact_contract"]["version"] == 9\n', 'assert first_rows[0]["artifact_contract"]["version"] == 10\n')
replace_once(
    path,
    'assert collect_module.STATE_DIRECTORY == "shortcut_route_state"\n',
    'assert collect_module.STATE_DIRECTORY == "shortcut_route_state_v10"\n',
)

path = "experiments/attention_mechanism_audit/tests/test_evaluate.py"
replace_once(
    path,
    '''    # This fixture's history is completely explained by the relay span, so
    # no residual exists on which an autonomous alignment could be defined.
    assert not audit["shortcut_route_candidate_mean__valid"].any()
''',
    '''    # This fixture's history is completely explained by the relay span.
    # The signed autonomous support is still defined, but the shortcut product
    # is exactly zero because route incompleteness is zero.
    assert audit["shortcut_route_candidate_mean__valid"].all()
    np.testing.assert_allclose(audit["shortcut_route_candidate_mean"], 0.0)
''',
)

path = "experiments/attention_mechanism_audit/tests/test_shortcut.py"
replace_once(
    path,
    'def test_shortcut_candidate_is_residual_history_aligned_with_autonomy():\n',
    'def test_shortcut_candidate_uses_non_degenerate_autonomous_support():\n',
)
replace_once(
    path,
    '''    np.testing.assert_allclose(
        layer["shortcut_autonomous_residual_alignment"], 1.0, atol=1e-6
    )
''',
    '''    np.testing.assert_allclose(
        layer["shortcut_autonomous_support"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        layer["shortcut_evidence_relay_support"], 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        layer["shortcut_additive_support_error"], 0.0, atol=1e-6
    )
''',
)

# Add a regression test proving that the new statistic is not identically one.
test_path = Path("experiments/attention_mechanism_audit/tests/test_shortcut.py")
text = test_path.read_text(encoding="utf-8")
addition = '''\n\ndef test_autonomous_support_is_not_the_old_tautological_residual_cosine():
    layers, tokens, vectors, hidden = 1, 2, 7, 2
    state = torch.zeros(layers, tokens, vectors, hidden)
    state[..., 0, 0] = 1.0  # full history
    state[..., 2, 0] = 0.8  # evidence-conditioned carrier supports most of it
    state[..., 4, 0] = 0.2  # autonomous history supports the remainder
    gram = torch.einsum("ltkd,ltmd->ltkm", state, state)
    metrics = shortcut_layer_metrics(
        {
            "shortcut_route_gram": gram,
            "shortcut_rewire_valid": torch.ones(layers, tokens, dtype=torch.bool),
        }
    )

    np.testing.assert_allclose(
        metrics["shortcut_evidence_relay_support"], 0.8, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["shortcut_autonomous_support"], 0.2, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["shortcut_additive_support_error"], 0.0, atol=1e-6
    )
'''
if "test_autonomous_support_is_not_the_old_tautological_residual_cosine" in text:
    raise RuntimeError("autonomous-support regression already exists")
test_path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
