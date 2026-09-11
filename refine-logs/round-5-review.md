# Round 5 Final Reporting-Gap Check

## Scope

This final review only checks the Round 4 reporting gap in:

- `route_graph/evaluation.py`
- `tests/test_route_evaluation.py`

No implementation files were modified.

## Result

The reporting gap is closed.

`route_graph/evaluation.py` now:

- requires `reference_active_features` to be a nonnegative integer (`evaluation.py:52-58`);
- builds an `active` mask from `reference_active_features > 0` (`evaluation.py:101-103`);
- reports `inactive_reference_tokens` for every subset (`evaluation.py:130-135`);
- reports global `reference_coverage` with active token count, inactive token count, active fraction, sources with inactive tokens, and the explicit warning that zero score means no reference variation, not trusted normal (`evaluation.py:164-170`).

`tests/test_route_evaluation.py` now constructs 2 inactive tokens out of 6 via `reference_active_features: 0 if index == 0 else 2` (`test_route_evaluation.py:24`) and asserts:

- `inactive_tokens == 2` (`test_route_evaluation.py:61`);
- `active_token_fraction == 4 / 6` (`test_route_evaluation.py:62`).

## Final Conformance

Implementation conformance: **CONFORMANT for the finalized research specification**.

Scientific status remains unchanged because no new natural source-disjoint performance evidence was reviewed:

- Score: **8.5 / 10**
- Verdict: **REVISE**
- Paper readiness: **not ready until natural residual gains are demonstrated**
- Implementation readiness: **ready for natural evaluation**

This is the final method-review round under the configured loop limit.
