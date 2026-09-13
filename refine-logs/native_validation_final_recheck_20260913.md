# Native validation final limited recheck

Date: 2026-09-13  
Scope limited to the two fixes requested after `native_validation_round2_engineering_20260913.md`.

Required status: **0 open**.

Checked fixes:

1. N merge now counts a native internal certificate only when the selected position also satisfies `template.passed`, `template.origin_passed`, and `origins[id].passed`. It preserves position-only evidence in `raw_position_certificate_ids` and labels N certificate scope as `two_template_position_and_origin`.
2. The frozen validation protocol now uses branch-specific order: `C_or_supported_recovery` includes paired-role and MLP interaction; `N` uses `position`, `origin`, `N_template`, `layer_bands`.

Added regression test:

- `tests/test_audit_pipeline_review.py::test_null_position_only_template_does_not_count_internal_certificate`

Targeted validation:

```text
1 passed in 16.71s
```

No full test suite was run in this limited recheck. No GPU was used.

