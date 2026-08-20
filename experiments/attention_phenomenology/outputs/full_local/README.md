# Local full-audit result provenance

These compact artifacts were generated on 2026-08-20 from commit `6a72152`
with the complete local RAGTruth split:

- 2,497 unlabeled train samples;
- 449 test samples;
- 73,994 evaluated response tokens;
- 4,594 positive tokens (prevalence 0.0621).

The checked-in subset contains the fitted reference, score manifest, and final
evaluation tables. The 449 per-sample score files are intentionally excluded
because they occupy about 738 MiB.

## Known limitations

- `rewire_role_max_abs_error` is 0.9961, so the endpoint-rewired null did not
  preserve the advertised coarse role simplex. Real-versus-rewired topology
  comparisons from this run are invalid.
- Some robust reference scales collapsed below `1e-6`, producing extreme
  `lockin` and `all` scores. Their mean onset curves must not be interpreted as
  mechanism strength.
- Commit `c4622d4` subsequently refactored the audit and changed its artifact
  schema. These files document the earlier run and are not presented as output
  reproduced by the refactored code.

The family AUROC/AUPRC and raw layer-level tables remain useful as exploratory
development evidence, subject to the limitations above. A confirmatory result
requires regenerating artifacts after the null-model and scale invariants pass.
