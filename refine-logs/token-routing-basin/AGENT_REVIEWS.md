# Independent design and implementation reviews

Three read-only agent routes reviewed the method: a minimal label-free design,
a spectral/operator design, and an implementation/scientific audit.

## Shared conclusions

- Detection must be one row per response token and prefix invariant.
- Same-position attention is post-emission; future-token forecasting must be
  evaluated separately.
- Causal adjacency eigenvalues are degenerate. Use a rectangular
  query-by-route operator and its Gram/SVD spectrum.
- Preserve layer, head, exact source, and relative lag. A layer-band/position
  histogram is only a coarse baseline.
- Do not use final response length as a scoring feature.
- Missing/floor-censored rows need explicit validity and must reset temporal
  state rather than silently becoming zero routes.
- Fit/component/final calibration source groups must be disjoint, and the
  score artifact must carry a verifiable held-out source audit.

## Review-driven corrections

The first implementation still discarded heads in its spectral route
histogram. A failing wiring test was added, then the representation was changed
to multiplex `(layer, head, source)` and `(layer, head, role, lag/position)`
operators. Signed CountSketch replaced slow sparse-sparse Gram multiplication,
reducing the 8-sample fit/score extraction time by roughly 4--5x.

The audit also caused these fixes:

- transition calibration excludes tokens without a valid predecessor;
- invalid rows reset rolling spectra and commitment smoothing;
- `residence` was renamed `smoothed_commitment`;
- calibration uses separate component and final source groups;
- thresholding uses a conformal `alpha=0.05` tail rather than an empirical
  quantile with inflated ties;
- evaluation reports label-conditional coverage, conservative missingness
  sensitivity, forecasting horizons, onset eligibility, delay, and false
  alerts per 1,000 normal tokens;
- scripts resolve project/data paths, validate manifests, run tests, and check
  every stage artifact.

## Final audit judgment

The module boundary and causal invariants are sound for an interpretable
baseline. The pilot efficacy is not sound enough for a detector claim. Agents
recommended a supervised causal sequence learner only after freezing these
states and evaluating source-disjoint controls and ablations.
