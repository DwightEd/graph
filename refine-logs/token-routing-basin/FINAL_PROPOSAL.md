# Final proposal: causal multiplex token-state baseline

## Decision

Retire the artificial age/top-k/PCA residual from the active path. Implement a
prefix-causal, token-level multiplex routing baseline, but do not claim that it
detects hallucination until held-out results exceed prevalence and controls.

## Observable operator

For every retained off-diagonal edge, use only
`max(attention - attention_floor, 0)`. At token `t`, build causal rows whose
columns preserve `(layer, head, exact source)`, plus a relative operator whose
columns preserve `(layer, head, role, prompt-position/RR-lag)`. A deterministic
signed CountSketch bounds memory while approximating their inner products.

Rolling Gram spectra over the last `W` valid rows yield effective-rank fraction
and dominant-mode share for combined, prompt, response, and relative routing.
Invalid rows reset the window. This avoids both dense `[L,H,R,N]` materialization
and the degenerate eigenvalues of a triangular causal adjacency matrix.

## Baseline score

Source groups are split into fit, component-calibration, and final-calibration
sets. Nuisance-adjusted state novelty, valid one-step transition surprise,
preregistered routing commitment, and its causal EMA are empirically calibrated.
The EMA is named `smoothed_commitment`; it is not itself attractor evidence.

The implementation reads no labels during fit/score and freezes a complete
source-audited token artifact before evaluation. Evaluation separately reports
same-token post-emission detection and horizons 1, 2, and 4 forecasting, plus
coverage and onset eligibility.

## Falsification outcome

The local 64-sample development run is operational but negative: AUROC 0.473
and AUPRC 0.055 at prevalence 0.062. Horizons 1/2/4 are also below chance.
`prompt_top1_share` alone reaches AUROC 0.620/AUPRC 0.096, so a localized
prompt-concentration signal remains plausible even though the fixed unlabeled
composite does not. Therefore this implementation is an interpretable
mechanism baseline, not the final learned detector.

The next justified model is a source-disjoint supervised causal GRU/TCN over
these frozen multiplex states, with controls-only, old PCA, coarse route, and
multiplex ablations. It should only become the active detector if it improves
fresh held-out token AUPRC, onset delay, and false alerts per 1,000 normal
tokens.

## Required scientific checks

- current-token and horizon 1/2/4 metrics reported separately;
- sample/source-cluster uncertainty and task/source breakdown;
- prompt/source/channel/lag shuffle tests;
- window and CountSketch-width sensitivity;
- controls-only, old PCA, concentration-only, and multiplex comparisons;
- fresh source/task/generator/observer holdout before a publishable claim.
