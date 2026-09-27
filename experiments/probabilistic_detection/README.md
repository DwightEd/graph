# Source-conditioned probabilistic hallucination detection

This is an explicitly supervised experiment on existing native RAGTruth traces.
It does not change the language model, the original answers, the official labels,
or the older unsupervised detectors. It uses the observer model's cached replay,
not the generating model's original internal trajectory.

## Statistical question

Given source support and position/length context C, does the joint response X
contain hallucination information beyond that context?

    logit P(Y=1 | C,X) = logit P(Y=1 | C) + log p(X | C,Y=1) / p(X | C,Y=0)

This is Bayes' identity, not a new theorem. It motivates two explicit estimators:

* `model.py`: shared train-fitted normal-score X and spline C transforms,
  class-specific ridge conditional means, regularized residual covariance and
  Gaussian log-density ratios. Full, diagonal, shared-covariance and
  shared-correlation/class-specific-variance controls separate assumptions.
* `readout.py`: L2 logistic estimates log odds directly. Nested designs add
  per-channel squares, cross-channel products, then source-anchor × response
  products. The linear, squares, interactions and conditioned models have the
  same train-only transforms and source weights.

The Gaussian score includes a development-selected likelihood-ratio temperature;
its fixed primary is shrinkage .5, temperature .25. Readout C is chosen from
{.01, 1}. Gaussian, linear, quadratic and tree models are established estimators.
Empirical normal-score transforms have ties and clipped tails, so density ratios
are defined in the transformed space, not asserted to equal exact native-space
ratios. Equal-source weighting changes the training population; output scores
are not automatically calibrated natural-token probabilities.

## Inputs and implementation

`data.py` defines 6 context and 11 observation features from source-first cache:

* Context: local/full source gain unit means (risk sign = absent minus present
  log probability), relative and log absolute position, log answer/prompt lengths.
* Observations: token deviations from the two source means, present-source
  full/local original-token log probabilities, route, attention, entropy,
  fixed 16-token offline route/entropy means, and unit-centered route/entropy.

Units come from the saved original response, not hallucination spans. Windows
stay inside an answer. Token IDs, valid offset masks and contiguous unit coverage
are checked. No generator/source/answer IDs or labels enter features.

The route signal is a head-message-magnitude-weighted history/source contrast.
Attention is an unweighted history/source attention-mass contrast. Their
predictive interaction would not by itself prove factual information transfer:
value magnitudes ignore message direction/cancellation, and attention is not
semantic evidence. Scalar cache `save_heads=false` cannot establish head-level
coordination or FFN mechanisms.

`iteration.py` compares the nested readouts plus fixed linear feature ablations:
remove absolute original-token logp, route-family features, or explicit
position/length; context-only is also reported. Feature ablations apply to the
linear readout, not automatically to the winning nonlinear readout.

## Data separation

Official train sources are deterministically split using the original hash42
20% development sources. All six generator answers for a source stay together.
All scalers, quantiles, spline transforms and parameters fit only on remaining
training sources. Development sources select candidates and set each method's
normal-token 95th percentile threshold (strict `>` alarm).

`prepare-test` creates features without labels. `score` writes all predictions;
`evaluate` requires all requested tasks frozen before obtaining their official
labels. The inherited annotation reader parses the dataset JSONL before filtering
IDs; test labels are not returned to training/selection. We do not claim test
records were never physically parsed. Historical official test results informed
previous project design, so new test results are exploratory revalidation.

`freeze` additionally picks a usable detector from the development winners of
raw logistic, matched logistic, HGB, Gaussian and quadratic readout. This is an
engineering selector, not an additional new model. All alternatives remain
visible in the reports.

## Run

From the repository root, with the existing scientific Python environment:

```bash
python main.py probabilistic --phase prepare-train --output outputs/my_run
python main.py probabilistic --phase develop --output outputs/my_run
python main.py probabilistic --phase iterate --output outputs/my_run
python main.py probabilistic --phase freeze --output outputs/my_run
python main.py probabilistic --phase prepare-test --output outputs/my_run
python main.py probabilistic --phase score --output outputs/my_run
python main.py probabilistic --phase evaluate --output outputs/my_run
```

For a training/development pilot, pass `--train-source-limit 96
--dev-source-limit 32` to `prepare-train`. Source limits are hash ordered and
label independent. Other phases reuse its saved packs. Use a new output directory
to change a frozen experiment. Set OMP/OPENBLAS/MKL_NUM_THREADS explicitly for
repeatable CPU resource use. No new LLM forwards are needed for these caches.

## Evaluation and evidence

`evaluation.py` reports pooled token AUROC/AP, positive-negative-pair-weighted
within-answer/unit AUROC, hallucination onset/first-error metrics, token recall
and FPR, normal-answer any-alarm rate, and per-generator scores. Source-cluster
bootstrap intervals are exploratory, unadjusted for multiple comparisons.
Thresholds remain those fixed on development data. Token-level FPR is different
from a normal answer receiving at least one false alarm.

Known difficult examples are an exposed regression panel, not representative
independent evidence. The case reporter marks fit examples as in-sample and
compares error tokens with neighboring normal tokens. Offline measurements are
not advance warning or online onset prediction.

The shared research handoff is at
`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/probabilistic_detection_20260928/`.
It records the plan, development-driven addendum, failed hypotheses, source
counts, exact result paths and review limitations. Raw arrays stay under
`outputs/probabilistic_detection_20260928_{pilot,full}`.
