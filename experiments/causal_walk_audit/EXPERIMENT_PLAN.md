# Experiment Plan

## Goal

Test whether a channel-resolved route grammar provides a reproducible
label-free hallucination signal, and determine which parts of the graph are
actually necessary.

## Frozen hypotheses

### H1: route grammar anomaly

Hallucinated tokens have larger calibrated grammar rupture than correct tokens.

Primary output:

```text
score
```

### H2: order-two increment

The variable-order backoff grammar improves unlabeled held-out prediction over
order one:

\[
E[H_1-H_{\mathrm{backoff}}] > 0.
\]

This is checked without hallucination labels through `order2_gain_mean` and the
mean interpolation weight. If the gain is absent, set `backoff_tau` high enough
to reduce the method to order one and remove the De Bruijn claim.

### H3: exact topology increment

Endpoint rewiring increases route-grammar rupture under sufficient null
coverage. If the topology gate fails, do not claim an exact-token graph
contribution.

### H4: response closure

`rupture_closure_mean` exceeds rupture alone and rises around hallucination
onset. If it does not, retain closure only as a descriptive diagnostic.

## Mandatory comparisons

1. Primary calibrated `score`.
2. `order1_surprisal_mean`.
3. Backoff `grammar_surprisal_mean`.
4. `rupture_mean`.
5. `closure_mean`.
6. `rupture_closure_mean`.
7. Exact graph versus endpoint-rewired topology gate.
8. QA, Summary, and Data2txt reported separately.
9. At least five seeds for the source-group split and calibration reservoirs.
10. Whole-response bootstrap confidence intervals.

## Stopping rules

- If grammar surprise does not exceed prevalence meaningfully on two tasks,
  stop the typed-route detector.
- If order two does not improve unlabeled prediction, remove order two rather
  than tuning it with labels.
- If endpoint coverage or paired rupture gap fails, remove the exact topology
  claim.
- If closure does not improve over rupture, do not use the lock-in narrative.
- If the high-dimensional attention signal remains below the historical RR
  spectral residual, collect Q/K/V, residual, and MLP outputs instead of adding
  more attention-only handcrafted states.

## Full run matrix

```text
task: QA / Summary / Data2txt
seed: 20260825 ... 20260829
recent_lag: fixed at 4
alpha: fixed at 0.5
backoff_tau: fixed at 32
primary score: grammar rupture
```

Hyperparameters are not selected on test labels. Any later change requires a
new discovery/confirmation split and a new result directory.

## Result interpretation

A successful run must establish three distinct facts:

1. the route grammar predicts unlabeled held-out dynamics;
2. its frozen anomaly score separates hallucination tokens;
3. exact endpoints or closure contribute beyond the simpler grammar.

Only fact 2 is required for a detector. Facts 1 and 3 are required for stronger
mechanism and graph claims.
