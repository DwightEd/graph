# Initial proposal: token-level routing-basin detection

## Problem anchor

The existing RR spectral residual is an offline deviation score, not a token
detector.  It drops prompt routes, sorts away source identity, and applies PCA
to an artificial diagonal/age descriptor.  Its full-run residual is correlated
with response length and does not rise reliably at hallucination onset.

The replacement must answer a narrower question: after response token `t` has
been emitted, does the attention routing available at query `t` look like entry
into, or residence in, a narrow self-reinforcing routing basin?

## Initial method

For every response query, stream the retained sparse attention edges and use
`max(attention - attention_floor, 0)` as the observable edge signal.  Extract:

- exact-source prompt/response mass, effective source count, top-1 share;
- recent response feedback and repeated prompt-anchor identity;
- a real rolling singular spectrum of a rectangular query-by-route matrix;
- causal controls: token index, prompt length, retained mass, edge count.

Fit a label-free normal reference on one set of training source groups and
calibrate it on disjoint training source groups.  Emit four token-level
components: state novelty, transition surprise, basin commitment, and causal
residence.  Labels remain unavailable until the score artifact is frozen.

## Public seam

```python
detector = TokenRoutingDetector(config).fit(train_dataset)
scores = detector.score(test_dataset)
```

The detector owns sparse decoding, feature extraction, nuisance adjustment,
dynamics, and empirical calibration.  Persistence, CLI parsing, and post-hoc
evaluation stay outside this seam.

## Falsification criteria

The basin interpretation is unsupported if transition surprise does not rise
near onset, if commitment/residence do not remain elevated inside annotated
spans, or if route/source shuffles and controls-only scores perform as well as
the topology-aware score.
