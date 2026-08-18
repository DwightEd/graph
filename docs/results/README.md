# Attention signal reporting artifacts

This directory contains presentation-ready, version-aware summaries of the
attention-graph hallucination experiments.

## Main table

- `attention_signal_summary.csv`

The CSV contains one row per feature or detector and records:

- mathematical/operational definition;
- token/layer/head granularity;
- AUROC, AUPRC, separability, or onset effect;
- observed direction;
- dataset/token scope;
- original result location;
- current implementation path;
- whether the value is an observed result, a failed baseline, or a pending
  mechanism audit.

## Two prompt-location features that are easy to confuse

### `prompt_centroid`

For one response token `t` and layer `l`, the compact layer route first keeps
the strongest retained head value for each `(prompt source, response target)`
pair. Prompt positions are normalized to `[0,1]`. The feature is the weighted
mean prompt position:

```text
centroid[t,l]
  = sum_s w_l(s -> t) * (s / (P-1))
    / sum_s w_l(s -> t)
```

It describes **where in the prompt the token directly attends**. A lower value
means the direct prompt sources are located earlier in the prompt. It does not
by itself mean weaker prompt grounding.

Implementation:

```text
attention_graph/token_representation.py
  compact_layer_structure()
```

### `prompt_provenance_centroid_hop1`

Every response token first carries the mass, first moment, and second moment of
its **direct prompt sources**. These three quantities are propagated once over
the response-to-response graph:

```text
S_t^(1) = sum_{j<t} w_l(j -> t) * S_j^(0)
centroid_t^(1) = first_moment_t^(1) / mass_t^(1)
```

It describes **which prompt region the current token inherits indirectly
through one response-history relay**. It is not direct prompt attention, not
attention entropy, and not a factuality score.

Implementation:

```text
attention_graph/token_representation.py
  compact_layer_structure()
  provenance loop, hop=1
```

## Where to inspect the original outputs

### Active RR spectral experiment

Current runner output:

```text
experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2/full/
  reference.npz
  test_scores.npz
  evaluation.json
```

Important version note: the reported historical `rr_raw_residual_energy`
AUROC `0.6601366430` was produced before commit `5fbbd95`, which hardened the
fit/calibration protocol and changed the artifact schema. A new run of the
current code must use a fresh output directory and must be reported separately.

### RR topology-dynamics mechanism audit

After a full run:

```text
experiments/rr_topology_dynamics/outputs/setwalk_coordination/full/evaluation/
  report.json
  feature_metrics.csv
  within_sample_effects.csv
  onset_effects.csv
  phase_curves.csv
  layer_metrics.csv
  spectral_rank_metrics.csv
  residual_correlations.csv
```

These features are implemented but do not yet have real full-data results in
the summary table.

### Attention multiplex signal audit

The current multiplex SVD-role audit writes:

```text
<representation_root>/signal_audit/
  feature_signal_report.json
  feature_signal_ranking.csv
  feature_signal_ranking.png
  position_reference.npz
```

This audit covers the multiplex SVD-role features in
`attention_multiplex/attention_multiplex/signal_audit.py`. It is **not** the
source of the historical `prompt_centroid` and
`prompt_provenance_centroid_hop1` numbers.

### Historical Lookback and compact graph feature screen

The exact structured output artifact for the historical Lookback 1024-D and
compact layer-structure screen was not located in the repository. The values
were recovered from the saved text summary named:

```text
粘贴的文本 (1)(3).txt
```

The committed CSV is therefore the canonical structured report of those
historical values. It explicitly marks these rows as
`historical_feature_screen` and does not misrepresent them as current
train-oriented detector results.

## Metric warning

`best_separability` is:

```text
max(AUROC, 1 - AUROC)
```

The label is used post hoc to choose whether high or low values are associated
with hallucination. It is suitable for feature discovery, but is not a
direction-frozen unsupervised detector result.
