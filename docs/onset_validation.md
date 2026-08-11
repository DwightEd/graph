# Paired onset validation

`scripts/validate_onsets.py` tests whether prespecified structural attention
features change at labeled hallucination onsets beyond changes in matched,
fully correct responses. It reads one canonical split and its evaluation label
sidecar; it rebuilds original causal relations directly from canonical
attention, so it does not take a graph-cache path or a threshold option.

Install the analysis dependencies, then run the test split on the remote data:

```bash
python -m pip install -r requirements.txt -r requirements-analysis.txt
python scripts/validate_onsets.py \
  --canonical-split /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --output-dir outputs/onset_validation/test \
  --device cuda \
  --effect-width 3 \
  --bootstraps 10000 \
  --permutations 10000 \
  --rewires 100 \
  --rewire-burn-in-sweeps 10 \
  --rewire-thinning-sweeps 2 \
  --seed 0
```

The only required paths are `--canonical-split` and `--output-dir`. The other
arguments set the fixed onset width, resampling budgets, topology-null chain,
and random seed. `--device` defaults to `cpu`; the remote command uses `cuda`
because that machine has a compatible PyTorch installation.

The output directory contains seven files:

- `matches.csv`: exact-metadata-stratum, one-to-one error/control matches and
  merged positive spans.
- `pair_effects.csv`: error and normalized-position control onset deltas for
  all 12 structural features. These are exploratory except for the five
  features selected for the primary analysis.
- `event_study.csv`: onset-relative mean error, control, and paired-effect
  trajectories.
- `primary_effects.csv`: bootstrap intervals, paired sign-flip p-values, and
  Holm-adjusted p-values for the five prespecified primary features.
- `rewire_null.csv`: history-lag effects after causal directed rewiring that
  retains node degrees, edge types, targets, and sparse channel payloads.
- `event_study.png`: a plot of the onset-relative paired trajectories.
- `metadata.json`: input hashes, method version, configuration, event counts,
  matching audit information, and the topology-null summary.

## Hypothesis and limits

The confirmatory question is whether the five prespecified structural features
(`prompt_mass_share`, `normalized_entropy`, `history_lag`, `in_density`, and
`history_edge_share`) have a paired onset effect after matching fully correct
responses in the same metadata stratum. Only these five tests enter the Holm
family. The other seven columns in `pair_effects.csv` are exploratory.

The history-lag topology test is a secondary sensitivity analysis and is not
included in the five-test Holm correction. It compares the observed effect to
an approximate lazy constrained-rewire MCMC null. The source swaps hold all 11
other structural features fixed and can change only `history_lag`. The reported
p value is therefore an approximate MCMC randomization p-value. Burn-in and
thinning are configurable, but chain mixing has not been established; this is
not an exactly calibrated randomization test.

The input is not complete attention. The original graph is reconstructed only
from canonical CSR entries with attention strictly greater than the archive's
`attention_floor`; an absent weak edge is known only to be at or below that
floor. Prompt shares, entropy, topology, and rewiring results therefore apply
to the retained-edge graph, not to the model's full attention distribution.
Claims about weak connections require a separate lower-floor extraction and a
floor-sensitivity analysis.

This is a label-conditioned, retrospective event study. It is not an online
hallucination detector: canonical attention uses
`post_token_query_at_same_position`, so features at position `t` are available
only after the observer has read token `t`. The matched and rewired comparisons
reduce selected confounds but do not establish that an attention pattern causes
a hallucination. A merged onset without a complete, equally wide pre-window is
excluded; its sample ID and the exclusion count are recorded in `metadata.json`.
Conclusions apply only to onsets with that pre-window and an exact-stratum
control that entered the matched analysis.

When one response contains multiple retained onsets, their effects are averaged
first. The matched sample pair is therefore the bootstrap and sign-flip unit.
The current analysis does not cluster-bootstrap repeated `source_id` values;
`matches.csv` records both source IDs for auditing. If sources repeat across
pairs, inference should be rerun with source-clustered resampling.
