# Conditioned detector benchmark

This package compares **already frozen token anomaly scores** under the same
task, data-source, generator, response-position, evaluation-unit, and positive
prevalence conditions. It never refits a detector after labels are opened.

## Why this is separate from each method

The active experiments do not all expose the same kind of object:

| Experiment | Reusable object | Default benchmark treatment |
|---|---|---|
| `spectral_feasibility` | label-free calibrated token scores | automatically registered |
| `trajectory_geometry` Gate-A | label-free `score_*` token arrays | automatically registered |
| `rr_topology_dynamics` | label-free multi-dimensional mechanism features | requires an explicitly frozen column and direction |
| `mechanism_validation/predictions.npz` | exploratory probe/decoder predictions | loadable, but marked supervised when fields start with `probe_` |
| `attention_multiplex/signal_audit` | feature discovery report; train labels orient candidates | not treated as a frozen unsupervised detector |

The heavy attention extraction, reference fitting, and score construction are
run once by their owning experiment. This benchmark only aligns the saved
scores and opens labels afterward.

## One command

```bash
bash experiments/conditioned_benchmark/run.sh \
  /absolute/attention_cache/test \
  /absolute/output/conditioned_benchmark \
  rr_spectral=/absolute/rr_spectral/test_scores.npz \
  route_geometry=/absolute/trajectory_geometry/scores_label_free.npz
```

The default grid contains overall plus every `task_type`, and evaluates native,
1%, 3%, 5%, 10%, 25%, and 50% positive prevalence. Override it without editing
code:

```bash
TASK_TYPES="QA Summary Data2txt" \
POSITIVE_RATES="native 0.03 0.062086 0.10" \
METRICS="auroc auprc auprc_lift tpr_at_fpr_05" \
BOOTSTRAP=1000 \
bash experiments/conditioned_benchmark/run.sh TEST_SPLIT OUTPUT_DIR \
  rr_spectral=TEST_SCORES.npz
```

For named feature columns, lower-is-anomalous directions, data-source grids,
response-level aggregation, and other advanced settings, copy
`config.example.json` and run:

```bash
python -m experiments.conditioned_benchmark.main --config my_config.json
```

## Positive-ratio controls

`ratio_mode=reweight` is the default. It retains every selected row and applies
constant class weights so that the evaluated prevalence equals the requested
target. Therefore:

- AUROC should be unchanged apart from numerical precision;
- AUPRC changes with prevalence;
- `auprc_lift = AUPRC / prevalence` is reported for cross-ratio comparison;
- every method is evaluated on identical token rows and bootstrap draws.

`ratio_mode=subsample` performs repeated, shared stratified subsampling. It is
useful as a sensitivity analysis, but is noisier and discards data.

Positive-ratio manipulation is **evaluation-only**. It is never used to fit a
PCA, density model, anomaly direction, threshold, or score calibration.

## Token versus response conditions

Token evaluation is the default. Response-level experiments set:

```json
{
  "evaluation_unit": "response",
  "response_aggregation": "max"
}
```

Available response aggregation rules are `max`, `mean`, and `topk_mean`.
Response labels are positive when at least one response token is positive.

## Outputs

- `results.json`: full versioned report, method protocols, conditions, metrics,
  and confidence intervals.
- `metrics_long.csv`: one tidy row per condition, method, and metric.
- `metrics_wide.csv`: one human-readable row per condition and method.
- `summary.txt`: short run inventory.

All score artifacts are intersected on `(sample_id, token_index)` by default.
This prevents methods with missing rows from being compared on different test
populations. Metadata and token labels are re-read from `ResearchDataset`; any
artifact/dataset mismatch fails loudly.
