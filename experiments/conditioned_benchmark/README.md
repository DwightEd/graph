# Conditioned detector benchmark

This package compares already-frozen token anomaly scores under the same task,
data-source, generator, response-position, evaluation-unit, and positive-
prevalence conditions. The single workflow interface is:

```python
ConditionedBenchmark(config).run(split_root, output_dir, artifact_specs)
```

The runner captures every score file identity, strict-loads every current owner
schema, and validates every complete dataset binding before it opens canonical
labels. Detector fitting and anomaly directions are never selected after labels
become available.

## Supported artifacts

Only current owner contracts are accepted:

| Schema | Benchmark score |
|---|---|
| `cmrp-score-v2` | frozen `score` primary |
| `rr-spectral-score-v2` | frozen `score_rr_residual` primary |
| `rr-topology-dynamics-features-v3` | explicitly named `features_z` column and fixed `higher`/`lower` direction |

Legacy, unversioned, generic numeric, probe, and trajectory artifacts are
rejected. RR topology features are mechanism measurements rather than frozen
detectors, so a configuration must state their feature name and direction:

```json
{
  "name": "rr_topology",
  "path": "/absolute/path/to/test_features.npz",
  "column": "prompt_groundedness",
  "direction": "lower"
}
```

## Running

```bash
bash experiments/conditioned_benchmark/run.sh \
  /absolute/attention_cache/test \
  /absolute/output/conditioned_benchmark \
  rr_spectral=/absolute/rr_spectral/test_scores.npz \
  cmrp=/absolute/cmrp/test_scores.npz
```

For an RR topology feature or a custom condition grid, copy
`config.example.json` and run:

```bash
python -m experiments.conditioned_benchmark.main --config my_config.json
```

The default grid evaluates overall and per-task conditions at native, 1%, 3%,
5%, 10%, 25%, and 50% positive prevalence. Environment variables such as
`TASK_TYPES`, `POSITIVE_RATES`, `METRICS`, `RATIO_MODE`, `RATIO_REPEATS`, and
`BOOTSTRAP` override the shell runner defaults.

## Alignment and response labels

Every artifact is first evaluated on its own complete frozen rows. Canonical
`token_label`, full-response `response_positive`, `source_id`, and
`response_length` facts are then intersected on `(sample_id, token_index)` and
must agree across artifacts.

Each evaluation binding also carries `audit_scope`. A `complete_split` artifact
must cover exactly the dataset sample IDs, while `selected_samples` may bind a
subset of complete responses. Batch sealing validates every artifact first,
reads canonical source/response-length facts once per dataset sample across all
artifacts, and only then prepares one canonical label snapshot for projection
back to each artifact's original row order.

A relative-position window limits only the token scores being evaluated or
aggregated. A response label always comes from canonical `response_positive`
over the complete answer. Therefore an intersection or position window cannot
turn a response negative merely because it omitted the positive token.

Response score aggregation supports `max`, `mean`, and `topk_mean`.

## Positive-ratio controls and uncertainty

`ratio_mode=reweight` retains every selected row and applies constant class
weights to reach the requested prevalence. AUROC is prevalence-invariant apart
from numerical precision. AUPRC and `auprc_lift = AUPRC / prevalence` are both
prevalence-sensitive and must not be described as cross-ratio controls.

Reweighted results report source-cluster percentile intervals with:

```text
uncertainty_scope=source_cluster_bootstrap_percentile_95
ci_low, ci_high
```

An interval is emitted only when at least two finite bootstrap replicates are
available. Otherwise the JSON metric omits interval bounds and reports
`uncertainty_scope=not_estimated_insufficient_resamples`; the CSV bound cells
are blank.

`ratio_mode=subsample` performs repeated shared stratified row subsamples. Its
quantiles describe repeat-to-repeat variability, not a confidence interval:

```text
uncertainty_scope=repeated_row_subsample_variability
repeat_q025, repeat_q975
```

Native prevalence uses one full-row point estimate and reports
`uncertainty_scope=not_estimated_native_prevalence` without repeat quantiles.
A non-native target needs at least two finite repeats; otherwise it reports
`not_estimated_insufficient_resamples` and leaves the CSV quantile cells blank.

Positive-ratio manipulation is evaluation-only and never refits a reference,
density model, direction, threshold, or score calibration.

## Outputs

- `results.json`: `conditioned-detector-benchmark-v2`, conditions, metrics, and
  an artifact manifest containing resolved path, frozen SHA-256, dataset
  manifest binding, schema, and complete evaluation-row count. Top-level
  `aligned_token_rows`/`aligned_samples` describe the sealed artifact
  intersection before position filtering or response aggregation;
  `evaluated_rows`/`evaluated_samples` and `evaluation_unit` describe the rows
  actually passed to conditions and metrics.
- `metrics_long.csv`: one row per condition, method, and metric.
- `metrics_wide.csv`: one row per condition and method.
- `summary.txt`: concise run inventory and metric caveat.
