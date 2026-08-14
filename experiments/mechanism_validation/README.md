# Mechanism validation experiment

Run from `research/graph` with `python -m experiments.mechanism_validation.main`.

1. `screen` is label-free: it streams a split and writes one `<sample_id>.pt` compact-feature artifact, `metadata.json`, and a concise `index.json` with sample files and token counts. Artifacts contain values, validity masks, IDs, and prompt length; labels are never copied. Metadata records schema, EMA decay, and the sparse-attention floor.
2. `evaluate-mechanisms` loads both completed label-free directories before labels. `--max-train-tokens` selects global train rows with the fixed seed before concatenation; test is always full. Univariates report raw held-out metrics, the train-derived direction, and train-oriented held-out metrics. Bootstrap (default 200) is only for predeclared `global_mean` univariates; the rest are point estimates. The report also audits train/test sample and source overlap.
3. Sparse-floor summaries use explicit cache-proxy bounds: they treat fp16 cache values as exact and complete `OTHER` to one, not as unconditional bounds on dense original attention. Missing values remain invalid. EMA carries the last valid state; innovations require a valid current value and valid history.
4. `build-graph` stores fixed response node features and their masks once in `base/`; variants store graph descriptors. Random weight/source ablations derive their seed from the global seed and sample ID. Metadata declares `randomization_repeats=1` and the outputs are exploratory.
5. `evaluate-graphs` reports two distinct diagnostics: `representation_sufficiency` retrains for each variant, while `decoder_sensitivity` fixes exact decoders and applies them to every test variant. `nuisance_only` includes position, length, task, and source controls. The 200-resample intervals are exploratory; publication claims need more resamples and a new source-disjoint holdout. `rp_only`/`rr_only` are current asymmetric descriptor sufficiency comparisons, not intrinsic source attribution.

`results.json` is concise; `predictions.npz` is the detailed held-out output. The diagnostic probes use train labels and therefore measure available discriminative information; they are not the final unsupervised detector. Remove `experiments/mechanism_validation/` to delete all of this experiment without changing the base project.

For an exploratory randomization stability check, rerun the complete build and evaluation with a different `--seed` (for example `--seed 1`) and compare the two output directories. `run.sh` accepts `ROOT` and `OUT` environment overrides.
