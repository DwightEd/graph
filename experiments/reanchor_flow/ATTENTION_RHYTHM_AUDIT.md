# Attention rhythm audit: merged entry point

The raw per-head audit is in `main` since `785a1cf`. Do **not** apply the older
`attention_rhythm_audit.patch`: it adds incompatible versions of existing
`attention_rhythm.py`, `attention_rhythm_report.py`, and their tests.

This merge preserves that canonical observer, report, research-dataset adapter,
and concurrent documentation. `audit_attention_rhythm.py` is a thin compatibility
entry point, not a second implementation of the measurement algorithm.

## Run the previously supplied commands

```bash
python -m experiments.reanchor_flow.audit_attention_rhythm \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --split test --task QA --sample-id 12693 \
  --query-chunk 8 --plots-per-task 1 \
  --output experiments/reanchor_flow/outputs/attention_rhythm_qa_pilot
```

```bash
python -m experiments.reanchor_flow.audit_attention_rhythm \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --split both --samples-per-task 0 --max-response-tokens 0 \
  --query-chunk 8 --plots-per-task 1 \
  --output experiments/reanchor_flow/outputs/attention_rhythm_all
```

`--split both` maps to `--split all`, `--labels` maps to `--evaluate`, and
`--heads 10:4 20:7` maps to repeated `--head`. Multiple values after
`--sample-id` are also supported. The compatibility entry defaults to all tasks,
both splits and all available samples; the canonical `attention_rhythm_run`
entry retains its single-QA-sample defaults. A `--scans` split directory is
resolved from its `run_manifest.json`; a parent root containing train/test also
works. The source-info JSONL must be available, as required by the canonical
runner; use `--source-info` to override its path.

## Deliberate conflict resolution

The **canonical** definitions apply to both entry points: default FAI lag
10..100 (inclusive), head groups ranked per sample for descriptive analysis,
and strict adjacent-peak rules documented in [ATTENTION_RHYTHM.md](ATTENTION_RHYTHM.md).
These are not numerically identical to the earlier ZIP's strict-future,
full-suffix FAI, train-frozen head groups or SciPy-prominence convention.
Do not pool artifacts from the two implementations. The ZIP's `--future-min`,
`--future-max`, `--prominence` and `--demo` options are not ported as silent aliases.
Use the canonical `--future-lo` and positive `--future-hi` options for horizon
controls; unsupported options fail rather than silently changing the experiment.

Outputs now follow the canonical format: per-sample `.npz`, `.audit.npz`,
`.events.json`, optional `.png` and `.full_sources.png`; the output root contains
`index.json`, `summary.json` and population heatmaps. This entry does not promise
the previous ZIP's HTML gallery layout. All heads contribute to measurements;
only preselected heads/windows get full-map figures. No PCA, classifier, learned
propagation, or per-edge intervention is introduced by this merge.

```bash
python -m pytest -q \
  experiments/reanchor_flow/tests/test_attention_rhythm.py \
  experiments/reanchor_flow/tests/test_attention_rhythm_entrypoint.py
```

The new entry-point tests check argument translation and dispatch without loading
a language model. The existing numerical suite tests the actual observer/report;
its optional Hugging Face Llama integration test still requires Transformers.
No 8B/RAGTruth run is claimed by this merge.
