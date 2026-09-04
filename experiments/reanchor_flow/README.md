# Internal Routing Rhythm Audit

The audit discovers model-internal events for every response token `p`, using the predictor
`q=p-1`. Punctuation is shown only as a visual reference.

The Value-aware source distribution is

```text
P(s | p,l,h) ∝ A[l,h,q,s] * ||W_O[l,h] V[l,g(h),s]||
```

Four token trajectories are retained:

- `route_change`: JS change from the preceding source distributions;
- `prompt_delta`: renewed transport from any prompt token;
- `nonlocal_delta`: renewed continuous expected source distance, not a hard far-token cutoff;
- `future_influence`: later tokens' reuse of the generated token.

Prompt-revisit peaks and nonlocal-review peaks are detected independently. Each is paired with
future-anchor peaks and compared with a circular-shift null. Reports include both pooled peak
coupling and the source-level mean coupling lift with a bootstrap confidence interval, so a
population claim is not inferred from one sample or from pooled peaks alone.

## Run

```bash
bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32

bash experiments/reanchor_flow/run_all.sh \
  --limit 20 \
  --query-chunk 64 \
  --output experiments/reanchor_flow/outputs/pilot_v5

bash experiments/reanchor_flow/run_all.sh --query-chunk 64
```

`--distance-scale 16` is the distance at which the continuous nonlocal weight saturates. A source
at lag `d` receives weight `min(d / distance_scale, 1)`; no source is excluded by a distance gate.

To draw one sample:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --plot-sample-id 12471 \
  --plot-limit 0 \
  --output experiments/reanchor_flow/outputs/sample_12471_v5
```

Default outputs:

```text
outputs/<model>/rhythm_v5/
  results/<task>/<sample>.npz
  figures/sample_<task>_<sample>.png
  reports/<task>/rhythm_report.json
  reports/<task>/rhythm_summary.png
  run_manifest.json
```
