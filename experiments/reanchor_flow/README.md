# Internal Routing Rhythm Audit

This experiment first discovers model-internal events; punctuation is only a visual reference.
For every response token `p` (observed at `q=p-1`) it records:

- `route_change`: JS change of the full `A ||W_O V||` source distribution;
- `revisit_delta`: renewed transport from far prompt tokens relative to the previous window;
- `prompt_breadth`: selective re-read versus broad prompt review;
- `future_influence`: how strongly later queries reuse token `p`.

Revisit peaks and future-influence peaks are detected independently. Their short-lag coupling is
compared with a circular-shift null. Hallucination labels are opened only after capture, to compare
the first hallucinated token with a nearby clean token in the same response.

## Run

```bash
bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32

bash experiments/reanchor_flow/run_all.sh \
  --limit 20 \
  --query-chunk 64 \
  --output experiments/reanchor_flow/outputs/pilot_v4

bash experiments/reanchor_flow/run_all.sh --query-chunk 64
```

To draw a particular sample:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --plot-sample-id 12471 \
  --plot-limit 0 \
  --output experiments/reanchor_flow/outputs/sample_12471_v4
```

## Outputs

```text
outputs/<model>/rhythm_v4/
  results/<task>/<sample>.npz
  figures/sample_<task>_<sample>.png
  reports/<task>/rhythm_report.json
  reports/<task>/rhythm_summary.png
  run_manifest.json
```

The terminal prints only peak coupling and hallucination-onset matched effects. The one-sample
figure contains the source-to-token route map, the revisit-anchor curves, a head-resolved prompt
read map, and a sparse top-route graph.
