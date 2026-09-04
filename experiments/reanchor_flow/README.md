# Re-Anchor Mechanism Audit

The audit is token-aligned: response token `p` is analysed at its causal predictor `q=p-1`.
It now tests the complete registered chain instead of only drawing revisit peaks.

The full interpretation, operator-graph definitions, falsifiable failure taxonomy and phased
validation protocol are in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

## What is measured

1. **Normal direct-route drift.** Prompt/history message shares are divided by a Value-capacity
   availability null. Their slopes are estimated only on fully correct responses.
2. **Internal transition and re-entry.** Route-change peaks are detected from the complete source
   distribution, independently of prompt/evidence share. Each peak is compared with a non-event in
   the same response, matched on token identity, boundary status, relative position, entropy and
   target log-probability.
3. **Hallucination onset.** The first token of every hallucinated span is matched to a nearby clean
   token with the same pre-outcome matching protocol. Predictor-state reuse (`q`) and later use of
   the emitted-token state (`p`) are measured separately.
4. **Deep mechanism pass.** A baseline and four grouped message cuts measure direct evidence entry,
   evidence/other-prompt interaction, response-history control, layerwise evidence-conditioned
   state presence, layerwise fixed-readout control, late control loss, final readout gain, the
   context-induced vocabulary candidates, distribution JS and actual-token adoption margin.

Schema v7 saves raw attention role mass and capacity-aware transport traces for every layer and head
as float16 arrays. Population means remain available for backward-compatible reports, but they are
not a substitute for local event discovery. The legacy `future_influence` array is retained as an alias for
`emitted_token_anchor`; it is not `predictor_reuse`.

The deep cuts delete source Value messages only on response-query rows. They test direct re-entry
into response computation; they do not claim to remove every possible semantic relay already stored
inside another prompt token. `other prompt` is an operational question/instruction group, not a
hand-labelled validator. Exact support/validator claims require a controlled dataset.

Finite `--mechanism-limit` runs use a deterministic source-hash sample, taking at most one sample per
source before filling any remainder. Dataset order therefore no longer determines the deep subset.

## Run

Smoke, including one deep sample per task:

```bash
bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32
```

All samples with a 30-sample-per-task deep audit:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --query-chunk 64 \
  --mechanism-limit 30 \
  --output experiments/reanchor_flow/outputs/mechanism_v7_30
```

Deep audit every selected sample:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --query-chunk 64 \
  --mechanism-limit -1 \
  --output experiments/reanchor_flow/outputs/mechanism_v7_full
```

One selected sample automatically receives both a route figure and a layerwise mechanism figure:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --plot-sample-id 12471 \
  --plot-limit 0 \
  --query-chunk 32 \
  --output experiments/reanchor_flow/outputs/sample_12471_v7
```

## Outputs

```text
results/<task>/<sample>.npz
figures/sample_<task>_<sample>.png
figures/sample_<task>_<sample>_mechanism.png
reports/<task>/mechanism_report.json
reports/<task>/rhythm_summary.png
reports/<task>/mechanism_atlas.png
run_manifest.json
```

The terminal prints only H0 direct drift, H1 transition/re-entry, H2 onset differences, and the H3/H4
mechanism effects. No classifier or AUROC is used at this discovery stage.
