# Re-Anchor Mechanism Audit

The audit is token-aligned: response token `p` is analysed at its causal predictor `q=p-1`.
It now tests the complete registered chain instead of only drawing revisit peaks.

The full interpretation, operator-graph definitions, falsifiable failure taxonomy and phased
validation protocol are in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md). The frozen train-to-test
detection method is specified in [`DETECTOR.md`](DETECTOR.md).

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
4. **Functional context pass for every sample.** One context cut measures target effect, the
   context-induced vocabulary candidates, distribution JS and actual-token adoption margin.
5. **Grouped mechanism subset.** Three additional cuts measure evidence/other-prompt interaction,
   response-history control, layerwise evidence-conditioned state presence, late control loss and
   final readout gain.
6. **Unsupervised failure detection.** Task-specific source-balanced conditional CDFs are fitted on
   unlabelled train captures. A causal-prefix transport/adoption score is frozen for test before
   labels are opened; future reuse is reported only as a secondary offline score.

Schema v8 saves raw attention role mass and capacity-aware transport traces for every layer and head
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

Smoke on the test split, including one grouped sample per task:

```bash
bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32
```

All train and test samples receive the functional context pass. Thirty source-diverse samples per
task and split additionally receive the grouped audit. This one command also runs the frozen
detector before the descriptive labelled reports:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --split all \
  --query-chunk 64 \
  --mechanism-limit 30 \
  --output experiments/reanchor_flow/outputs/reanchor_v8_all
```

Run only the detector on an existing complete train/test capture:

```bash
python -m experiments.reanchor_flow.run detect \
  --output experiments/reanchor_flow/outputs/reanchor_v8_all
```

Deep audit every selected sample:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --query-chunk 64 \
  --mechanism-limit -1 \
  --output experiments/reanchor_flow/outputs/reanchor_v8_full
```

One selected sample automatically receives both a route figure and a layerwise mechanism figure:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --plot-sample-id 12471 \
  --plot-limit 0 \
  --query-chunk 32 \
  --output experiments/reanchor_flow/outputs/sample_12471_v8
```

## Outputs

```text
results/<task>/<sample>.npz
figures/sample_<task>_<sample>.png
figures/sample_<task>_<sample>_mechanism.png
reports/<task>/mechanism_report.json
reports/<task>/rhythm_summary.png
reports/<task>/mechanism_atlas.png
detection/token_scores.npz
detection/detection_report.json
run_manifest.json
```

With `--split all`, each layout is nested under `<output>/train/` or
`<output>/test/`; a single-split run keeps the established layout directly under `<output>/`.
Train and test mechanism summaries are reported separately. The detector is calibrated only on
unlabelled, source-disjoint train captures and reports held-out test token/onset AUROC and AUPRC.
Re-running the same command resumes completed samples inside each split; changing capture flags
requires a new output directory.
