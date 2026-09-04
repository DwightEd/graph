# Re-Anchor Phenomenon Audit v3

This experiment tests whether a re-anchor phenomenon exists before any
hallucination detector is trained. The old claim-level AUROC score is not the
main result: one scalar failing to rank hallucinations cannot decide whether a
multi-stage mechanism exists.

## Primary variables

At the query `q=p-1` that predicts response token `p`, the observer records per
layer and source role:

- attention share `A`;
- transported-message share `A * ||W_O V||`;
- an availability null for each share.

The attention null is the fraction of currently visible source tokens in each
role. The functional null also weights every visible source by
`sum_h ||W_O[h] V[h,s]||`. Thus a growing response-history pool cannot by
itself produce a positive result.

The main signal is

```text
role log lift = log(observed share / availability-null share)
evidence specificity = evidence log lift - other-prompt log lift
```

The subtraction distinguishes an evidence re-read from a generic return to the
question, instruction, BOS, or other prompt tokens. Attention-only results are
reported beside `AVW_O` results rather than presented as transported content.

## Three preregistered phenomenon tests

1. **H1, exposure-adjusted drift.** Compare the last and first response thirds.
   Evidence enrichment should fall and history enrichment should rise after
   correcting for the number/capacity of visible sources. Raw share drift is
   descriptive only.
2. **H2, clean boundary re-anchor.** At natural punctuation boundaries, compare
   the center-only evidence-specificity entry change with up to three distributed
   local controls inside the same clean span. Length-forced chunks are excluded.
3. **H3, missed entry.** Pair hallucination runs beginning exactly at a natural
   boundary with nearby clean boundaries in the same response, matching only
   pre-event position and preferably preceding punctuation. This is a difference-in-differences
   relative to each boundary's own pre-window and is gated on H2 being supported.
   Near-boundary and late onsets are reported separately and never pooled into
   the primary status.

Only event offset `0` is used for the pre-outcome H2/H3 statistic. Its
query has not seen the token being predicted. Offsets `>0` are visualized as
post-generation persistence/recovery, not treated as a cause of that error.
The long plotting window never determines which events enter a scalar test.

Sentence boundaries and the complete RAG evidence span are still proxies. A
positive v3 result justifies the next experiment with atomic-claim boundaries
and claim-specific support/validator roots; it does not already establish that
full mechanism.

## Optional whole-evidence controls

`--causal-cuts` adds two reruns:

- delete evidence messages into response queries;
- delete evidence messages into every query.

Both interventions happen after softmax without renormalization. MLPs stay
active, so parametric knowledge can compensate. These controls measure whether
the fixed answer currently depends on attention-mediated evidence; they are not
called localized re-anchor-circuit interventions.

## Memory and runtime

Attention scores, causal masks, and deletion masks are all constructed by query
chunk. `W_O` Gram matrices are built in four-head blocks, cached once per model
on CPU, and reused across samples. Route summaries remain on GPU for one layer
and transfer to CPU once per layer, then each sample is saved and released
before the next sample.

The default `--query-chunk 64` is usually appropriate for a 24 GiB GPU. Use 32
when the GPU is shared.

## Run

Smoke test:

```bash
bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32
```

Twenty samples per task, observational audit only:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --limit 20 \
  --query-chunk 64 \
  --output experiments/reanchor_flow/outputs/pilot20_v3
```

Full audit:

```bash
bash experiments/reanchor_flow/run_all.sh --query-chunk 64
```

Add the expensive evidence-dependence controls only to a balanced pilot:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --limit 20 \
  --query-chunk 32 \
  --causal-cuts \
  --output experiments/reanchor_flow/outputs/causal_pilot_v3
```

The default output is deliberately new:

```text
experiments/reanchor_flow/outputs/<model>/phenomenon_v3/
  results/<Task>/<sample>.npz
  figures/<Task>/<sample>.png
  reports/<task>/phenomenon_report.json
  reports/<task>/phenomenon_audit.png
  run_manifest.json
```

Schema-v1/v2 outputs are rejected instead of being silently mixed with v3.

## Interpret the report

| Result | Valid conclusion |
|---|---|
| H1 fails | Raw prompt/history drift was source-pool geometry, or this observer has no preference drift. |
| H1 passes, H2 fails | Global drift exists, but no sentence-boundary evidence-specific reset is detected. |
| H2 passes, H3 fails | A boundary rhythm exists, but missed entry is not supported as the hallucination mechanism. |
| H2 passes, H3 inconclusive | Increase exact-boundary positive/source counts before detector design. |
| H1-H3 pass | Align atomic claims to support/validators and test localized integration, overwrite, and readout interventions. |

An interval crossing zero is `inconclusive`, never evidence that the mechanism
does not exist. If generator and observer differ, generation-mechanism statuses
are withheld; only teacher-forced observer processing is reported.

## Verify

```bash
python -m pytest -q \
  experiments/reanchor_flow/tests \
  experiments/common/tests/test_llama_message_intervention.py
python -m compileall -q experiments/common experiments/reanchor_flow
bash -n experiments/reanchor_flow/run_all.sh
```
