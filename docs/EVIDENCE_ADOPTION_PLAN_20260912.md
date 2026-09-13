# Evidence adoption and downstream influence: pre-pilot protocol

2026-09-12. The user expands the question from first-error detection to continued
error adoption, causal influence range, and transitions between claims. This
supersedes the earlier attention-only context readout as the next research
direction; earlier negative results and raw artifacts remain intact.

## Questions and estimands

1. Does an attention message support or oppose an output alternative after the
   residual stream, final MLP, and final normalization transform it?
2. Does a suspected erroneous history segment affect later predictions, and does
   that dependence persist across a sentence boundary?
3. Only subsequently: can label-free internal features predict such intervention
   effects and improve factual span detection beyond entropy and a non-graph model?

Factual error spans, influence descendants, and discourse transitions are distinct
targets. Correct self-correction can depend strongly on an earlier error. High
entropy is an onset candidate, not a reliable boundary annotation. No claim of
an unsupervised factuality detector follows from this pilot.

## Frozen implementation and pilot

Use the existing local Llama-3.1-8B-Instruct, bf16/eager, without training. Recompute
the saved natural response `00012.npz` from samples_20260911_145421_235. It was
already inspected in prior mechanism work; this is a diagnostic case, not a
held-out detection benchmark. Prediction step t uses query P+t-1.

Main implementation belongs in graph/route_graph/adoption.py. The experimental
driver belongs in reanchor/src/decoding/adoption_probe.py. Do not modify the old
capture/intervention implementation or overwrite any existing result directory.

Predeclared message queries: t = 96, 99, 128, 131, 145, 166, 180. Group past tokens
by source/non-source prompt/history and causal punctuation boundaries. Grouping
does not use hallucination labels. Candidate pool: native top 8 UNION every unique
source token whose decoded text contains an alphanumeric character, excluding
special tokens. Candidate pool and units are not semantic propositions.

For the terminal decoder layer, reconstruct real head messages
`m[h,j->q] = A[h,q,j] W_O[h] V[h,j]`, with GQA value-head repetition. Record
reconstruction error against the actual attention output. Compute signed group
contributions to each top-1-versus-candidate logit margin with a gradient through
an explicit fp32 copy of the final MLP, both downstream RMS normalizations, and
the selected output weights. This is a local suffix surrogate, not an exact
all-layer attribution. Keep per-head signs before summing. Compare first-order
predictions against finite message attenuation rho = .1 and 1 in this same
surrogate; record its baseline logit mismatch against native bf16. No threshold
on these diagnostic values is tuned to labels.

Before the full pilot, engineering review clarified the saved rank check as
candidate-pool rank (not full-vocabulary fp32 rank), and the finite attenuation
arrays were extended to all candidate contrasts using the already computed
altered logits. Native top1/top2 remains the primary summary pair. The no-MLP
comparison remains limited to that pair. Smoke artifacts retain their original
code hash and are not overwritten.

For signed group contributions a, report raw positive/negative mass and
`competition = (sum(abs(a)) - abs(sum(a))) / sum(abs(a))`, with zero-mass cases
explicitly undefined. This measures opposition in an output contrast, not
contradiction between factual propositions. Also report whether the final MLP
changes the attention-only margin contribution.

For downstream influence, compare ordinary inference, an identical causal-mask
sham, outgoing key access cut for history t in [119,133), and a length-matched
earlier history cut [85,99). Activate both cuts only at input positions >= P+133;
thus prediction t=134 is the first affected readout. Keep the saved continuation
fixed. Record full-vocabulary JS divergence (nats), entropy, native argmax flips,
and the saved token's log probability change for every prediction. The cut
measures access to these key positions, not removal of all semantic information
about the event; teacher-forced continuation fixes potential mediators. Do not
declare an influence endpoint from an arbitrary JS threshold.

## Go/no-go and next experiments

- First require reconstruction, finite-difference, causal-prefix, and sham checks.
  Numerical mismatch is reported separately from actual intervention effects.
- A single chosen case cannot establish discrimination, calibration, recovery,
  generalization, or superiority over entropy. It only validates an instrument.
- Next collect interventions on disjoint unlabeled sources, covering correct,
  incorrect, irrelevant, repeated, repaired, and contradictory continuations.
  Hold source and template families out. Measure intervention prediction before
  checking span labels; preserve genuine causal versus factual targets.
- Optional small causal GNN predicts withheld intervention effects using signed
  message/residual/MLP edges. Targets come from model interventions, not truth
  labels. Compare a matched-input MLP/DeepSets and direct readouts. Keep the graph
  only if it adds measured value. This is self-supervision, not training-free.
- Evaluate factual continuation and recovery with source-grounded labels,
  source-balanced span precision/recall, matched false-alarm budgets, onset/end
  delay and paired intervals. Topic transition alone is never a recovery label.

Scientific review backend is unavailable in this environment. Status:
REVIEW_UNAVAILABLE / pending independent scientific review. Engineering review
and a successful smoke run are not substitutes.

## Local invocation (existing environment, no install)

From reanchor, smoke: `bash scripts/run_adoption_probe.sh --queries 128 --output
outputs/adoption_smoke_20260912`. Full pilot: `bash scripts/run_adoption_probe.sh
--output outputs/adoption_pilot_20260912`. Each directory must be new. These run
the same saved case; smoke restricts the expensive message readout to one query.
Both run the sham and history-access interventions. A repeated invocation must
choose a fresh directory; automatic resume is intentionally unsupported.

## Post-v1 numerical iteration (predeclared before v2 execution)

The full seven-query v1 pilot exposed large full-removal linearization errors at
t145/t166, and a 17.9% error even at rho=.1 for t166. Add rho=.01 to test a smaller
local neighborhood, keeping .1/1 and all seven queries unchanged. This extension
was chosen after seeing v1 and is not presented as an original preregistration
or a new detection result. V1 output and verified executed sources are preserved.
V2 now saves its executed source files directly and records all three fractions.

V2 invocation from reanchor: `bash scripts/run_adoption_probe.sh --output
outputs/adoption_pilot_v2_20260912`. Analyze after completion with the existing
research Python: `python scripts/analyze_adoption_probe.py
outputs/adoption_pilot_v2_20260912 results/adoption_analysis_v2_20260912`.
