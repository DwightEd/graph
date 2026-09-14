# P3: test revisit as a selector, not as a hallucination label

2026-09-14. Development-only iteration after P1/P2. No validation labels or
model/threshold selection on validation. Reuse the fixed R04 development roster
(16 sources, 32 responses), observer Llama-3.1-8B, not original-generator traces.

## Questions fixed before extraction/evaluation

1. Does the EXISTING head-wise content-revisit operator locate annotated span
   starts, allowing an explicitly reported lead of 0, 4 or 8 tokens?
2. Is any coverage beyond equal inspection budget and position-matched controls?
3. Does holding uncertainty between reading events improve full-span detection?
   This is a small falsifiable candidate, not an established mechanism.

Existing operator: reanchor/src/decoding/reading_graph.py, SHA256
381f18933e49cfc32ee2413f8fe010ba2fa235f67124c195bc932f95bb965d2e.
Existing script settings window=16, quantile=.95 are retained, NOT tuned.
For each layer/head and pre-token query q=P+t-1, exclude special tokens,
instructions, self and newly visible keys; compare normalized attention over
common content keys using W1 (token-coordinate distance). Subtract the past
16-step median W1, clip at zero, multiply by min(original common content mass).
Mean across all 32 layers/32 heads, no selected heads. Causal past-16 .95
quantile defines active; event is an active interval's first token. No event
before t>16. Report this cold-start blind spot, not only eligible starts.

P2 mean-head cache cannot reproduce mean(W1(head)); capture head-wise statistics
from actual eager A used in A@V. Also compare a mean-attention operator as a
declared information-loss diagnostic, NOT an equivalent implementation.
Preserve per-layer/head shift and content mass, per-query mean attention, full
token identity, query positions and native entropy/NLL/margin. No full hidden/V
graph is claimed. Save frozen predictions before annotation join.

Primary exploratory detector: event_hold_entropy. Before first revisit use H_t;
at each revisit set state=H_t; retain it until the next event. It tests piecewise
decision state rather than P2's linear diffusion. Controls: native entropy,
the same state update at entropy-quantile events, and periodic 16-token updates.
None uses hallucination labels, future suffix, fitted thresholds, or sign flips.
No high-AUROC or universal-mechanism claim from this development experiment.

## Evaluation

Use the existing annotation join (any character overlap; first overlapping
token per annotated span). Report all-token AUROC/AP, full-stream onset, source
cluster bootstrap, within-answer pair-weighted and macro AUROC. Freeze all
candidate predictions before human annotation access by evaluator. The roster
and all prior predictions already served development; this is not blind study.

Selector coverage: event in [onset-lead,onset], leads 0/4/8; denominator ALL
annotated starts and a separately named t>16 subset. Also report fraction of all
tokens covered by event-forward windows and precision of those windows.
Equal number of events PER RESPONSE: retrospective top-k entropy (not online),
uniform random positions and shuffled positions within fixed 32-token blocks
(500 draws, seed 20260914). Position matching preserves each block's event
count. Report observed-minus-random onset hits and source-cluster CI; never call
window coverage AUROC. Raw recall without budget comparison is not success.

Matched onset diagnostic: among actual events, compare event windows containing
an onset against event windows containing ONLY normal tokens; separately report
entropy and revisit rank differences. No identified correct relation/owner is
inferred. Report small sample counts and abstain on one-class samples.

Proceed criterion: an event selector needs excess coverage beyond both uniform
and position-matched chance; a detector needs gains in both full-span AP and
within-answer AUROC over native entropy and event-clock controls. Otherwise
retain only the supported localization evidence and stop this candidate.
Even success cannot establish applicable-constraint reading. That requires
source relation interventions with preserved values and measured answer changes.

## Execution (existing research@03909e02, no installs)

One fresh document witness runs tests and EXACTLY ONE pilot:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
bash scripts/run_revisit_state.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p3_revisit_pilot_20260914_v1
```

Parent may run development after finite/alignment and old-operator equivalence
tests pass, native controls equal P2 on the pilot, runtime estimate < 10 minutes,
and peak GPU allocation < 23 GiB. Pilot MUST NOT load natural annotations.

```bash
bash scripts/run_revisit_state.sh development /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p3_revisit_development_20260914_v1
```

Output must not already exist; exclusive log file with matching .log suffix.
Use subprocess.Popen(start_new_session=True) because screen is unavailable.
Check GPU idle first. No push, no overwrite of old runs, no automatic retry.
