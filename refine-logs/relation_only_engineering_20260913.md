# Relation-only O4 engineering review — 2026-09-13

Separate engineering-review agent applying
`/root/.agents/skills/code-review-and-quality/SKILL.md`. Reviewed the frozen plan,
`relation_readout.py` and tests, `relation_inputs.py`, `relation_only.py`, its
launcher, and the subsequently supplied `interleave_relation_only.py` scheduler.
Earlier ownership/population review records are preserved.

This reviewer performed CPU/read-only checks and wrote this report. No GPU work,
signals to the population process, implementation edits, parameter changes, or
real-output edits were performed. Scientific review remains unavailable:
`REVIEW_UNAVAILABLE`.

## Verdict

**Engineering review: approve. No unresolved Required issue.** The separate
runtime witness may execute the authorized pause/O4/resume sequence. Successful
engineering checks do not establish scientific validity or detector performance.

## Issues resolved during review

- **Candidate-use description:** The initial plan said numeric candidates were
  used only in evaluation, although their identities already selected source
  nodes for readouts and endpoint interventions. The revised plan correctly states
  that candidate nodes are fixed beforehand, while expected-answer/stage labels
  enter evaluation only and do not determine scores or fitted weights.
- **Complete fixed condition count:** Four `all_queries_x` conditions were present
  in the driver but absent from the initial plan. The plan now explicitly includes
  them, giving 28 fixed patches plus the 132-query scan.
- **World-level full-vocabulary effect:** The driver now writes
  `world_effects.json`, comparing world 0 to world 1 separately for both query
  contexts using complete endpoint vocabulary logits. The fixed observed-token
  index is 12 in both contexts, matching the inspected input arrays.
- **Scheduler concurrency:** Initial existence checks did not atomically exclude
  two scheduler processes. One could launch O4 while the other's log-creation
  failure entered `finally` and prematurely resumed population. The revised
  scheduler takes an independent nonblocking exclusive file lock before signals
  and holds it across the entire sequence. After population exits it also holds
  the population run's lock until the relation child ends, then releases it before
  launching the frozen population command with `--resume`.

## Semantic control and actual input identity

The actual response says **“Remove the bratwurst from the grill and cook the
onions in the beer mixture for 10 to 12 minutes.”** The new rules name this same
from-the-grill removal action. This corrects the earlier shorthand that described
removal from the beer mixture. The original source does not supply a unique
explicit duration for this full combined action; changing its number to 14 is
not established as a natural factual repair.

The experiment replaces the specified original source sentences with constructed
Before/After rules, preserves the remaining original chat prompt, and separately
retains the actual response or constructs the opposite-stage prefix. The
real-after interpretation follows the declared removal-then-cooking reading of
the original imperative. The contrast explicitly says “Before removing…”. The
expected durations belong to these constructed rules, not to newly labeled
natural RAGTruth examples.

Independently rebuilt all four inputs with the local tokenizer:

| Context | World | Prompt length | Prediction step | Absolute query | Observed token | Constructed expected value |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| real_after | 0 | 549 | 131 | 679 | 12 | 14 |
| real_after | 1 | 549 | 131 | 679 | 12 | 12 |
| contrast_before | 0 | 549 | 132 | 680 | 12 | 12 |
| contrast_before | 1 | 549 | 132 | 680 | 12 | 14 |

Source numeric nodes are fixed at **89 → token 717 (12)** and **114 → token 975
(14)**. Only positions **67 and 92** exchange the single tokens ` Before` and
` After`. Source-world pairs preserve complete token counts, token multisets,
numeric IDs/positions, source positions, and response tokens. The real-after
response is token-for-token identical to the stored natural response. Every
source key precedes the first prediction query. The natural reference separately
uses the original prompt and its original grill/onion readouts; it is not assigned
a new correctness label by this code.

The literal numbers stay fixed; their contextual V representations may change
because the preceding relation words change. Thus X exchanges contextual source
representations, not raw numeric identities stripped of context.

## Readout dimensions and computation

For the local model, source V/K use KV-head dimensions, query Q uses attention-head
dimensions, and A retains the actual attention-head/source axes. Q and K are
captured after projection and before RoPE and are archived only; the current
compatibility scores do not use them. Query residuals are taken at layer input,
in the residual-stream coordinates shared with O's output.

The N/G computation repeats KV heads in the native GQA grouping, concatenates
heads in O's column order, and applies the actual O weights. N is the cosine of
the query residual with the node's unit-weight projected message. G projects the
actual per-head A-weighted message onto the normalized query and divides by that
node's unit-message norm. A averages node attention over heads. N/G combine heads
through O; the stored raw A/V/Q/K retain their head axes. Readout tensors have
shape `[layer, query, two numeric nodes]`, and final selection averages all layers
with fixed equal weights. Candidate ordering is 12/14 in these tensors and
14-minus-12 in the separate logit margin; the driver uses each ordering correctly.

An independent CPU calculation explicitly summed each head's projected V under
GQA and reproduced N and G. Maximum absolute differences were **8.20e-8** and
**7.45e-9**, respectively. The parent reported both tiny tests passing, including
native-forward identity, Q/K shapes, and hook cleanup; this reviewer did not
redundantly rerun that suite.

N, A, and G are three fixed readouts of the same contextual model execution.
N's contextual residuals and V already reflect the model's prior attention
computation. Consequently this comparison is not a graph architecture versus
an otherwise independent non-graph model, and no graph-model necessity or
generalization gain follows from four constructed cases. G is a compatibility
quantity, not a logit derivative or a proven measure of output adoption.

## Intervention, sham, and scan audit

Donors always come from the opposite source world of the same query context.
There is no cross-context swap between the differently sized response prefixes.
Captured factor query indices are prediction steps; patches convert them to
absolute queries with `P + step - 1`. This correctly handles both the first query
and the different final steps 131/132.

All 32 layers receive X, E, XE, MLP, or numeric-endpoint interventions in both
directions. The reused ownership operator reads current receiving A/V at each
layer, preserves receiving source mass for donor-E exchanges, and uses current
A for the endpoint permutation with V fixed. O is replayed at full original
shape, while residuals, MLP, and subsequent layers are recomputed normally.

Each of the four worlds has one same-world final-query X sham, five final-query
interventions, and one all-query X intervention: **28 fixed patches**. The real
after/world-0 scan adds one all-layer X intervention at each step **0–131**, for
**132 scan patches / 160 total patches**. Every input world saves all prediction
logits through its endpoint. Each patch saves the full endpoint vocabulary
logits and diagnostics. Earlier queried logits must be exactly unchanged, and
same-world sham additionally requires exact equality of the entire measured
query matrix. For a step-0/all-query intervention there is no earlier answer
query; an earlier-query count of zero does not assert a nonempty prefix check.

The scan measures individual-query sufficiency under this particular donor
intervention and known endpoint. The top-eight ranking is computed from clipped
positive changes in mean baseline source attention, with the first score fixed
at zero and deterministic tie order. It does not use scan outcomes to construct
the ranking. Any later recall report must identify the exact scan success set
(for example any native argmax change versus transfer to the declared numeric
candidate) and preserve the full scan results. This is not automatic first-error
localization or a population recall estimate.

## Scheduler and population protection

The scheduler identifies the current target by module command line, exact output
argument, progress PID/status, and frozen population code hashes. Independently
inspected the current command and scientific settings: the running population
uses the default frozen model/dataset, max input 8192, chunk 64, history window 16,
seed 20260912, and smoke=false. The scheduler's resume command matches that actual
configuration; the population runner also verifies its full contract on resume.

The final scheduler checks fresh journal/output/log paths, takes its independent
lock before signaling, sends SIGINT only to the validated population PID, and
does not launch O4 until that process exits. It then holds the population run lock
during O4, preventing an external population resume from loading another model.
Its `finally` path waits for a still-running relation child to exit before
releasing the run lock and starting population resume. The journal records the
old progress, relation PID/exit status, and resumed PID/log.

This is static engineering review, not a signal or GPU fault-injection witness.
The separate runtime witness must verify actual population resumption and new
published work. Use a post-resume verified count or a response published after
the resume timestamp: a pre-pause progress snapshot can lag atomic publication
by one response, so a recount alone is insufficient proof of new processing.
No population implementation or scientific setting was changed by this review.

## Interpretation limits

Preserve results for both stages and both worlds, including failed readouts or
failed transfers. Neither a constructed truth table, a numeric flip, nor success
at selected internal nodes adds natural hallucination samples or demonstrates an
untrained detector. Independent-source validation, continuous error endpoints,
and a justified graph-versus-non-graph gain remain unestablished by O4.

`REVIEW_UNAVAILABLE` must remain attached to scientific-review status regardless
of this engineering approval or the subsequent runtime outcome.
