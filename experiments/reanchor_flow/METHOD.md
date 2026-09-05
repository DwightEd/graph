# Registered mechanism audit

The statistics in this file are discovery and mechanism tests. The deployable score, unlabelled
train calibration, temporal scope, fixed composition and frozen test-label protocol are registered
separately in [`DETECTOR.md`](DETECTOR.md).

For response token `p`, the causal predictor is `q=p-1`. The native transported-message capacity is

\[
c_{l,h,p,s}=A_{l,h,q,s}\lVert W_{O,l,h}V_{l,g(h),s}\rVert_2,
\qquad
P_{l,h,p}(s)=c_{l,h,p,s}/\sum_u c_{l,h,p,u}.
\]

Schema v8 preserves the core `P`-derived traces at `[layer, head, event]`; layer/head means are only
descriptive summaries. Capacity is not yet the signed message vector, so it is called transport
budget/share rather than semantic contribution. The operator-graph extension is specified in
`RESEARCH_PLAN.md`.

The code tests four stages instead of treating a prompt-revisit peak as the whole mechanism.

## H0: normal direct-route drift

The observed prompt, evidence and history shares are divided by a null that uses the same visible
source set and the same `||W_OV||` source capacities but removes attention selection. For role `r`,

\[
L^r_{l,p}=\log\frac{R^r_{l,p}+\epsilon}{B^r_{l,p}+\epsilon}.
\]

On fully correct responses, the registered background prediction is a negative prompt-lift slope and
a positive history-lift slope over relative response position. This concerns direct endpoints only;
it does not identify the semantic ancestry already stored inside response tokens.

## H1: internally detected transition and re-entry

A model-internal transition is a peak of the Jensen-Shannon change of the complete source
distribution. It is selected independently of prompt/evidence share. At those peaks the audit tests,
relative to matched non-events in the same answer, whether there is renewed prompt transport,
renewed RAG-context transport, predictor-state reuse, and emitted-token anchoring. Controls prefer
the same token and boundary state and balance relative position, entropy and target log-probability.
Circular shifts remain only a sensitivity summary for event-to-anchor coupling.

Prompt revisit means a renewed share assigned to any prompt source. Nonlocal review uses the
continuous weight `min((q-s)/D,1)` and therefore has no hard far-token threshold. Future influence is
split into two coordinates within horizon `H`:

- `predictor_reuse[p]`: later prediction rows read source `q=p-1`;
- `emitted_token_anchor[p]`: later prediction rows read source `p` after that token exists.

The legacy `future_influence` artifact is an alias for the second quantity.

## H2: hallucination-onset entry and anchoring

After capture, each first hallucinated token is matched to a nearby clean token in the same response.
The audit compares evidence-entry change at `q=p-1`, before the error token has entered history, and
both future quantities after the token is generated. Emitted-token anchoring measures propagation of
an error, not its cause; predictor reuse tests a distinct hidden-state coordinate. Teacher forcing
does not create a hidden-state edge from predictor `q` to emitted-token position `p`.

## H3: all-sample functional context and adoption

Every selected sample receives one context cut. The cut is evaluated against the complete
vocabulary in small event batches, producing distribution JS, context-supported alternatives,
actual-target gain/rank and adoption margin. This is the primary full-data test of whether routed
context changes the candidate distribution and is adopted by the emitted token.

## H4: grouped functional integration and state persistence

The source-diverse subset adds three cuts to the context cut already run for every sample, giving
four post-softmax, pre-Value-sum cuts on response-query rows:

- RAG evidence sources;
- other prompt sources (question/instruction operational group);
- all prompt sources;
- response-history sources.

For fixed target-versus-runner margin `F`, evidence and other-prompt necessity are `F-F_{-E}` and
`F-F_{-Q}`. Their factorial interaction is

\[
\mathcal I_{E,Q}=F-F_{-E}-F_{-Q}+F_{-(E,Q)}.
\]

The interaction is reported together with both main effects because valid integration can be
additive. `other prompt` is not claimed to be a hand-labelled validator.

The context cut is also evaluated against the complete vocabulary without storing a token-by-vocab
artifact. For baseline distribution `p` and cut distribution `p^{-C}`, the code saves distribution
JS, the top context-supported candidates under

\[
d_t^C(v)=\log p_t(v)-\log p_t^{-C}(v),
\]

the observed target's gain/rank, and adoption margin

\[
A_t^C=d_t^C(y_t)-\max_{v\ne y_t}d_t^C(v).
\]

Because the present mask covers the whole external context rather than a claim-specific support
span, these fields deliberately use the prefix `context`, not `evidence` or `fact`.

### Persistence, override and readout silence

The evidence cut is rerun while storing every decoder-layer input. For depth `l`,

\[
\delta r_{l,p}=r_{l,p}-r_{l,p}^{-E}.
\]

State presence is `||delta r||/||r||`. A fixed target-runner logit lens is applied to baseline and cut
states at every depth; their difference is evidence-state control. The final-depth value exactly
matches the grouped evidence effect. The audit also reports:

- maximum middle/late absolute control minus final absolute control (`late_control_loss`);
- response-history cut effect (`history_effect`);
- final cosine gain between the evidence-conditioned state difference and the fixed unembedding
  direction (`readout_gain`).

The registered signatures are:

- **entry failure:** low direct evidence entry and low evidence effect;
- **integration failure candidate:** entry is present but evidence/query effects and their joint
  control are weak;
- **override candidate:** middle-layer control is stronger than final control while history control
  is strong;
- **readout silence candidate:** final state presence remains high but readout gain is low.

These are group-level operational tests. They do not prove that every deleted message carries the
relevant fact, nor do they remove evidence ancestry already relayed through another prompt token.
Exact support/validator semantics require matched controlled evidence or source annotations.
