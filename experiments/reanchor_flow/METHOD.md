# Registered mechanism audit

For response token `p`, the causal predictor is `q=p-1`. The native transported-message capacity is

\[
c_{l,h,p,s}=A_{l,h,q,s}\lVert W_{O,l,h}V_{l,g(h),s}\rVert_2,
\qquad
P_{l,h,p}(s)=c_{l,h,p,s}/\sum_u c_{l,h,p,u}.
\]

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
relative to circularly shifted event locations, whether there is renewed prompt transport, renewed
RAG-evidence transport, and subsequent future influence. Punctuation is only a visual reference.

Prompt revisit means a renewed share assigned to any prompt source. Nonlocal review uses the
continuous weight `min((q-s)/D,1)` and therefore has no hard far-token threshold. Future influence is
the mean normalized message share later prediction events assign to token `p` within horizon `H`.

## H2: hallucination-onset entry and anchoring

After capture, each first hallucinated token is matched to a nearby clean token in the same response.
The audit compares evidence-entry change at `q=p-1`, before the error token has entered history, and
future influence after the token is generated. The latter measures propagation of an error, not its
cause.

## H3: grouped functional integration

The optional deep pass performs four post-softmax, pre-Value-sum cuts on response-query rows:

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

## H4: persistence, override and readout silence

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
