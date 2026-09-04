# Method: exposure-adjusted re-anchor audit

## 1. What this experiment can decide

The audit separates four questions that a claim-level AUROC collapses:

1. Does ordinary continuation develop a preference for response history over
   prompt evidence?
2. Does a new content boundary interrupt that preference with an
   evidence-specific read?
3. Is that entry response weaker when a hallucination begins at the boundary?
4. Is the output dependent on attention-mediated external evidence even while
   MLP parametric knowledge remains available?

The first three are preregistered observational tests. The fourth is an
optional whole-evidence dependence control. None is called a full causal graph.

## 2. Exact prediction coordinates

If response token $y_t$ occupies absolute position $p_t$, its predictive
state is at

\[
q_t=p_t-1.
\]

Every reported event is indexed by $p_t$, while the observed attention row is
$q_t$. Evaluation verifies both $q_t+1=p_t$ and the target token ID.

The removed v1 graph treated $p_t$ as though the predictive state at $q_t$
were written into token $p_t$. Teacher forcing does not contain that edge.
It also averaged layers before traversal, permitting paths whose layer order was
impossible. Its scalar can be reproduced as a historical result, but it is not
a Transformer computation path.

## 3. Observed message and availability null

For layer $l$, query head $h$, its actual GQA KV head $g(h)$, query $q$,
and source $s$, define the transported-message magnitude

\[
c_{l,h,q,s}
=
A_{l,h,q,s}
\left\lVert W_l^{O,[h]}V_{l,g(h),s}\right\rVert_2.
\]

For role $r\in\{E,O,H\}$, corresponding to evidence, other prompt, and
response history, the observed functional share is

\[
R^r_{l,p}
=
\frac{\sum_h\sum_{s\in r}c_{l,h,p-1,s}}
{\sum_h\sum_{s\le p-1}c_{l,h,p-1,s}}.
\]

Raw attention role share is retained as a selection-only control.

### Why raw shares are not a phenomenon test

At response event $t$, exactly $t$ history tokens are available. Under
uniform attention,

\[
R^H(t)=\frac{t}{n_{prompt}+t},\qquad
R^E(t)=\frac{n_E}{n_{prompt}+t}.
\]

Thus history rises and evidence falls without any learned routing change. v3
constructs a layer- and event-specific null. For attention, it is the visible
token fraction. For functional routing it is

\[
B^r_{l,p}
=
\frac{
\sum_{s\in r,\,s\le p-1}\sum_h
\left\lVert W_l^{O,[h]}V_{l,g(h),s}\right\rVert_2
}{
\sum_{s\le p-1}\sum_h
\left\lVert W_l^{O,[h]}V_{l,g(h),s}\right\rVert_2
}.
\]

The primary role enrichment is

\[
L^r_{l,p}=\log\frac{R^r_{l,p}+\epsilon}{B^r_{l,p}+\epsilon}.
\]

The main evidence variable additionally removes a generic prompt return:

\[
S_{l,p}=L^E_{l,p}-L^O_{l,p}.
\]

So $S>0$ means a preference for RAG evidence beyond the instruction,
question, BOS, and other prompt content. Layer indices are never collapsed
before these quantities are computed.

## 4. Pre-outcome event coordinate

For a boundary at response index $b$, define its entry change using only the
current predictor and earlier events:

\[
J_S(b)
=
S_b-\frac1w\sum_{k=-w}^{-1}S_{b+k},\qquad w=5.
\]

At $b$, the model query is $q=b-1$; it has not seen the token being
predicted. This is the primary H2/H3 statistic. A three-token post-window and a
longer event curve are descriptive only: after offset zero, generated tokens
have entered response history and may be consequences rather than causes.

Scalar inclusion requires only the scalar window. Plot-window truncation never
removes an otherwise valid scalar event.

## 5. Hypotheses

### H1: exposure-adjusted preference drift

For every fully clean response, compare the last and first thirds of $L^E$ and $L^H$.
The preregistered signs are

\[
\Delta L^E<0,\qquad \Delta L^H>0.
\]

Raw share changes are reported with a warning and cannot support H1.

### H2: natural-boundary evidence specificity

The label-free splitter records whether a span began at response start, after
natural punctuation, or after a length cap. Length-forced chunks are excluded.
A clean natural boundary is compared with up to three complete local-control
positions distributed inside the same span:

\[
D_{clean}=J_S(b_{boundary})-J_S(b_{inside}).
\]

The expected sign is $D_{clean}>0$. History release, the three-token pulse,
and early/middle/late layer bands are secondary descriptions.

This is a sentence-boundary proxy, not yet an atomic factual-claim test.

### H3: missed entry closed against H2

The primary positive group contains annotation runs whose first hallucinated
token occurs exactly at a natural boundary. The test is

\[
M=
\mathbb E[J_S(b)\mid hallucination\ starts\ at\ b]
-
\mathbb E[J_S(b)\mid clean\ boundary],
\]

with expected $M<0$. Because each $J_S$ is already current-minus-past, this
is a difference-in-differences across event time and correctness group. It uses
no token after the error begins.

Each exact-onset boundary is matched to a clean natural boundary in the same
response within a response-position caliper; matching prefers the same
preceding punctuation token. Only pre-event matching variables are used. H3 status is
withheld unless H2 is independently supported: without a normal boundary
re-anchor phenomenon there is nothing well-defined to call "missed".

Onsets within the first three tokens and late onsets are reported separately.
They never enter the primary H3 status. A second analysis matches every
hallucination onset to a clean token within the same response. Token identity is
used only inside a position caliper; otherwise the nearest position inside the
same caliper wins. This
matched 1:1 analysis is exploratory, and its AP is not comparable with detector
AP at natural prevalence.

## 6. Statistics and scope gate

Effects are averaged within source before inference. Whole sources are the
bootstrap unit. Reports contain event/source counts, equal-source estimates,
95% source-bootstrap intervals, and sign-flip tests where applicable.

- `supported`: the full interval has the preregistered sign;
- `contradicted`: the full interval has the opposite sign;
- `inconclusive`: otherwise, including insufficient independent sources.

If generator and observer differ, the data only show how the observer processes
a fixed answer under teacher forcing. Observer statistics remain visible, but
all generation-mechanism statuses become `not_tested_for_generation`.

## 7. Whole-evidence dependence controls

Optional post-softmax gates remove evidence messages without renormalizing
attention:

\[
A_{q,s}V_s\longrightarrow 0\quad(s\in E).
\]

The direct-response cut targets response queries. The global cut targets every
query, including potential carriers. All MLPs remain active. For the fixed
target-versus-runner margin,

\[
C_t=m_t^{full}-m_t^{cut}.
\]

$C_t>0$ means external attention-mediated evidence helps this target despite
any parametric compensation. A small $C_t$ does not imply that RAG evidence is
generally unnecessary, and these global cuts do not localize integration,
overwrite, or readout silence.

## 8. Decision to build a graph

A global information-flow graph is justified only if H2 is stable and H3 has
adequate exact-boundary power. The next graph must use layer-position residual
states plus explicit attention, MLP, and readout stages, and it must add
explanatory value beyond these event-local contrasts.

The next causal experiment should separate:

1. evidence entry;
2. support/validator integration;
3. persistence versus prior/history overwrite;
4. residual evidence that is silent at final readout.

If graph connectivity, cuts, or path interactions do not outperform matched
local summaries, the graph has not earned a role in the method.
