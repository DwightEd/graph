# Dual-register attention mechanism audit

## Research question

A generated token may legitimately attend to earlier response tokens. The
mechanism question is therefore not whether response attention exists, but
whether the response-carrier route remains connected to external evidence or
becomes autonomous after direct evidence writes are removed.

The audit is frozen-model and teacher-forced. It measures the observed response
and the causal predictor position

\[
q_t = P - 1 + t,
\]

where `P` is the first response position and `t` indexes response targets. The
current target token is never visible at its own predictor.

## Four aligned replay branches

The same sequence is replayed in four branches:

| Branch | Removed attention writes at response predictor queries |
|---|---|
| `full` | none |
| `no_evidence` | direct evidence-source writes |
| `no_history` | strict response-history writes |
| `no_evidence_history` | both |

Deletion occurs after attention probabilities are formed and before the
source-value sum is written through the matching layer's `W_O`. The deleted
mass is not renormalized. Later Q/K/V states and MLP states evolve from the
modified residual stream.

## Finite-difference registers

For every captured hidden quantity the audit forms

\[
P_{reg}=F-noE,
\qquad
R_{reg}=noE-noEH.
\]

`P_reg` is the evidence-adoption branch difference and `R_reg` is autonomous
history under the evidence-cut branch. They are operational contrasts, not
claims of unique full ancestry.

Every layer preserves the native residual identity

\[
x_{out}=x_{in}+\Delta x_{attn}+\Delta x_{mlp}.
\]

The same finite difference is applied to each term. The artifact stores the
four stage norms, cross-layer step Gram, MLP alignment, factorial interaction,
and closure errors.

## Exact residual-message routes

For query head `h`, source `j`, target predictor `q`, and layer `l`, the
head-resolved residual write is

\[
m_{j\to q}^{l,h}=W_{O,l,h}
\left(A_{qj}^{l,h}V_j^{l,\kappa(h)}\right).
\]

The implementation uses the actual branch-specific attention probabilities,
value states, GQA mapping, and matching `W_O` block. It never substitutes bare
attention, a hidden-state surrogate, or an averaged head.

For a branch difference, each non-root edge obeys the exact midpoint identity

\[
\Delta(AV)=\operatorname{mean}(A)\Delta V+
\Delta A\operatorname{mean}(V).
\]

These terms are stored as `carrier` and `gate`; the directly removed source
write is stored as `root`. Their signed residual contributions reconstruct the
complete branch-difference attention write. `gate` includes Q/K changes and
softmax competition and is not presented as a unique causal attribution.

## Labels and claims

Capture, graph construction, raw geometry, and score construction never open
hallucination labels. Labels are loaded only after all arrays and validity masks
have been frozen, for task-separated QA, Summary, and Data2txt evaluation.

The audit does not claim complete causal flow, unique token ancestry, or
identification of parametric knowledge. AVWO tracing is prior art; the formal
question is whether response-history routes remain evidence-conditioned or
become autonomous, including the explicit role of MLP finite differences.

## Shortcut-route completeness audit

For prediction position `q`, let `H_q` denote strict response-history sources.
The full response-history write is

\[
h_q^l = W_O^l\operatorname{concat}_a\sum_{j\in H_q}
A_{F,qj}^{l,a}V_{F,j}^{l,\kappa(a)}.
\]

Deleting direct evidence gives the exact midpoint decomposition over response
carriers

\[
e_{\mathrm{carrier},q}^l = W_O^l\operatorname{concat}_a
\sum_{j\in H_q}\frac{A_F+A_{noE}}{2}
(V_F-V_{noE}),
\]

\[
e_{\mathrm{gate},q}^l = W_O^l\operatorname{concat}_a
\sum_{j\in H_q}(A_F-A_{noE})
\frac{V_F+V_{noE}}{2}.
\]

The direct evidence write and the exact history-root write for `noE - noEH`
complete the observed vector set. Capture verifies the midpoint identity for
every layer and prediction event, then stores its closure error together with
the aggregate and per-head Gram matrices. It also swaps adjacent response value
endpoints before the two
relay calculations. This control keeps the coefficient and value multisets but
breaks the observed endpoint pairing with at most one-token displacement inside
each pair.

Let `S=[direct evidence, carrier, gate]`. Route completion is the fraction of
the full-history energy projected onto `span(S)`. The shortcut candidate is the
unexplained energy fraction multiplied by the positive cosine between the
residualized full-history and autonomous-history writes. These directions are
frozen before labels are opened. They remain mechanism-audit candidates; the
locked primary detector is unchanged until full QA, Summary, and Data2txt
evaluation supports replacement.

