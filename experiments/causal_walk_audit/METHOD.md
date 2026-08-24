# Method: Typed Route Grammar

## 1. Observable causal graph

A retained attention event is

\[
e=(s\rightarrow t,l,h,w).
\]

The graph preserves the exact source, response target, layer, head, and retained
attention weight. Response edges are typed as `near` or `far` by causal lag.
Every response row is partitioned into prompt, response-history, self, and
unresolved mass:

\[
p^P_{tlh}+p^R_{tlh}+p^S_{tlh}+p^U_{tlh}=1.
\]

Unretained sparse-cache entries are assigned to `unresolved`; they are never
treated as observed zeros.

## 2. Typed lineage automaton

Each token/layer/head has a seven-state distribution:

```text
P0       direct prompt lineage
P_PLUS   prompt lineage after a response relay
R0       response-token base carried by self
R_NEAR   response-closed lineage using only recent RR transitions
R_FAR    response-closed lineage using only long-range RR transitions
R_MIXED  response-closed lineage using both transition types
U        unresolved lineage
```

A prompt edge adds `P0`. The diagonal carries the previous-layer state.
A response edge maps prompt lineage to `P_PLUS`; response-base lineage enters
`R_NEAR` or `R_FAR`, and a change of response-edge type enters `R_MIXED`.
Exact response endpoints select the source lineage that is transported.

The output remains

\[
q\in\mathbb R^{T\times L\times H\times 7}.
\]

Because the cache does not contain \(W_V\) or \(W_O\), cross-layer transport
uses the equal mean of the preceding heads. This assumption is explicit and is
not described as physical hidden-state contribution.

## 3. Variable-order De Bruijn grammar

For channel \(c=(l,h)\), the route sequence over generated response tokens is

\[
q_{1,c},q_{2,c},\ldots,q_{T,c}.
\]

The unlabeled train split supplies full-soft fractional counts for

\[
p_1(d\mid b)
\]

and

\[
p_2(d\mid a,b).
\]

Unlike the previous implementation, no top-k state truncation is applied. The
order-two prediction is interpolated with order one according to the observed
context support:

\[
\lambda_{c,a,b}=
\frac{N_{c,a,b}}{N_{c,a,b}+\tau},
\]

\[
p(d\mid a,b)
=
\lambda_{c,a,b}p_2(d\mid a,b)
+
(1-\lambda_{c,a,b})p_1(d\mid b).
\]

Thus sparse order-two contexts automatically back off instead of producing a
poorly supported high-order prediction.

Before phase statistics are fitted, a disjoint unlabeled source stream measures
the mean held-out gain \(H_1-H_{backoff}\). Order two is enabled only when the
mean gain is positive and more than half of token gains are positive. Otherwise
the frozen method reduces to the order-one grammar.

The token/channel grammar surprise is

\[
H_{t,c}
=
-\sum_d q_{t,c,d}\log \widehat q_{t,c,d}.
\]

`order1_surprisal_mean`, `order2_gain_mean`, and the mean interpolation weight
are frozen diagnostics. They are not separate trained detectors.

## 4. Grammar rupture

For each channel, train-only median and MAD standardize surprise:

\[
z_{t,c}
=
\frac{H_{t,c}-\operatorname{median}_c}
     {1.4826\,\operatorname{MAD}_c}.
\]

A one-sided CUSUM records a sustained violation:

\[
C_{t,c}
=
[C_{t-1,c}+z_{t,c}-\kappa]_+.
\]

A decaying memory retains recent rupture:

\[
R_{t,c}
=
\max(C_{t,c},\rho R_{t-1,c}).
\]

The primary detector calibrates and fuses \(R_{t,c}\). It does not require the
response-closure hypothesis to be true.

## 5. Closure diagnostic

Response-closed mass is

\[
D_{t,c}
=
q_{t,c,R_{\text{near}}}
+
q_{t,c,R_{\text{far}}}
+
q_{t,c,R_{\text{mixed}}}.
\]

A separate closure diagnostic combines current closure, its causal EMA,
predicted closure, and consecutive-state stability. The mechanism score

\[
R_{t,c}\times \operatorname{Closure}_{t,c}
\]

is saved as `rupture_closure_mean`, but remains secondary until it exceeds
grammar rupture on independent data.

## 6. Hierarchical calibration

Complete source groups are divided into grammar-fit, channel-calibration, and
fusion-calibration streams. For every channel, the channel stream defines an
empirical upper-tail probability. The fusion stream then calibrates:

1. Cauchy fusion across heads inside each layer;
2. empirical layer-tail probabilities;
3. Cauchy fusion across layers;
4. a final empirical global tail.

The final score is

\[
-\log_{10} p_{\text{global}}.
\]

No head or layer is selected with hallucination labels.

## 7. Topology gate

On the independent fusion stream, response endpoints are rewired while
preserving target, layer/head, edge weight, prompt/response relation, near/far
type, coarse lag bin, and causal validity. Exact topology is authorized only
when:

- at least the configured fraction of edges changes;
- rewiring increases mean grammar rupture;
- more than half of paired token gaps are positive.

A failed topology gate means the method may still model typed route dynamics,
but cannot claim that exact token endpoints add value.

## 8. Evaluation scope

The attention row for token \(t\) is aligned with the label of token \(t\).
This is post-token white-box detection under teacher forcing, not prediction
before the token is emitted. Labels are opened only in the evaluation command.
