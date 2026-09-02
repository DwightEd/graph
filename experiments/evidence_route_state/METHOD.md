# Registered information-route graph

## Research question

The method does not assume that hallucination has one scalar signature. A
focused route can be correct in extractive QA, span summarization, or
record-to-text generation; an answer-dominated representation can likewise be
either a valid completion or an unsupported one. The question is therefore:

> Is the complete information-route graph at the current prediction a normal
> next state for this task, position, prompt scale, and the two preceding route
> states?

The graph separates where information originated, where attention selected it,
what was actually written into the residual stream, what the MLP added, and
which origin supported the final token margin. The detector models the normal
multimodal transitions of that graph without hallucination labels.

The two-pathway observation in prior work motivates multimodality; it does not
justify forcing samples into two discrete pathway classes. This method uses
neither a truthfulness probe nor probe-loss gradients.

## Frozen claims and boundaries

The proposed contribution is the combination of:

1. an exact additive origin ledger for the observed forward pass;
2. a head- and layer-resolved graph frame derived from actual dynamic
   `A`, `V`, and the matching `W_O` block;
3. exact final-readout attribution to those additive origins; and
4. a label-free conditional energy score over multiple actual graph
   transitions instead of a single normal center.

"Exact" refers to an additive accounting conditioned on the gates observed in
the native forward pass. Attention weights, RMS scales, and nonlinear MLP
activations would change under an intervention. The registers are therefore
not Shapley values and are not claimed to be counterfactual causal effects.

The endogenous register is not "parameter knowledge." It contains native MLP
writes and any small affine or floating-point closure remainder. A high
endogenous contribution can occur in correct or incorrect computation.

## 1. Prediction events

Let `P0` be the first response position in the cached sequence. The response
token

\[
y_t=x_{P0+t}
\]

is predicted at query position

\[
q_t=P0-1+t.
\]

The physical graph is attached to `q_t`; the score and label belong to
`q_t + 1`. Both coordinates are stored. The source set for this event is only
`s <= q_t`, so the current target embedding is never visible to its own score.

## 2. Four additive origins

The origin order is fixed:

\[
\mathcal C=(E,P,R,M),
\]

where:

- `E` is external evidence from passages, documents, or structured records;
- `P` is the remaining prompt: question, task instruction, system text, and
  chat template;
- `R` is a response token embedding already available in the teacher-forced
  prefix;
- `M` is endogenous nonlinear state introduced by model computation.

For every token position `s`, the input embedding `e_s` is initialized as

\[
X^E_{0,s}=e_s\mathbf 1[s\text{ is external evidence}],
\]

\[
X^P_{0,s}=e_s\mathbf 1[s<P0\text{ and is not evidence}],
\]

\[
X^R_{0,s}=e_s\mathbf 1[s\ge P0],
\qquad
X^M_{0,s}=0.
\]

At every decoder boundary,

\[
x_{l,s}=\sum_{c\in\mathcal C}X^c_{l,s}.
\]

This differs from a prompt-versus-response attention ratio. Once evidence has
entered a response position, it remains in the `E` register when that response
position relays it later. Conversely, the lexical response embedding remains
in `R`. Each response token keeps its physical source endpoint; the method
does not merge all response tokens into one root.

## 3. Propagation through observed attention gates

For native RMSNorm input `x`, define the observed scale

\[
\alpha_{l,s}
=
\left(
\operatorname{mean}(x_{l,s}^{2})+\epsilon
\right)^{-1/2}.
\]

The normalized contribution of register `c` is

\[
\widetilde X^c_{l,s}
=
\gamma_l\odot\alpha_{l,s}X^c_{l,s}.
\]

Because `alpha` is the one scalar observed on the complete native state,

\[
\sum_c\widetilde X^c_{l,s}=\operatorname{RMSNorm}_l(x_{l,s}).
\]

For KV head `k`, the registered dynamic value is

\[
V^c_{l,k,s}=W^V_{l,k}\widetilde X^c_{l,s}.
\]

For query head `h` and its actual GQA KV head `g(h)`, the registered local
message is

\[
m^c_{l,q,h,s}
=
A_{l,h,q,s}
W^O_{l,h}
V^c_{l,g(h),s}.
\]

The per-head and total writes are

\[
U^c_{l,q,h}=\sum_{s\le q}m^c_{l,q,h,s},
\qquad
U^c_{l,q}=\sum_hU^c_{l,q,h}.
\]

For bias-free Llama projections, linearity under the observed RMS scale gives

\[
\sum_cV^c_{l,k,s}=V^{native}_{l,k,s}.
\]

The implementation derives `E`, `P`, and `R` values directly and obtains `M`
as the native-value complement. This both preserves the identity under BF16
arithmetic and assigns any projection bias or numerical remainder to the
declared endogenous account. It also captures the native pre-`W_O` head
context and closes the four head contexts to that observed tensor before
forming the head Gram.

All geometric products use FP32 operands. Consequently their sum equals the
FP32 linear expansion of the native head context. A BF16 fused output
projection can still differ slightly from that expansion. The implementation
saves this relative reconstruction error and assigns the layer-boundary
difference to `M`; it does not falsely describe the FP32 per-head expansion as
a bit-exact decomposition of a fused BF16 kernel.

The attention matrix is an observed routing gate. The method does not
decompose how Q/K produced that gate and does not replace attention with a
static `W_O W_V` operator.

## 4. Residual and MLP updates

After attention,

\[
X^{c,mid}_{l,q}=X^c_{l,q}+U^c_{l,q}.
\]

Let the native MLP write be

\[
F_{l,q}=r^{out}_{l,q}-r^{mid}_{l,q}.
\]

The three input-derived registers pass through unchanged at this same-token
update:

\[
X^c_{l+1,q}=X^{c,mid}_{l,q},
\qquad c\in\{E,P,R\}.
\]

The endogenous register receives the native nonlinear write:

\[
X^M_{l+1,q}=X^{M,mid}_{l,q}+F_{l,q}.
\]

Numerically, `M` is closed from the native state:

\[
X^M_{l+1,q}
=
r^{out}_{l,q}
-X^E_{l+1,q}
-X^P_{l+1,q}
-X^R_{l+1,q}.
\]

The relative attention reconstruction and layer closure errors are saved and
tested. A large remainder is an implementation failure, not a signal that may
silently be called endogenous information.

At later layers, all four registers go through the same observed RMS, dynamic
value, attention, and output projection construction. Consequently an MLP
write made at a prompt or response position can later travel across token
edges while retaining `M` origin.

## 5. Dense graph and compact graph frame

The conceptual multiplex DAG has nodes `(layer, token, origin)` and edges
`(layer, query, head, source, origin)`. Every causal source participates.
Materializing every edge as a hidden-size vector would require
`O(L T H S D)` storage, so the implementation evaluates the dense edge field
online and saves the following fixed frame. No top-k edge controls a score.

Let `C=4`, `D` be hidden size, `L` the number of layers, `H` the number of
query heads, and `T` the number of response prediction events.

### 5.1 Final registered node embeddings

The four final residual registers pass through the final observed RMS gate:

\[
Z^c_t
=
\gamma_f\odot\alpha_{f,q_t}X^c_{L,q_t}.
\]

They satisfy

\[
\sum_cZ^c_t=h^{final}_{q_t}.
\]

They are saved without PCA, random projection, or a learned adapter:

```text
node_embedding [T, C, D]
```

### 5.2 Residual Gram across layer boundaries

\[
G^X_{t,l}[c,c']
=
\langle X^c_{l,q_t},X^{c'}_{l,q_t}\rangle.
\]

```text
residual_gram [T, L+1, C, C]
```

This preserves origin magnitudes, alignment, and cancellation through the
ordered residual trajectory.

### 5.3 Per-layer, per-head write Gram

After summing every physical source separately within an origin, but before
heads are combined:

\[
G^U_{t,l,h}[c,c']
=
\langle U^c_{l,q_t,h},U^{c'}_{l,q_t,h}\rangle.
\]

```text
head_write_gram [T, L, H, C, C]
```

Thus the detector sees whether individual heads reinforce or oppose evidence,
prompt, response, and endogenous writes. It never treats a head mean as the
input graph. Residual and head-write Gram tensors are persisted in FP32 so a
large inner product cannot overflow the FP16 range; the other compact graph
blocks use FP16 storage and are restored for FP64 distance accumulation.

### 5.4 Dense endpoint topology

For an origin-specific source message, define route capacity

\[
w^c_{t,l,h,s}=\lVert m^c_{l,q_t,h,s}\rVert_2.
\]

It is computed without a `[source, hidden]` tensor through

\[
(w^c)^2
=
A^2
(V^c)^T
\left((W^O_{l,h})^TW^O_{l,h}\right)
V^c.
\]

For a nonempty row,

\[
p^c_s=\frac{w^c_s}{\sum_uw^c_u}.
\]

The physical source roles are mutually exclusive:

- `prompt`: `s < P0` and `s != q_t`;
- `history`: `P0 <= s < q_t`;
- `self`: `s = q_t`.

The seven topology entries are fixed as

\[
\left(
\log(1+\sum_sw_s),
\log(1+\exp H(p)),
\max_sp_s,
f_{prompt},
f_{history},
f_{self},
c_{head}
\right).
\]

The three fractions are capacity fractions and sum to one. For head
consensus, let

\[
\bar p_s=\frac{1}{H_{active}}\sum_{h\in active}p_{h,s}
\]

and retain each head's Bhattacharyya overlap

\[
c_{head}=\sum_s\sqrt{p_{h,s}\bar p_s}.
\]

```text
route_topology [T, L, H, C, 7]
```

Every dense endpoint contributes before these fixed graph statistics are
formed. Exact source IDs may additionally be retained for visualization, but
they are not substituted for the dense calculation.

### 5.5 MLP relation

For the native MLP write `F` and each post-attention registered residual:

\[
a^c_{t,l}
=
\frac{
\langle F_{l,q_t},X^{c,mid}_{l,q_t}\rangle
}{
\lVert F_{l,q_t}\rVert
\lVert X^{c,mid}_{l,q_t}\rVert+\epsilon
}.
\]

The final entry records relative MLP update size:

\[
a^{scale}_{t,l}
=
\log\left(
1+
\frac{\lVert F_{l,q_t}\rVert}
{\lVert r^{mid}_{l,q_t}\rVert+\epsilon}
\right).
\]

```text
mlp_relation [T, L, C+1]
```

These values describe an observed nonlinear update. They do not establish
that the MLP injected a fact or caused an error.

### 5.6 Exact final margin contribution

For the observed target `y_t`, let the strongest native competing token be

\[
\hat y_t=\arg\max_{v\ne y_t}z_t(v).
\]

The registered contribution to its target-versus-competitor margin is

\[
\mu^c_t
=
W_U[y_t]Z^c_t-W_U[\hat y_t]Z^c_t.
\]

Therefore

\[
\sum_c\mu^c_t
=
z_t(y_t)-z_t(\hat y_t).
\]

```text
margin_contribution [T, C]
```

The complete signed contribution vector enters the graph metric. Native token
surprisal remains a control; the method can claim a route contribution only if
its held-out gain survives that control.

## 6. Full-tensor product metric

A graph frame is

\[
\mathcal G_t=
\{Z_t,G^X_t,G^U_t,T^{route}_t,A^{mlp}_t,\mu_t\}.
\]

The detector does not convert it into a short hand-built feature vector. Each
block retains its complete named axes. For block `j`, its raw distance is the
root-mean-square difference over corresponding tensor entries:

\[
r_j(G,G')
=
\sqrt{
\frac{1}{|G_j|}
\sum_a(G_{j,a}-G'_{j,a})^2
}.
\]

Its scale `s_j` is the median strictly positive `r_j` over label-free
reference frame pairs. The fixed product-space distance is

\[
d(G,G')
=
\frac{1}{6}
\sum_{j=1}^{6}
\frac{r_j(G,G')}{s_j}.
\]

All six blocks receive equal weight. There is no correlation transform,
learned block weight, supervised scaling, or preliminary feature selection. A
layer coordinate is compared only with the same ordered layer coordinate, and
a head only with the same head in that layer. The RMS is the final reduction
after all corresponding tensor entries have been compared.

The conditional detector converts this distance to a product-space kernel

\[
K(G,G')=\exp\left(-d(G,G')/\tau\right),
\]

where `tau` is fixed from label-free reference distances.

## 7. Multimodal conditional transition model

The normal object is a three-frame transition window

\[
W_t=(\mathcal G_{t-2},\mathcal G_{t-1})\rightarrow\mathcal G_t.
\]

The context width is frozen at two preceding graph frames. The first two
answer events are still captured and available for mechanism inspection, but
they have no complete two-frame context and therefore receive no primary
conditional-transition score. Reported `evaluated_tokens` must make this
exclusion explicit.

Reference windows are stratified without labels by:

```text
task type x response-position decile x prompt-length quartile
```

If a finite reference or calibration split has no observation in the exact
cell, the closest populated cell is used under Manhattan distance on the two
declared bin coordinates, with deterministic ties. This is an explicit
finite-sample conditioning rule, not a score-dependent fallback.

Within each stratum, retain `K=8` actual source windows, or all windows if the
stratum contains fewer than eight. Candidates use stable source/position
order. The first candidate is selected first; each subsequent prototype is the
candidate whose minimum distance to the selected set is greatest. This
deterministic farthest-first traversal uses the full product metric. A
prototype is always an observed two-frame context and its observed next graph,
never an averaged feature vector, optimized medoid, or learned centroid. This
permits evidence-anchored focus, answer-local completion, and other normal
modes to coexist.

For a query context `W`, prototype `k` has context `W_k`, next state `Y_k`, and
empirical cluster weight `pi_k`. Context distance is the mean distance across
the two preceding frames; the joint window distance used for prototype
selection and bandwidth is context distance plus next-frame distance. Its
context-conditioned weight is

\[
\omega_k(W)
=
\frac{
\pi_k\exp(-d(W,W_k)/\tau)
}{
\sum_j\pi_j\exp(-d(W,W_j)/\tau)
}.
\]

The cluster weights are the label-free fractions of reference windows assigned
to each prototype. `tau` is the median positive assignment distance in that
stratum. Let `d` be the distance induced by the full-tensor metric. The raw
conditional transition energy is

\[
a_t
=
-\log
\sum_k
\omega_k(W_t)
\exp\left(-d(\mathcal G_t,Y_k)/\tau\right).
\]

Equivalently, this is the difference between the context-only and joint
context-plus-next-state log kernel energies. A token receives low energy if
any context-compatible observed mode explains its next graph; the eight next
states are never averaged into one center. The model does not decide
beforehand that low entropy, response dominance, or a large MLP write is
hallucination.

Prototype sources, calibration sources, and evaluated sources are disjoint.
To score both physical halves, their roles are reversed. Source ordering, not
file hashing or labels, defines any required deterministic subdivision.

Calibration windows are scored with the frozen prototype bank. In the same
task/position/length stratum, each calibration source receives equal total
weight. If `v_i` is the reciprocal of the number of calibration tokens from
the same source in that stratum, the primary score is the weighted empirical
upper-tail rank

\[
S_t
=
\frac{
\sum_{i\in cal(t)}v_i\mathbf 1[a_i\le a_t]
}{
\sum_{i\in cal(t)}v_i
}.
\]

No label determines metric scales, prototypes, score direction, or
calibration.

## 8. Interpretation after scoring

The primary score says only that the route graph made an unusual conditional
transition. Its tensors can then distinguish candidate explanations:

- evidence never entered the residual ledger;
- evidence entered but head writes opposed or cancelled it;
- evidence was relayed through response nodes normally;
- response-origin state became locally self-sustaining;
- endogenous MLP state aligned against an evidence-origin residual;
- final token margin was mainly supported by `R` or `M` rather than `E`/`P`.

None is individually necessary or sufficient for hallucination. Mechanism
names are assigned after inspecting the physical ledger, not encoded as fixed
positive weights in the detector.

## 9. Label boundary

Data reconstruction, register propagation, graph framing, product-metric
scales, prototype selection, conditional energy, and calibration never open
hallucination labels. `evaluate.py` is the only label-opening module. It joins
frozen scores to annotations and reports token AUROC, average precision,
source-cluster bootstrap intervals, paired differences, and task-specific
results.

The observer experiment remains a teacher-forced replay of existing responses.
It does not claim to recover the generator model's original hidden state when
the observer and generator differ.

## 10. Implemented locked controls

The executable report contains four controls whose equations are fixed before
labels are opened:

- independent-frame graph energy, using the same prototypes but removing the
  two-frame context;
- the historical functional `AVW_O` prompt-route-collapse equation;
- the matching raw-attention prompt-route-collapse equation; and
- native token surprisal (`-log p(y_t)`).

The old QA functional-route-collapse result, approximately
`AUROC=0.7337`, is a locked baseline rather than an ingredient in the primary
score. Endpoint rewiring, value shuffling, layer shuffling, and block removal
remain predeclared follow-up ablations. They are not reported by the current
run and are not described as completed evidence.

## 11. Stop gates

The current implementation is a candidate method, not a validated detection
contribution. It is retained only if held-out, source-clustered comparisons
support the following central claims:

1. two-frame conditioning improves on independent-token matching;
2. the registered graph improves on both locked route-collapse controls;
3. the MLP/readout blocks add information beyond confidence rather than merely
   copying it;
4. QA is competitive with the locked `0.7337` route-collapse baseline;
5. correct narrow-focus Summary and Data2txt tokens are not systematically
   assigned high risk;
6. follow-up endpoint and layer-order ablations support the claimed graph
   interpretation before a paper makes those claims.
7. gains repeat beyond a smoke-test subset and their paired source-bootstrap
   intervals exclude zero.

If these gates fail, the additive register capture remains a mechanism-audit
tool. The detector must not be rescued by reading labels, reversing score
direction, selecting favorable heads, adding a supervised combiner, or
renaming a failed statistic.
