# Evidence-conditioned route-state graph

## Research question

The method tests one mechanism, rather than treating every hallucination as the
same failure:

> Does a generated token enter a narrow route state that is sustained by
> response history lacking prompt-evidence ancestry?

A narrow route alone is not anomalous. Extractive QA, a summary of one source
span, and verbalization of one structured field all require narrow routing.
The distinction is whether that focus remains evidence-rooted or becomes an
unrooted autoregressive feedback chain.

The output is a token-level mechanism-risk score. It is neither a proof of
factual error nor a complete causal decomposition of the model.

### Frozen predictions

The method is worth keeping only if the following predictions survive held-out
evaluation; they are not adjusted after reading hallucination labels.

| Prediction | Required comparison | Failure meaning |
|---|---|---|
| Actual writes matter in the inherited collapse audit | equation-locked AVWO collapse beats equation-locked raw-attention collapse | Values and `W_O` add no useful information to that inherited topology |
| Ancestry matters | captured posterior beats locked route collapse | The new graph only renames concentration |
| Narrow focus can be correct | captured posterior stays low on correct tokens whose inherited collapse control is already in its top decile | The method still confuses legitimate focus with hallucination |
| Exact endpoints matter | endpoint/weight controls degrade after independent state fitting | Graph topology is not doing work |
| Capture is temporal | sticky state beats an independent-token state assignment | Persistence is unnecessary |
| MLP is only a diagnostic | MLP amplification repeats on a frozen subset without defining the primary score | Otherwise no claim is made about MLP injection |

## 1. Prediction events and computation nodes

Let `P` be the first response position. Response token `y_t = x[P+t]` is
predicted at

\[
q_t=P-1+t.
\]

The physical graph edge targets query position `q_t`. The score and eventual
label belong to prediction position `q_t+1`. These coordinates must never be
collapsed into one token index.

For decoder layer `l`, nodes distinguish the pre-attention residual, the
post-attention residual, and the layer output:

\[
r^{mid}_{l,q}=r^{in}_{l,q}+u^{attn}_{l,q},
\qquad
r^{out}_{l,q}=r^{mid}_{l,q}+u^{mlp}_{l,q}.
\]

MLP is a same-position nonlinear update. The graph does not invent a
cross-token MLP edge or call it a parameter-knowledge source.

## 2. Exact local attention-write edges

For query head `h`, source `s`, and its GQA KV head `g(h)`, define

\[
m_{l,q,h,s}
=A_{l,h,q,s}\,W^O_{l,h}V_{l,g(h),s}.
\]

With a bias-free Llama output projection,

\[
u^{attn}_{l,q}=\sum_h\sum_{s\le q}m_{l,q,h,s}.
\]

This identity is checked against the tensor returned by the frozen model. It
is a local additive decomposition of the attention residual write. It does not
attribute how Q/K produced `A`, and it is not a final-logit causal effect.

Each edge has two distinct accounts:

\[
w_e=\lVert m_e\rVert_2
\]

for nonnegative route capacity, and

\[
c_e=\frac{\langle m_e,r^{mid}_{l,q}\rangle}
{\lVert r^{mid}_{l,q}\rVert_2^2+\epsilon}
\]

for signed support of the actual post-attention state. Capacity describes
topology; positive and negative support describe construction and
cancellation. They are never substituted for each other.

The frozen model runs in its checkpoint dtype. Derived geometry converts both
operands to FP32 before projection, so FP32 attention arithmetic is never
multiplied by an unconverted BF16 `W_O`.

## 3. Sparse storage without invented paths

The detector consumes the complete dense scalar accounts over every causal
source token and every head. No top-k operation, sparse tail, or graph-storage
setting can change ancestry, contraction, takeover, or the primary score.

The persisted inspection graph is deliberately smaller. Independently for
each `(layer, query, head)`, it stores at most `K=2` exact source endpoints,
chosen by joint capacity and positive-support coverage. Omitted capacity,
support, and net write remain an `unknown` tail. The tail has no token endpoint
and therefore cannot be expanded into a fabricated path. Increasing `K` only
changes inspection resolution and disk use.

Prompt-query rows are used transiently during ancestry propagation. Response
prediction rows and their exact endpoint/head identities are persisted for
inspection and graph controls.

## 4. Evidence units and multi-hop ancestry

The prompt is divided before model replay into external evidence units:

- QA: retrieved passage blocks;
- Summary: document sentences;
- Data2txt: records or field-value units;
- question, instruction, system, and template text: other-prompt context.

At the input boundary, every prompt token starts in its evidence unit or the
other-prompt unit. Response positions start in a response-root unit.

At each additive attention node, positive support from every causal source is
normalized into a route transition. Residual self support carries the current
node ancestry. Processing the causal layer/token DAG in topological order gives
a boundary-hitting distribution `Pi[l,q]`. The formal dense graph has no
omitted tail; `unknown` is reserved for sparse small-graph oracles and the
inspection artifact.

This is an operational route lineage conditioned on the observed forward pass.
Because RMSNorm, attention gates, and MLPs are input-dependent nonlinear
operations, it is not claimed to be a Shapley value or complete causal
provenance.

## 5. Grounded relay versus unrooted feedback

For a strict response-history source `s < q`, its incoming message is divided
by the already-computed ancestry of the source node:

\[
\widetilde c^+_{l,q,h,s}=
\frac{\max(c_{l,q,h,s},0)}
{\max(c^{res}_{l,q},0)+\sum_{j,u}\max(c_{l,q,j,u},0)}.
\]

\[
G_{l,q,h}=\sum_{P\le s<q}\widetilde c^+_{l,q,h,s}\,\Pi_{l-1,s}(E),
\]

\[
U_{l,q,h}=\sum_{P\le s<q}\widetilde c^+_{l,q,h,s}\,\Pi_{l-1,s}(R).
\]

`G` is evidence-rooted response relay. `U` is unrooted response feedback.
Prompt-carried evidence support is `D`: its physical endpoint is in the prompt
and that endpoint has evidence ancestry. It may already include an earlier
prompt-to-prompt relay, so it is not called a purely direct causal effect.
Source `s=q` is predictor self and belongs to neither history term.

This is the central distinction missing from a prompt-versus-response ratio:
a response token may be carrying prompt evidence rather than replacing it.

## 6. Route capacity inherited from the QA finding

For every layer and prediction event, define a head-by-physical-source-token
matrix

\[
M_{h,s}=w_{l,q,h,s}\,\Pi_{l-1,s}(E),\qquad s<q.
\]

This retains the token-level narrowing behind the earlier QA result while also
allowing an evidence-rooted response token to act as a grounded carrier. It is
not collapsed to passage, sentence, or field units, and heads remain separate.
Every active head is normalized before heads are compared:

\[
\widehat M_{h,s}=\frac{M_{h,s}}{\sum_u M_{h,u}},
\qquad
\bar M_s=\frac{1}{|H_{active}|}\sum_{h\in H_{active}}\widehat M_{h,s}.
\]

Three related degrees of freedom are read from these normalized rows:

\[
N_{source}=\exp H(\bar M),
\]

\[
N_{head}=\frac{\operatorname{tr}(\widehat M\widehat M^T)^2}
{\lVert\widehat M\widehat M^T\rVert_F^2+\epsilon},
\]

and `N_anchor`, the effective number of per-head evidence anchors in the recent
causal window. Their log volume is

\[
V=\log N_{source}+\log N_{head}+\log N_{anchor}.
\]

The earlier QA route-collapse result is retained as an equation-locked
control. It preserves the f7344e2 volume equation, lower-volume score
direction, source-equal nuisance WLS, robust residual scale, and position-wise
ECDF, but it is not a numerical reproduction of that run. The old protocol
rotated three nuisance-fit folds, one calibration fold, and one test fold.
Here the physical train/test halves must remain intact. Sorted sources in each
training half use a fixed modulo-four partition (approximately 3:1) for
nuisance fitting and ECDF calibration before the untouched opposite half is
scored. The new state uses the same topology after conditioning response
carriers on their evidence ancestry.

Each term is also normalized by its attainable physical-source/head/window
size at that query.
The resulting absolute deficit is saved only as a capture diagnostic:

\[
C^{raw}_t=1-\frac{1}{L}\sum_l \widetilde V_{l,t}.
\]

The HMM instead uses the equation-locked relative contraction. Source-equal
WLS predicts each layer's log volume from normalized response position,
position squared, and prompt-plus-response length. With robust training scale
`s_l`, the uncalibrated lower-tail score is

\[
R_t=\frac{1}{L}\sum_l
\max\left(\frac{\widehat V_{l,t}-V_{l,t}}{s_l},0\right).
\]

An independent source-disjoint calibration subset maps `R_t` through the
matching response-position-bin ECDF. The primary contraction coordinate is

\[
C_t=\widehat F^{cal}_{b(t)}(R_t).
\]

The takeover coordinate is

\[
T_t=\frac{U_t}{D_t+G_t+U_t+\epsilon}.
\]

This task-internal position/length baseline absorbs systematic narrow focus in
Summary and Data2txt. A locally unusual but legitimate narrow span can still
have high `C` and low `T`; a captured route requires both high calibrated
contraction and high unrooted takeover.

## 7. Label-free temporal states

The observation for each valid response token is only

\[
o_t=(C_t,T_t).
\]

A three-state sticky Gaussian HMM is fitted separately per task, without
hallucination labels:

1. `exploration`: lowest mean contraction;
2. `grounded_focus`: narrow routes with lower takeover;
3. `captured`: narrow routes with the highest takeover.

State identities are fixed by these structural constraints, not by test AUROC.
The transition fit uses one pseudocount for every transition and ten additional
self-transition pseudocounts. This weak, fixed regularizer prevents a short
training sequence from erasing a state; it is negligible relative to the full
task token count and is never tuned on labels.
The primary online score is the filtered posterior

\[
S_t=P(z_t=\text{captured}\mid o_{1:t}).
\]

The first two answer tokens have no strict response history separate from
predictor self and are excluded from history-state fitting and comparison.
Smoothed posteriors may be plotted but are not the online primary score.
The same fitted emissions are also evaluated with token order removed. Learned
self-transition probabilities and expected dwell times are saved with every
fold. Persistence is supported only if the filtered score improves over this
independent-token control on paired held-out tokens; a sticky prior alone is
not treated as evidence.

## 8. What the current intervention audit becomes

The old four-branch deletion experiment is not imported. Its scientific role
is an optional, separate mechanism audit: on a small frozen subset, test
whether tokens assigned high captured posterior are more sensitive to evidence
or history interventions. It never supplies the production detector score.

## 9. Label boundary and evaluation

`data`, `capture`, `messages`, `graph`, `lineage`, `state`, and `detector` do
not open hallucination labels. `evaluate.py` joins frozen scores to labels and
reports task-specific token AUROC, average precision, source-cluster bootstrap
intervals, and the locked confidence/route-collapse controls.

Three required dense graph controls rebuild lineage and fit their own contraction
calibration and HMM before labels are opened:

- one-hop ancestry in place of multi-hop ancestry;
- endpoint rewiring with row/head/role mass preserved;
- weight shuffling with endpoints preserved;

A no-message graph is tested as a structural boundary: it has no identifiable
route state and is not forced into a three-state model. Primary-minus-control
AUROC/AP comparisons use the exact intersection of valid tokens and paired
source-cluster bootstrap intervals. A separate post-hoc audit reports the
source-equal captured posterior on correct tokens whose equation-locked
functional-collapse score is at least the predeclared `0.9`; this is the direct
check that legitimate narrow focus is not automatically called hallucination.

The method is supported only if multi-hop ancestry improves over the old
collapse control, attention-only routes, and the topology controls, especially
on correct narrow-focus Summary and Data2txt tokens.

## 10. Claim boundary

The defensible candidate contribution is:

> head-resolved attention-write route lineage that divides response-history
> messages by their multi-hop evidence ancestry and detects the transition from
> grounded focus to persistent unrooted capture without hallucination labels.

AVWO decomposition, graph dynamic programming, and HMMs are established tools.
No claim of novelty attaches to them individually. No result may claim that
route capture is necessary or sufficient for hallucination, that MLP equals
parameter knowledge, or that an observer replay recovers the generator's
original causal mechanism.

## 11. Relation to prior work

The implementation deliberately treats its ingredients as established unless
the combination changes the scientific question:

- [Attention is Not Only a Weight](https://aclanthology.org/2020.emnlp-main.574/)
  motivates using transformed values rather than raw attention alone.
- [ALTI](https://aclanthology.org/2022.emnlp-main.595/) includes the attention
  block, residual connection, and layer normalization in layer-wise context
  mixing. Therefore neither AVWO accounting nor layer-wise propagation is
  claimed as new here.
- [Information Flow Routes](https://aclanthology.org/2024.emnlp-main.965/)
  extracts attribution routes from a single forward pass. The present method
  must consequently earn its value from ancestry-conditioned response-history
  decomposition and token-level state dynamics, not from merely drawing a
  route graph.
- [(How) Do Language Models Track State?](https://arxiv.org/abs/2503.02854)
  studies exact latent state in permutation-composition tasks. We borrow its
  discipline of defining a state and testing multiple predictions of that
  state; we do not claim that the model contains two literal hallucination
  registers.

The proposed research contribution is therefore conditional: multi-hop
evidence ancestry should distinguish legitimate narrow focus from persistent
unrooted response capture on held-out tasks and topology controls. If it does
not, the graph remains an audit representation rather than a successful
detector.
