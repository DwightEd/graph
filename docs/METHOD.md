# Method: factorial constraint-control graph

## 1. Target mechanism

The method tests whether a factual error is associated with loss of source
control, rather than merely with a change in attention distance. At one factual
commitment event, the model is evaluated in two source worlds (`A`, `B`) and
after two generated prefixes (`A`, `B`). The outcome is always the same
predeclared answer-candidate logit margin:

```text
M = logit(answer A) - logit(answer B)
```

This creates a small causal experiment around one event instead of tracing the
entire generated sequence.

## 2. Four graph coordinates

Let `M_sp` be the follow-up margin under source world `s` and prefix `p`.
Together with the two pre-prefix onset margins, the graph weights are:

```text
source_onset = (onset_A - onset_B) / 2

source_followup = (M_AA + M_AB - M_BA - M_BB) / 4
prefix_followup = (M_AA - M_AB + M_BA - M_BB) / 4
interaction = (M_AA - M_AB - M_BA + M_BB) / 4
source_prefix_coupling = |interaction|
```

They form one fixed-role signed graph:

```text
source_constraint ───────────────> onset_choice
        │
        ├────────────────────────> followup_choice
        │                                  ▲
generated_prefix ──────────────────────────┘
        ╲__________________________________▲
          source-prefix interaction hyperedge
```

The absolute interaction is used because its sign changes with arbitrary A/B
naming while its magnitude does not. The other signs retain the declared
answer-margin orientation.

This decomposition corrects a flaw in the retired 2x2 implementation. Its
quantity

```text
(M_AA - M_BB)/2 - (M_AB - M_BA)/2
```

equals twice the prefix main effect; it is not an isolated measure of source
control erosion.

## 3. Event construction

An event is admissible only when all of the following are fixed before label
use:

1. one source passage and one factual relation;
2. two minimally different source worlds with answer A versus answer B;
3. a factual commitment position in the sampled response;
4. two prefixes that differ in the candidate commitment while preserving the
   surrounding syntax as far as possible;
5. answer candidates and their tokenization;
6. six finite margins measured with the same model and decoding context.

The two source worlds identify source influence; the two prefixes identify
continuation pressure; their interaction identifies whether source and prefix
effects combine non-additively. A natural RAGTruth label is not itself a
counterfactual world and cannot fill missing cells of the factorial design.

Events whose intervention changes tokenization, entity identity, or unrelated
facts need a separate validity flag at collection time and must not be silently
mixed into the detector input.

## 4. Label-free anomaly score

For each relation type independently, the fit split estimates a robust center
and scale for every graph edge. The center is the median. Scale uses MAD, then
IQR, then standard deviation with a numerical floor when a coordinate is
constant. For graph `g`,

```text
z_e(g) = (w_e(g) - median_e,relation) / scale_e,relation
score(g) = mean_e z_e(g)^2
```

The score is accompanied by all four signed `z_e` deviations and the dominant
absolute deviation. This is a one-class graph signature, not a supervised classifier. It
assumes the unlabeled reference set is mostly normal; contamination robustness
comes from median/MAD but is not unlimited.

Fit and score sources must be disjoint. Every scored relation must have its own
fit profile. These constraints address two known shortcuts: duplicated source
content and relation/task-specific edge scales.

## 5. How prior differences enter the graph

Normal-versus-hallucination observations influence the **causal variables we
choose to intervene on**—source constraint, generated prefix, onset, and
follow-up—not learned edge weights or graph topology. The schema is frozen
before test labels are read.

Using labeled differences to select graph edges and then reporting test
performance on the same samples would only disguise supervision. If discovery
labels are used to revise the schema, the revision must be frozen and evaluated
on untouched sources and preferably another model.

## 6. Why there is no graph autoencoder

The earlier autoencoder did not separate hallucinations, and next-layer
prediction is a poor target for this question:

- residual streams make adjacent layers strongly dependent even when factual
  control is wrong;
- position and common language-model computation dominate reconstruction;
- a decoder can minimize average error while ignoring the rare commitment edge;
- low reconstruction loss does not identify which constraint changed a logit.

[GraphMAE](https://arxiv.org/abs/2205.10803) similarly motivates masked
attribute reconstruction instead of naive graph-structure reconstruction.
[GOOD-D](https://arxiv.org/abs/2211.04208) learns label-free graph OOD scores
through hierarchical contrastive views, while
[one-class graph transformation learning](https://www.ijcai.org/proceedings/2022/0305.pdf)
combines one-class and transformation objectives. They are appropriate neural
baselines only after the fixed causal graph has enough event/topology diversity.

[CHARM](https://arxiv.org/html/2509.24770v2) demonstrates that attention graphs
can support hallucination classification, but its graph classifier is trained
with labels. Source-domain label training followed by cross-dataset testing is
zero-shot transfer, not unsupervised learning.

If a neural extension becomes justified, the preregistered candidates are:

1. mask one measured causal edge and reconstruct it from the other three;
2. require equivariance under the known A/B world swap;
3. contrast a valid factorial graph with causally invalid cell permutations.

All three preserve event semantics more directly than predicting layer `l+1`
from layer `l`.

## 7. Evaluation and interpretation

Labels live in a separate sidecar and are joined only to frozen scores. Report
AUROC, AUPRC, prevalence, and source-cluster bootstrap intervals. Also report:

- event and source coverage before any filtering;
- results by task, relation, model, onset versus inside-span position;
- each edge's normal/hallucination distribution on the discovery split;
- position, raw margin, entropy, and attention-distance controls;
- the fraction of anomalies dominated by each causal edge.

A useful detector result does not by itself establish the proposed mechanism.
The mechanism claim additionally requires consistent intervention directions:
normal events should retain source influence into follow-up, whereas error
events should show abnormal prefix dominance or source-prefix coupling after
matching relation, position, length, and answer-margin difficulty.

## 8. Scope

The implemented code begins at the validated six-margin event boundary. It does
not yet generate counterfactual source worlds or capture model logits. This is
intentional: those operations require dataset-specific factual alignment and
cannot be safely inferred from RAGTruth labels alone.

## 9. Existing-trace screening experiment

Before paying for factorial recapture, the `audit-score` command uses the
already captured v3 arrays as an observational screen. For layer `l`, head
`h`, response predictor `q`, and source group `g`, define

```text
share(l,h,q,g) = message_mass(l,h,q,g) / message_ordinary_mass(l,h,q)
support(q,g) = sum_lh share(l,h,q,g) * head_margin(l,h,q)
               / sum_lh |head_margin(l,h,q)|
```

`head_margin` is the whole head's signed contribution to the observed emitted
token versus its runner-up. `message_mass` is attention weighted by projected
value norm. Their product allocates the observed head action according to
route strength; it does not recover each source message's signed projection.

The fixed four-edge token graph is:

```text
evidence ---------> emitted-token margin
other prompt -----> emitted-token margin
far history ------> emitted-token margin
local+self -------> emitted-token margin
```

The primary ranking score is history support minus evidence support. It is
evaluated against three predeclared shortcuts: the same route displacement
using attention alone, negative observed margin, and response position. Labels
are joined only after score serialization. Source-cluster bootstrap avoids
treating many correlated tokens from one RAG source as independent samples.

This screen can establish a reusable signal in existing data, but cannot by
itself establish that a constraint was causally transmitted or overwritten.
That claim still requires the factorial interventions in Sections 1--4 or an
exact source-level signed decomposition from the saved Q/K and state files.
