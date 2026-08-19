# Attention Phenomenology Audit

This subproject tests whether cached Transformer attention contains routing
phenomena analogous to the **Detection–Fracture–Breach** account in *The
Phenomenology of Hallucinations*. It does not assume that attention weights are
hidden-state activations, semantic conflict, residual contribution, or output
sensitivity.

The project is an attention-only, token-level mechanism audit. Representation,
reference fitting, and null-model scoring are label-free. Token labels are
opened only by the final `evaluate` command.

## Scientific questions

The audit separates five questions that should not be collapsed into one
reconstruction score.

### H1 — Routing detection

Does the layer-resolved routing state become atypical before or at a
hallucination onset under an unlabeled task- and causal-position-conditioned
reference?

The reference uses train-only robust medians and MADs. It reports separate
fracture, integration, and lock-in atypicality rather than learning a weighted
fusion from hallucination labels.

### H2 — Head-routing fracture

Do heads in the same layer split into separated source-routing coalitions near
onset?

For token `t`, layer `l`, and head `h`, retained attention is represented on a
fixed role simplex:

```text
[prompt-position bins, response log2-lag bins, self, unresolved]
```

Two geometries are kept separate:

- **known-role geometry** renormalizes only observed prompt/RR/self mass and is
  the primary fracture object;
- **full-role geometry** includes unresolved/censored mass and is a censoring
  sensitivity control.

The 32 head vectors form a point cloud under Hellinger distance. The audit
computes exact finite H0 persistence death times through the minimum spanning
tree, together with effective rank, local intrinsic dimension, and unordered
head-set transitions. A fracture claim requires a change in known-role geometry,
not merely a change in unresolved mass.

### H3 — Prompt detection without routing integration

Do some heads remain prompt-grounded while others route through response states
with weak prompt ancestry?

Prompt provenance is propagated through ordered layers. Prompt states are fixed
anchors; response states start with zero provenance. For layer `l`, head `h`,
and response token `t`:

\[
q_{l,h,t}^{\mathrm{known}}
= m^P_{l,h,t}
+ a^{\mathrm{self}}_{l,h,t}g_{l,t}
+ \sum_{j<t} a_{l,h,t,j}g_{l,j}.
\]

Censored mass has unknown endpoints, so the audit keeps lower and upper bounds:

\[
q^- = q^{\mathrm{known}},
\qquad
q^+ = q^{\mathrm{known}} + m^{\mathrm{unresolved}}.
\]

The next-layer attention proxy is the equal-head mean because `V` and `W_O` are
not available. Unsupported RR feedback is computed from off-diagonal response
sources; self attention is not mislabeled as response-history feedback.

### H4 — Fracture-to-lock-in dynamics

Hallucination may reorganize routing at onset and later settle into a stable,
weakly grounded response-history regime. The predicted sequence is:

```text
head-set reorganization
    -> prompt provenance decreases
    -> exact response anchors concentrate
    -> anchor turnover and route velocity decrease
```

The audit therefore distinguishes onset from later span occupancy. It preserves
exact response-source distributions per token and layer and measures their
effective number, top-1 share, recent-history mass, lag, anchor turnover, and
Hellinger velocity.

### H5 — Exact endpoint topology

A topology claim requires more than prompt/RR mass and lag statistics. The score
stage creates an exact-endpoint null that preserves:

- target token, layer, head, edge weight, and diagonal;
- prompt versus response role;
- prompt-position bin or response log-lag bin;
- row edge count and retained row mass.

Only the exact source endpoint is resampled. The artifact records the fraction
of edges actually changed. Evaluation reports paired sample-bootstrap confidence
intervals for the real-minus-rewired AUROC/AUPRC difference. If the interval
contains zero, the result supports routing marginals rather than endpoint
ancestry.

The current null does not preserve exact source in-degree. A fixed-margin null is
appropriate only after this first mechanism gate is passed.

## Data objects

All source attention is read through `research_dataset.py`:

```python
from research_dataset import open_research_dataset

dataset = open_research_dataset(split_root, device="cuda")
sample = dataset[sample_id]
```

No module scans or decodes the underlying PT/NPZ cache directly.

The main internal objects are:

```text
RoutingEdges
    retained off-diagonal endpoints + exact self diagonal

RoutingTensor
    full role simplex
    known-only role simplex
    exact RR source mass

SamplePhenomenology
    [token, layer, feature]
    known/full head geometry
    prompt-provenance bounds
    exact-source dynamics
```

## Module boundaries

```text
config.py
    numerical settings only

hypotheses.py
    feature registry, mechanism families, and pre-registered phase directions

routing.py
    ResearchSample -> retained edges -> role and exact-source tensors

nulls.py
    exact-endpoint rewiring and changed-edge fraction

geometry.py
    Hellinger head point clouds, H0 persistence, effective rank, LID,
    and permutation-invariant set transitions

provenance.py
    ordered-layer prompt-provenance and unsupported-RR bounds

dynamics.py
    exact response-anchor concentration, turnover, and velocity

features.py
    compose all mechanisms into [token, layer, feature]

reference.py
    unlabeled task/causal-position reference and layer-resolved family scores

artifacts.py
    lightweight NPZ/JSON I/O and schema names

experiment.py
    fit and score; never reads labels

evaluation.py
    post-hoc token metrics, standardized onset/lock-in effects,
    and paired real-versus-rewired tests

main.py / run.sh
    CLI and one-command workflow
```

The separation is intentional: data extraction, scientific mechanisms, null
models, reference fitting, and label evaluation must not call into one another
implicitly.

## Outputs

```text
experiments/attention_phenomenology/outputs/full/
  reference.npz
  scores/
    manifest.json
    samples/<sample_id>.npz
    details/<selected_sample_id>.npz
  evaluation/
    evaluation.json
    family_metrics.csv
    family_layer_metrics.csv
    layer_feature_metrics.csv
    onset_layer_effects.csv
    lockin_layer_effects.csv
    onset_phase_curves.csv
```

The main score artifact keeps:

```text
[token, layer, feature]
[token, layer, mechanism family]
[token, mechanism family]
```

Selected detail samples additionally keep the full head-role tensors,
persistence death times, prompt-provenance bounds, exact source mass, and real
and rewired endpoints.

## Run

Smoke test:

```bash
TRAIN_LIMIT=20 TEST_LIMIT=5 DETAIL_SAMPLE_IDS=12471 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/attention_phenomenology/run.sh
```

Full audit:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/attention_phenomenology/run.sh
```

Use a fresh `OUT` when changing the representation or reference configuration.

## Interpretation gates

1. **Fracture:** known-role geometry must change near onset after train
   standardization; the effect cannot be explained only by unresolved mass.
2. **Integration:** lower and upper provenance bounds must jointly support prompt
   decoupling; a wide bound alone is a censoring result.
3. **Lock-in:** later hallucination spans should show low grounding together with
   higher exact-source concentration and lower source/head-set velocity.
4. **Topology:** real endpoint scores must exceed the rewired null with a paired
   source/sample-level confidence interval excluding zero.
5. **Claim boundary:** attention-only evidence remains routing evidence. Testing
   residual activation fracture, logit sensitivity, MLP amplification, or true
   head-output conflict requires additional cached states or contribution data.

The project deliberately does not add Dirichlet spectra, graph reconstruction,
a GNN, persistent-homology classifiers, or learned score fusion before these
mechanism gates are evaluated on the full cache.
