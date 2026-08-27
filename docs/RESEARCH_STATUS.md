# Current research status

## Active question

The active experiment asks whether a label-free directed attention-row
hypergraph can learn useful token representations by recovering deliberately
held-out typed source endpoints against role- and lag-matched causal non-edges.

This is narrower than factual grounding or hallucination detection. The
available cache contains attention rows, token boundaries, diagonal mass and
native unresolved sparse-cache mass. It does not contain hidden states,
per-head OV messages, FFN outputs, prompt-query rows or factual evidence labels.

## Latest evidence: the ordered-layout objective failed

The completed 64D ordered-layout run used 149 QA samples, 30,619 tokens, 2,307
positive tokens and prevalence `0.075345`. Its best validation loss was
`1.946106` at epoch 5, but every frozen unsupervised reader was weak:

| Reader | AUROC | AUPRC |
|---|---:|---:|
| PCA-kNN | 0.548162 | 0.083510 |
| Isolation Forest | 0.550779 | 0.082963 |
| Autoencoder | 0.540130 | 0.081410 |
| Deep SVDD | 0.526438 | 0.078837 |
| LOF | 0.518836 | 0.076342 |

Absolute position reached `0.617076 / 0.112859`. The frozen embedding's linear
and MLP readability probes reached only `0.597574 / 0.096996` and
`0.552147 / 0.079619`, compared with first-order GCN results of
`0.7865 / 0.2999` and `0.7785 / 0.2760`. GCN PCA-kNN was
`0.6982 / 0.1617`.

**Decision.** Stop the old all-objectives configuration. It jointly optimized
clean local-row, P/R/U and ordered-layout targets and did not force every scored
positive support out of the student graph. The result rejects that training
objective as a useful representation learner. It does not by itself isolate or
falsify layer order, exact endpoints, or the directed-hypergraph architecture.

## Active implementation

`experiments/directed_route_hypergraph/` remains the only active implementation.
One sample produces one causal typed graph:

```text
node: token
edge: (source, response target, layer, head, retained weight)
clean row: retained + diagonal + native unresolved = 1
```

The new primary objective is forced held-out endpoint recovery:

1. sample positive retained edges from the clean graph;
2. force-remove those exact edges from the student graph;
3. use the existing GroundedRoute matched-negative sampler;
4. match negative sources on prompt/response role and logarithmic lag while
   preserving exact target, layer and head;
5. score positive versus negative sources from the final node latent;
6. optimize a weighted pairwise ranking loss without hallucination labels.

This is the primary mechanism gate. A representation that cannot distinguish a
hidden true endpoint from a role/lag-matched non-edge has not learned evidence
for exact typed topology.

## Corruption semantics

Native missing cache mass and artificial training masks are separate channels:

```text
native unresolved = endpoint absent because sparse cache did not retain it
masked mass        = known retained endpoint deliberately hidden for training
```

The student-only masked channel cannot mutate the graph's native `unresolved`
tensor. A corrupted row is conserved as retained-kept plus diagonal plus native
unresolved plus masked mass. If optional clean P/R/U or ordered-layout teachers
are enabled, they always use the uncorrupted graph. This prevents an artificial
holdout from being interpreted as inherent cache uncertainty.

## Deterministic and variational representations

The deterministic encoder is the mandatory fair baseline. With four 16D route
slots it exports one 64D vector per token.

The VAE is an explicit ablation, not the default explanation for poor results:

```text
latent_mode=vae, export=mean         -> 64D deterministic evaluation embedding
latent_mode=vae, export=mean_logvar  -> 128D deterministic evaluation embedding
```

The decoder may use a reparameterized posterior sample during training;
evaluation uses the posterior mean so repeated export is deterministic. KL
free bits and warmup are optimization controls. Posterior variance measures
latent dispersion under the artificial censoring task and prior; it is not
language-model confidence, factual uncertainty, or a hallucination score.

The evaluator accepts arbitrary embedding dimensions. Its PCA reader may use a
32D internal basis, but this does not constrain the encoder to 32D.

## Relation to Information Flow

The available attention cache can support only an attention-transport proxy,
not the value-aware contribution layout from *Information Flow Reveals When to
Trust Language Models*. That paper requires hidden states, per-head OV paths and
residual attribution, and its complete detector additionally uses a neural
reranker plus correctness-supervised XGBoost.

The old ordered layout transferred only the algebra of multiplying
non-commuting layer transitions; its failed joint-objective run provides no
positive detection evidence. Ordered P/R/U and endpoint-layout objectives now
remain optional auxiliaries with zero default weight. They may support a layer
order claim only if they add value over endpoint-only recovery and beat reverse,
shuffled and last-layer controls.

Permitted terms are `typed endpoint recovery` and, for the old auxiliary,
`layer-ordered attention transport endpoint layout`. Do not call either one
functional contribution, causal information flow, factual grounding, or
trust-before-generation.

## Required experiment matrix

Use the same source split, token rows, training budget, seeds and downstream
readers for:

1. failed ordered-layout checkpoint as the frozen negative reference;
2. deterministic endpoint recovery;
3. deterministic endpoint recovery plus P/R/U auxiliary;
4. deterministic endpoint recovery plus ordered-layout auxiliary;
5. VAE endpoint recovery with `mean` export;
6. VAE endpoint recovery with `mean_logvar` export;
7. first-order GCN;
8. position-only control; a no-message control only after implementing a
   clean-teacher/separate-student view (the generic `no_message` variant is not
   valid for endpoint recovery because it removes the positive teacher edges);
9. real endpoints versus role/lag-matched endpoint rewire and weight shuffle;
10. correct layer order versus reverse, shuffled and last-layer auxiliaries.

Report held-out pair count and forced-mask coverage before downstream metrics.
Report every embedding's actual dimension. Fit all label-free readers only on
the source-disjoint calibration split; use labels only after scores are frozen
and for explicitly named readability diagnostics.

## Acceptance gates

The endpoint-recovery mechanism requires all of the following:

1. every supervised positive edge is absent from the student graph;
2. native unresolved mass remains unchanged by artificial masking;
3. matched negatives preserve role, log-lag, target, layer and head and are
   verified non-edges;
4. final node latents receive gradient from endpoint ranking;
5. deterministic endpoint recovery improves substantially over the failed
   `0.5482 / 0.0835` PCA-kNN reference;
6. real endpoints beat matched rewires under paired source bootstrap;
7. node readability exceeds position readability;
8. GCN is matched or any remaining gap is explained by a preregistered control;
9. at least five seeds and QA/Summary/Data2txt task splits support the result.

A VAE claim has additional gates:

1. deterministic and VAE runs use identical graph/objective/split budgets;
2. evaluation export is deterministic;
3. KL, active dimensions and posterior scale show no collapse or explosion;
4. any `mean_logvar` gain survives controls for token position, response length,
   retained coverage and native unresolved mass;
5. VAE improves frozen readers, not only training ranking loss.

## Stop rules

```text
forced holdout leaks         -> implementation invalid; do not evaluate
real ~= matched non-edge     -> remove exact-endpoint mechanism claim
embedding <= position        -> representation remains shortcut-dominated
deterministic <= old failure -> stop adding posterior capacity
deterministic << GCN         -> retain GCN as method; diagnose objective/inputs
VAE <= deterministic         -> remove variational module
variance tracks coverage     -> treat it as censoring metadata, not uncertainty
ordered ~= reverse           -> remove layer-order contribution claim
no AUPRC gain                -> retain as representation audit, not detector
```

## Next experiment

Run one frozen QA sequence before adding another mechanism:

```text
1. deterministic endpoint-only smoke test and invariant checks
2. deterministic endpoint-only full QA, multiple seeds
3. identical VAE mean / mean_logvar runs
4. optional flow/layout auxiliaries one at a time
5. matched endpoint rewire and position control; then a dedicated
   clean-teacher/no-message-student control
```

Archive checkpoint method/version, full learning config, graph sidecars,
embedding indices, actual embedding dimension, frozen scores, evaluation report,
commit SHA and exact source split for every run. A lower reconstruction,
ranking or KL loss is not a hallucination result.
