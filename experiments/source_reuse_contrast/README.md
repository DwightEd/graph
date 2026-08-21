# CaSH-GR: Grounding-Sensitive Attention Graph Refinement

This directory now contains two explicitly separated experiments:

- **CaSH v2** (`main.py`, `model.py`, `experiment.py`): masked exact-source
  prediction. It is retained as a negative/reproducibility baseline because its
  endpoint NLL did not separate hallucinated tokens in the first smoke run.
- **CaSH-GR** (`grounding_main.py`, `grounding_model.py`): the active
  grounding-sensitive graph experiment described below.

Neither method is assumed to work. Training and scoring are label-free;
hallucination labels are opened only by their evaluation commands.

## Research question

The active method asks a narrower and better aligned question than exact-source
prediction:

> Which attention edges are needed to preserve the high-dimensional
> source-reuse field and prompt-origin structure, and is a token represented
> mainly by prompt-grounded or response-closed paths?

The method does **not** assume that hallucination is simply low likelihood. It
reports reconstruction, edge fragility, and prompt/response counterfactual
sufficiency as separate frozen scores.

## Graph and targets

A retained attention incidence is

```text
(source token s) --[layer, head, weight]--> (response token t)
```

Prompt and response sources remain distinct. A conservative prompt-provenance
lower bound is propagated in transformer depth:

```text
prompt edge                         origin = 1
response edge s -> t at layer l     origin = provenance[s, l]
unresolved attention                excluded from the lower bound
```

From the unmodified graph, the model receives three label-free targets:

1. `received_support[token, layer, head, topk]` -- the strongest causal
   source-reuse modes;
2. `grounding_field[token, layer, head, 3]` -- direct prompt, grounded response
   relay, and unsupported response mass;
3. `provenance[token, layer]` -- prompt-origin lower-bound trajectory.

These targets keep the high-dimensional fields that earlier scalar mechanism
summaries discarded.

## Self-supervised edge sensitivity

During training, a fraction of retained incidences is masked. A raw graph pass
reconstructs the three targets. For each observed edge, label-free predictive
sensitivity is

```text
S_e = |A_e * d L_self / d A_e|
```

where `L_self` is the received-support + grounding + provenance reconstruction
loss. Sensitivity is detached and passed, together with the original edge
attributes, to a soft edge gate. A second pass reconstructs the same targets on
the refined graph.

This is predictive sensitivity, not a claim that the edge caused the language
model output.

## Neural graph encoder

Each token-source pair is encoded in two stages:

```text
heads within each layer -> learned set aggregation
ordered layers           -> depth GRU
```

The resulting pair embedding is combined with the source birth/reuse state and
its prompt-origin score. Messages are aggregated to the current token through a
learned relation-aware readout. Source reuse memory is an optional branch rather
than the only information path.

## Counterfactual scores

Using the same frozen decoder and target, the model evaluates:

```text
full graph
prompt-origin removed
response-origin removed
small mass-preserving edge perturbation
```

For token `t`:

```text
prompt_gain        = loss(no_prompt)   - loss(full)
response_gain      = loss(no_response) - loss(full)
closure            = response_gain - prompt_gain
fragility          = loss(perturbed) - loss(full)
refinement_gain    = loss(raw) - loss(refined)
state_gain         = loss(no_source_state) - loss(full)
memory_specificity = loss(shuffled_state) - loss(full)
endpoint_specificity = loss(rewired_endpoint) - loss(full)
```

A positive `closure` means the reconstructed routing state depends more on
response-origin than prompt-origin paths. The scores are not fused or direction-
selected with test labels.

## Structural gates

The method is promoted only if all of the following hold:

1. refined reconstruction improves over the raw graph on source-disjoint
   validation data;
2. real endpoints outperform an endpoint-rewired control;
3. the graph embedding retains at least as much diagnostic information as the
   frozen `received_topk.causal` representation;
4. counterfactual scores are stable across mask seeds, tasks, and causal
   positions;
5. any hallucination result survives source-level paired bootstrap intervals.

Failure of these gates is reported as a negative result; it is not repaired by
adding SetWalk or additional hand-written statistics.

## Run

Smoke test:

```bash
ROOT=/path/to/attention_cache \
OUT=experiments/source_reuse_contrast/outputs/grounding_smoke \
TRAIN_LIMIT=30 TEST_LIMIT=5 EPOCHS=2 SCORE_ROUNDS=2 \
DEVICE=cpu \
  bash experiments/source_reuse_contrast/run_grounding.sh
```

Full run:

```bash
ROOT=/path/to/attention_cache DEVICE=cuda \
  bash experiments/source_reuse_contrast/run_grounding.sh
```

Outputs:

```text
train/model.pt
train/training.json
score/scores.npz
score/manifest.json
evaluation/metrics.csv
evaluation/onset_effects.csv
evaluation/coverage.csv
evaluation/evaluation.json
```

`scores.npz` contains the frozen token embeddings and counterfactual scores, and
never contains hallucination labels.
