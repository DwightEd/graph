# CaSH v2: Causal Source-Reuse Predictability

CaSH is **not assumed to be effective**. It tests a narrower question before any
source-reuse or hypergraph claim is accepted:

> Does the exact causal history of a source improve prediction of which source a
> token uses, beyond current attention marks and the source's birth state?

The previous real-vs-rewired discriminator was rejected after a smoke run
saturated: almost all sigmoid contrast scores were exactly `-1`. The model had
learned an easy corruption task, not a useful token ranking. Version 2 replaces
that objective with masked exact-source prediction.

## Self-supervised task

For every current token and every observed source, the exact source identity is
masked. The model receives the current layer/head/weight marks and must rank the
true source above several strictly matched alternatives:

```text
same RP/RR role
same fine prompt-position or response-lag stratum
same source-use-count bucket
nearest cumulative mass, last-use gap, and history norm
not another source already used by the current token
```

There is no silent fallback to easier negatives. Tokens without a valid matched
alternative remain in artifacts but are excluded from the endpoint loss, and
coverage is reported by position.

The score is raw predictive information:

```text
endpoint_nll       = -log p(true source | matched candidate set)
margin             = true logit - hardest negative logit
shuffled_nll       = endpoint NLL after shuffling memory among matched candidates
```

Embeddings are L2-normalized and scored by fixed-temperature cosine InfoNCE.
There is no sigmoid anomaly score and logits cannot grow without bound.

## Necessary model ladder

The same task is trained with three source-state views:

| Mode | Candidate information | Question |
|---|---|---|
| `current` | coarse current source statistics only | are local/value cues enough? |
| `birth` | current statistics + source birth state | does source origin help? |
| `dynamic` | current statistics + birth + subsequent reuse memory | does reuse history add information? |

A source-reuse claim requires both unlabeled and label-aware gates:

```text
validation NLL(dynamic) < validation NLL(birth) < validation NLL(current)
AUROC/AUPRC(dynamic) > AUROC/AUPRC(current, birth)
AUROC/AUPRC(dynamic real memory) > dynamic shuffled memory
```

The existing strict-causal `received_topk` residual remains the external frozen
baseline and must also be exceeded before CaSH is promoted to the main method.

## Causal source state

Each source has two explicitly separated states:

```text
birth state   fixed after the source is created
reuse state   updated only by later tokens that use the source
```

Before token `t` is scored, both states contain information from tokens `< t`
only. Position control is `log(1+t)` and never uses final response length.

## Validation and early stopping

Training data are split by `source_id` into fit and validation sources. The best
checkpoint is selected by **unlabeled validation endpoint NLL**, not training
loss. Training logs include validation accuracy, margin, candidate count, logit
mean/std, and valid-token coverage.

## Data and label contract

All attention is read through `research_dataset`. Training, validation, and
scoring never open hallucination labels. `evaluate` opens labels only after all
mode-specific scores are frozen.

## Run

Smoke test:

```bash
ROOT=/path/to/attention_cache \
OUT=experiments/source_reuse_contrast/outputs/smoke \
TRAIN_LIMIT=30 TEST_LIMIT=5 EPOCHS=3 SCORE_ROUNDS=2 \
DEVICE=cpu \
  bash experiments/source_reuse_contrast/run.sh
```

Full run:

```bash
ROOT=/path/to/attention_cache DEVICE=cuda \
  bash experiments/source_reuse_contrast/run.sh
```

Default modes are `current,birth,dynamic`. Use `MODES=dynamic` only for an
engineering check, not for a scientific comparison.

Outputs:

```text
<mode>/train/model.pt
<mode>/train/training.json
<mode>/score/scores.npz
<mode>/score/manifest.json
predictability_gate.json
evaluation/metrics.csv
evaluation/paired_deltas.csv
evaluation/coverage.csv
evaluation/onset_effects.csv
```

## Claim boundary

A positive result supports only:

```text
exact source-reuse history improves attention-route prediction and its
prediction error is associated with hallucination
```

It does not prove that the routed value was adopted by the residual stream or
that attention caused the generated token.
