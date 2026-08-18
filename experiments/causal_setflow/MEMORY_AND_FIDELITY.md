# CASF memory and fidelity contract

## Scope

This document separates the scientific Causal Attention Set-Flow (CASF) model
from its execution schedule.  Memory optimizations are acceptable only when
they leave the source-set definition, hierarchy, self-supervised targets, and
frozen anomaly score unchanged.

## Scientific invariants

The memory-efficient implementation keeps the following model choices intact:

- hidden dimension `64`;
- route and received-memory source-set branches;
- route source bound `32`, memory source bound `16`, and route mass coverage
  `0.98`;
- typed lag, attention-weight, received-support, received-delta, and ancestry
  encoders;
- Set Transformer source-member interaction;
- head identity and head mixer;
- causal depth recurrence used as the exact-source SetWalk ancestry state;
- ordered depth mixer;
- autoregressive token-time encoder;
- element, head, layer, temporal, and variance objectives;
- deterministic masked reconstruction and label-free calibration.

Smoke and full runs use the same architecture.  A smoke run changes only sample
count, epoch count, and the amount of calibration support.

## Removed implementation overhead

The following operations were not part of the scientific model and have been
removed or corrected:

1. **Full response-square materialization.**  The old implementation allocated
   `current`, `cumulative`, `received`, `previous`, and `received_delta` tensors
   with shape `[head, token, token]`.  The current implementation uses an exact
   running `[head, source]` state and a temporary
   `[head, query_chunk, source]` block.
2. **Whole-model checkpointing.**  Recomputing sparse graph materialization in
   backward was unnecessary.  Checkpointing is now applied per neural layer,
   after exact source sets have been materialized.
3. **Full source-state gathers.**  Source ancestry states are gathered inside
   source-set row chunks rather than materializing both full
   `[token, head, member, hidden]` branches at once.
4. **Unconditional empty members.**  The learned empty member is active only for
   genuinely empty source sets.  It no longer perturbs non-empty sets.
5. **Gradient flow into reconstruction targets.**  Element, head, layer, and
   temporal targets are stop-gradient targets.  This prevents the target branch
   from moving toward its own predictor and reduces retained autograd state.
6. **Single giant batch execution.**  Source sets, head mixers, and depth mixers
   are evaluated in independent leading-dimension chunks and concatenated.
7. **FP32-only neural activations.**  CUDA training uses BF16 when supported and
   FP16 otherwise, while sparse attention accumulation and source-set
   materialization remain FP32.

## Exactness of execution chunks

`materialize_query_chunk_size`, `set_row_chunk_size`, and
`mixer_token_chunk_size` are execution controls, not model hyperparameters.

- Query chunking evaluates the same cumulative received-support definition as
  dense materialization, up to floating-point summation roundoff.
- Source-set rows are independent samples for the Set Transformer; splitting
  their leading batch dimension does not change evaluation-time outputs.
- Head and depth mixer token rows are independent in their leading batch
  dimension; token chunking does not change evaluation-time outputs.
- During training, changing chunk boundaries can change the order in which
  dropout random numbers are consumed, but not the objective or dropout
  distribution.

## Why the hierarchy is retained

- The **route set** models the current token's incoming RR attention.
- The **received-memory set** models which historical response sources have
  accumulated support across the causal prefix.
- The **depth recurrence** is the ancestry state gathered by exact response
  source indices at the next Transformer layer.
- The **depth mixer** reconstructs and aggregates the ordered layer trajectory;
  it is not a replacement for the ancestry recurrence.
- The **time encoder** models the local transition and premature-collapse
  dynamics observed around hallucination onset.

These components may be removed only after a controlled scientific ablation,
not as a memory workaround.

## Remaining modelling approximation

CASF is currently a **bounded weighted source-set model**, not a literal encoder
of every retained source identity:

- the current route set keeps at most 32 sources and records omitted mass;
- the received-memory set keeps at most 16 sources;
- route selection targets 98% retained mass when the bound permits it.

The memory refactor does not reduce these bounds.  Before claiming a complete
attention measure, the project must compare this bounded model with an
all-retained ragged-set variant or demonstrate that performance and structural
statistics have saturated as the bounds increase.

## Complexity

For response length `T`, heads `H`, query chunk `C`, source-set size `K`, hidden
width `D`, and Transformer layers `L`:

- source-set materialization peak memory changes from `O(H T^2)` to
  `O(H C T)`;
- source-set neural peak memory is bounded by the configured row chunk rather
  than all `T H` rows;
- per-layer activation checkpointing prevents neural activations from all `L`
  layers being retained simultaneously;
- the exact arithmetic cost of received-support evaluation remains quadratic in
  `T`; the change is a memory schedule, not an approximation.

## Validation

Run the fidelity tests before a smoke experiment:

```bash
python -m unittest tests.test_causal_setflow_memory -v
```

The tests check:

1. query-chunk materialization against the dense received-support definition;
2. evaluation representation invariance to query/set/mixer chunk sizes;
3. finite gradients through internal per-layer checkpointing.

A faithful smoke run must not override `HIDDEN_DIM`, `MAX_ROUTE_SOURCES`, or
`MAX_MEMORY_SOURCES`:

```bash
OUT=experiments/causal_setflow/outputs/v1/smoke_faithful \
TRAIN_LIMIT=64 TEST_LIMIT=5 EPOCHS=1 \
PRECISION=auto ACTIVATION_CHECKPOINTING=1 \
MATERIALIZE_QUERY_CHUNK_SIZE=64 \
SET_ROW_CHUNK_SIZE=4096 \
MIXER_TOKEN_CHUNK_SIZE=512 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_setflow/run.sh
```

If CUDA memory is still insufficient, lower only the three execution chunk
sizes.  The model architecture, source-set bounds, and objectives must remain
unchanged.