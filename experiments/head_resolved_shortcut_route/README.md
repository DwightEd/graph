# Head-resolved shortcut-route audit

This experiment measures where each response-token prediction receives its
functional support and veto in one frozen, teacher-forced model pass. The
formal method is a per-endpoint, per-layer, per-query-head route table over
native `A V W_O` messages. It is an audit of an observed computation, not a
claim that attention is a complete causal explanation.

The exact definitions, preregistered directions, validity rules, controls,
and stopping criteria are in [METHOD.md](METHOD.md).

## Current implementation status

The mathematical core, suffix recurrence, compact artifact schema, native
model-hook adapter, dataset collector, and label-gated evaluator form one
independent public implementation in this package. The older
`attention_mechanism_audit` package remains intact as a separate four-branch
intervention audit; neither package imports the other.

The evaluator describes preregistered associations. A complete collection is
not itself evidence that the mechanism passed the structural controls or
independent-replication gates in `METHOD.md`, and the saved report states this
claim boundary explicitly. The one-click v1 pipeline does not yet execute the
Section 13 re-forward/rewiring controls, so it is an end-to-end v1 association
pipeline, not a complete causal mechanism validation.

## Causal alignment

For response token at physical position `p`, the prediction event is

```text
query_position      q = p - 1
prediction_position p = q + 1
target_token_id     token_ids[p]
```

Every explicit cross-token route satisfies `source_position < q`. Predictor
self is part of the same-position suffix and never appears as a history edge.

## One physical edge, two independent identities

For endpoint `t`, layer `l`, query head `h`, and source `s`, the physical
message is

```text
m[t,l,h,s] = A[t,l,h,s] W_O[l,h] V[l,s,g(h)]
```

where `g(h)` is the native GQA query-head to KV-head mapping and `W_O[l,h]`
is the matching input block of the native output projection. Head identity is
never averaged before nonlinear or signed statistics.

Each physical edge stores four additive ancestry columns:

| Root | Meaning |
|---|---|
| `E` | declared evidence embedding ancestry |
| `Q` | other prompt/question embedding ancestry |
| `R` | earlier response embedding ancestry |
| `N` | numerical closure remainder only |

It also stores one physical carrier role:

| Carrier | Meaning |
|---|---|
| `evidence_prompt` | source is a declared evidence token |
| `other_prompt` | source is another prompt token |
| `response_history` | source is an earlier response token |

Root and carrier remain orthogonal. In particular, evidence ancestry emitted
by an earlier response token is a grounded relay, not response-born content.
The main named cells are:

| Cell | Root × carrier |
|---|---|
| `D` | `E × evidence_prompt` direct evidence |
| `P_E` | `E × other_prompt` prompt relay |
| `G` | `E × response_history` grounded response relay |
| `B` | `R × response_history` response-born history |
| `Q` | all question-root cells, retained by carrier |
| `I` | predictor-local input/suffix injection |
| `N` | numerical closure only |

Signed target-versus-runner contributions are split into support and veto at
the physical-edge or root-edge atom before any aggregation.

## Three separate mechanism axes

The method reports three two-channel (`support`, `veto`) measurements. It does
not combine them into a learned score or manually engineered feature stack.

1. `carrier_drift` compares where prompt-carried and response-carried physical
   messages arrive across depth. The formal map is `[event, layer, head, 2]`;
   the endpoint summary is only a registered readout.
2. `prompt_source_dispersion` is normalized source entropy within each
   `[event, layer, head, sign]` row, followed by a mass-weighted endpoint
   summary. A narrow prompt route has lower dispersion.
3. `response_born_takeover` is the `R`-root share of resolved `E/Q/R` mass
   within response carriers. High response-carrier mass with low takeover is
   compatible with grounded evidence relay.

Undefined quantities are stored as `NaN` with an explicit false mask. A
quantity is not forced to zero when history, prompt mass, sign mass, or
numerical resolution is insufficient.

The three preregistered label-facing summaries use only support: higher
carrier drift, lower prompt dispersion, and higher response-born takeover.
Veto remains a signed mechanism/conservation audit with no post-hoc direction
choice.

## Sparse route artifact

Dense summaries are computed before compression. Within each
`(event, layer, head)` row, physical edges are ranked by target-independent
native message norm, with source position as the stable tie-break. The saved
prefix is the smallest one reaching `0.95` of row norm mass, capped at 64
edges.

The tail is an anonymous compression statistic, not a fourth carrier or a
fabricated endpoint. Carrier × root support/veto totals, physical signed
totals, `sum(x log x)`, attention, value energy, and message-norm sums/maxima
allow the registered dense quantities and the selection contract to be checked
without assigning invented source identities. Dense/sparse equality is the
same real-valued definition, checked with an explicit FP32 accumulation budget
rather than bitwise equality across different reduction orders.

The NPZ artifact is label-free. Hallucination labels are requested from the
dataset interface only by the final task-specific evaluator. The artifact also records source token IDs, the
declared source evidence mask, the `top_k`/coverage selection contract, the
strongest runner token ID, and target log-probability so carrier identity,
compression, readout, and the confidence control can be audited; none of those
identity fields changes an axis.

For canonical caches, attention data and labels are physically separate. A
legacy monolithic formal `.pt` cache is necessarily deserialized as one payload
by the shared loader, but `retain_embedded_labels=False` prevents its labels
from being retained or exposed to collection and route construction. The
firewall is therefore an algorithmic access boundary, not physical file
separation for that legacy format.

Before labels are requested, evaluation validates every artifact and digest, binds
every event back to the canonical label-free cache token sequence, and writes
`frozen_axes.npz`. Only then is the dataset reopened with labels. Reports keep
the three support axes separate and include independent position, response
length, and observer-target-surprisal controls, per-axis validity coverage,
fixed-stratum group differences, a common-validity sensitivity analysis, and
generator-model strata. Veto remains a raw signed audit and is never promoted
to a detector.

## Code map

- `route_shortcut.py`: physical AVWO atoms, D/G/B projections, one shared
  dense-or-sparse three-axis reducer, and deterministic sparse tails.
- `route_capture.py`: single-batch, one-pass native Llama operator capture.
- `route_suffix.py`: reverse observed-gate recurrence through final RMSNorm,
  symmetric SwiGLU, predictor self-attention, and residual paths.
- `route_pipeline.py`: operator-to-artifact assembly and root/native closure.
- `route_artifact.py`: fixed, numeric-only NPZ schema plus causal-alignment and
  schema validation.
- `collect.py`: label-free dataset traversal, atomic artifact journal, and
  exact resume.
- `evaluate.py`: canonical cache rebinding, frozen axes, label alignment, and
  task-specific association reports.
- `run.py` and `run_all.sh`: the single public full-sequence entry point.
- `tests/test_route_shortcut.py`: independent GQA/AVWO oracle, suffix autograd
  oracle, q-to-q+1, root/carrier counterexamples, exact-tail checks, label
  isolation, and serialization round-trip.

Run collection plus the three separate task evaluations with:

```bash
bash experiments/head_resolved_shortcut_route/run_all.sh \
  --model /path/to/model \
  --cache /path/to/attention_cache \
  --source-info /path/to/source_info.jsonl \
  --output /path/to/output
```

`--limit N` is a smoke subset and produces an explicitly partial, non-formal
report. Run the regression suite with:

```bash
python -m pytest -q experiments/head_resolved_shortcut_route/tests
```

The native hook already produces BF16-closed `CapturedRouteOperators` from one
full-sequence model pass. Root-preserving KV-cache chunking has not yet been
implemented, so long inputs may exceed the observer's memory budget; this v1
entry does not expose a fake chunk option.
