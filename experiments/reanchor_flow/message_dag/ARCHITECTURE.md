# Message-DAG architecture and method decision

Updated: 2026-09-09

## Decision

`message_dag` contains two different estimands. They share captured Llama
operands and an artifact lifecycle, but they are not one generic graph method.

| Method | Mathematical object | Role |
|---|---|---|
| Source allocation | Additive allocation at a fixed native reference, including an explicit SwiGLU allocation rule | Conditional ancestry diagnostic |
| Lookback transport | Native local Jacobian/JVP seeded by a detected response read | Main route-level audit |
| Last-crossing cut | Unique partition of every 1/2+ cross-position path by its final V/K crossing | Conservation-certified explanation of the local response |
| Counterfactual mediation screen | Aligned selective/common last-crossing transport plus a finite post-WO message exchange | Constraint-routing mechanism test |

The allocation engine must not be described as a native Jacobian. The tangent
engine must not inherit its allocation rules. Numerical closure means that an
accounting is complete; it does not establish semantic relevance, finite
causality, or hallucination detection.

## Current problems

1. `run.py` and `event_run.py` combine CLI parsing, scientific identity,
   coverage, planning, model access, scheduling, persistence, label copying,
   evaluation, and rendering. Their `argparse.Namespace` objects are broad,
   implicit interfaces.
2. `DifferentialLayer(LayerOperator)` previously mixed allocation and
   differentiation through `linear_allocation=False`. This made two distinct
   mathematical meanings appear interchangeable.
3. Both output families used an unqualified integer `schema=2`; consumers had
   to infer the artifact kind from paths or incidental settings.
4. Persistence was inconsistent: source allocation used a fixed temporary
   filename, lookback used Unix-only `fcntl`, and the edge recorder mixed path
   mathematics, preview selection, ZIP writing, and publication.
5. Scientific identity mixes model paths and some execution/display settings,
   while checkpoint and capture content identities are not frozen uniformly.
6. Label isolation is incomplete. Lookback event discovery and operators are
   label-free, but paired source-allocation selection intentionally reads H/N
   labels. This must be reported as `labels_used_for_selection=true`, not as an
   entirely label-free build.
7. The current lookback event is an operational attention shift. It does not
   identify an entity binding, negation, temporal condition, or other semantic
   constraint. Observed-versus-runner signs and `opposition` therefore cannot
   by themselves support a hallucination-mechanism claim.
8. Existing real-data detector results do not support the old score route.
   More tracing or a renamed score would not repair the missing semantic and
   causal identification.

## Public execution path

The new common CLI is:

```text
python -m experiments.reanchor_flow.message_dag lookback [lookback options]
python -m experiments.reanchor_flow.message_dag allocate [allocation options]
python -m experiments.reanchor_flow.message_dag counterfactual [counterfactual options]
python -m experiments.reanchor_flow.message_dag evaluate OUTPUT [--bootstrap N]
```

The historical `event_run` and `run` module commands remain available while
their orchestration is migrated. New outputs have an explicit container header:

```json
{
  "artifact_kind": "message_dag_study",
  "container_schema": 1,
  "method_id": "lookback_transport",
  "method_schema": "message-dag/lookback-tangent@2"
}
```

Source allocation uses `method_id=source_allocation` and its own schema URI.
The common offline evaluator refuses ambiguous legacy directories instead of
guessing. Legacy entry points remain the explicit compatibility boundary.

The intended lifecycle is linear:

```text
validate capture and method identity
-> freeze scope and compute plan
-> acquire one output lease
-> derive label-free numerical artifacts
-> certify closure and atomically commit each unit
-> release the lease
-> evaluate labels and render from committed artifacts only
```

Device, chunks, threads, and rendering budgets affect execution, not scientific
identity. Event rules, source partition, targets, readout contrast, constraint
alignment, and allocation rules do affect scientific identity.

## File responsibilities after the first refactor slice

| File | Responsibility |
|---|---|
| `__main__.py` | Unified command adapter only |
| `artifacts.py` | Explicit artifact identity, portable writer lease, unique temporary files, atomic publication |
| `evaluation.py` | Dispatch committed artifacts to the correct offline evaluator without capture/model access |
| `cache.py` | Read and validate v3 capture arrays and source partitions |
| `native_layer.py` | Reconstruct shared native Llama operands at one captured layer |
| `operators.py` | Conditional source-allocation operator only |
| `graph.py` | Source forward tape, target reverse allocation, and balance calculations |
| `events.py` | Label-free lookback event rule and scan |
| `reanchor.py` | Collapse label-free layer/head reads into token-level breadth/focality/agreement morphology |
| `reanchor_report.py` | Source-balanced all-token incidence and anchor-balanced fixed-horizon outcomes |
| `differential.py` | Native RMS/QK/OV/SwiGLU local JVP/VJP only |
| `event_trace.py` | Event-seeded 0/1/2+ hop propagation and batching |
| `transport.py` | Pure last-crossing V/K edge calculation |
| `cut_artifact.py` | Persist same-position suffix readers, stream physical cut edges, certify closure, and atomically commit |
| `selection.py` | Target/sample selection and explicit label-use provenance |
| `event_report.py`, `report.py`, views | Offline scoring, label join, and presentation |
| `counterfactual_pairs.py` | Validate and freeze reviewed, token-aligned semantic interventions |
| `counterfactual_capture.py` | Capture both worlds and freeze/trace the union constraint-linked event plan |
| `paired_transport.py` | Compare closed paired cuts and separate matched common/selective transport |
| `message_exchange.py` | Run bounded bidirectional post-WO carrier exchange and candidate-sequence readout |
| `counterfactual_study.py` | Expose the resumable phase workflow through one small public API |
| `counterfactual_report.py` | Evaluate the complete screen funnel and finite witnesses offline |

`native_layer.NativeLayer` is the only shared algorithm dependency of allocation
and tangent operators. The tangent operator no longer subclasses the allocation
operator.

## Dominant research method

The implemented experimental method is **Counterfactual Last-Crossing Mediation Screen (CLCMS)**. It uses an
aligned minimal pair that changes exactly one evidence constraint while holding
payload, candidate contrast, valid response prefix, and position alignment
fixed. Each world first receives an independently closed, per-head signed
last-crossing decomposition.

For aligned edges `e` and `pi(e)`:

```text
selective(e) = (contribution_plus(e) - contribution_minus(pi(e))) / 2
common(e)    = (contribution_plus(e) + contribution_minus(pi(e))) / 2
```

This separates constraint-switch-selective transport from pair-invariant
payload transport. The graph only proposes a relay. The current claim gate is a
bounded bidirectional post-WO head-message exchange in the original model,
measured on the evidence-consistent sequence log-odds in each world. It is not
called a natural indirect effect. The current finite intervention exchanges the
physical post-WO head message only; separate K/Q/MLP/residual interventions
remain future work. See `COUNTERFACTUAL_METHOD.md` for the executable protocol.

The contribution is the combination of:

- an externally fixed semantic counterfactual rather than an H/N-derived event;
- a unique signed last-crossing partition with physical V/K paths;
- paired separation of constraint and payload transport;
- a precisely declared finite bidirectional witness as a faithfulness gate.

This is a candidate contribution, not a claimed result or established novelty.
Integrated gradients, edge attribution/patching, circuit tracing, attention-flow
detectors, and factual-recall interventions are close prior work and must remain
in the novelty comparison.

## Claim gates

No mechanism claim is allowed unless all of the following hold on held-out,
source-balanced data:

1. individual-world `0 + 1 + 2+ = full` and `last-crossing = 1/2+` closure;
2. paired selective/common decomposition closure;
3. stability under lexical, positional, and paraphrase-matched controls;
4. improvement over direct-only, V-only, source-blind, negative-log-probability,
   and matched random/irrelevant-source baselines;
5. predicted sign agreement with finite bidirectional message exchange;
6. consistent direction across tasks and generator sources.

Failure of semantic selectivity, closure, intervention agreement, or held-out
stability falsifies the mechanism claim. In that case the result remains a
route-resolved attribution audit, not a hallucination mechanism or detector.

## Remaining migration work

- Move planning and orchestration out of `run.py`/`event_run.py` into typed
  study workflows without changing the established CLI behavior.
- Freeze capture/checkpoint/study identities by content rather than path.
- Make labels an evaluator-only dependency; retain paired source allocation as
  an explicitly label-selected diagnostic.
- Split preview/render fields from scientific artifacts where old schemas
  currently mix them.
- Add content hashes for checkpoint/capture identities; the current study
  freezes scientific inputs and tokenized pairs but still identifies the model
  by local path.
- Add preregistered lexical, positional, paraphrase, irrelevant-source, and
  random-carrier controls before interpreting a positive pilot as a mechanism.
