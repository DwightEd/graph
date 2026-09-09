# Refined proposal: Counterfactual Last-Crossing Mediation Screen

Status: candidate design; no new 8B result is claimed.

## Diagnosis of the current method

The current `message_dag` directory contains two scientific objects and one
path-partition procedure:

1. source allocation under fixed native reference quantities and an explicit
   SwiGLU allocation rule;
2. a native-reference local Jacobian/JVP seeded by lookback messages;
3. a unique last-crossing partition of the 1/2+ position-hop response into
   signed V-content and K-routing physical edges.

They share implementation and artifact conventions, but they are not the same
estimand. Conservation proves that an accounting is internally complete; it
does not prove semantic relevance, finite causal effect, or hallucination
detection. The observed-versus-runner contrast is also not a truth criterion.

## Dominant method

**Counterfactual Last-Crossing Mediation Screen (CLCMS)** uses aligned minimal pairs that
change exactly one evidence constraint while holding the payload, valid shared
response prefix, candidate contrast, and positional alignment fixed.

For each world, the existing native tangent analysis proposes signed physical
paths. Effects are partitioned by the unique last cross-position edge and retain
layer, head, source, query, and V/K branch. The paired decomposition separates
constraint-switch-selective transport from pair-invariant payload transport:

```text
A_e = (C_e(world +) - C_pi(e)(world -)) / 2
S_e = (C_e(world +) + C_pi(e)(world -)) / 2
```

The graph is a screening instrument, not the causal conclusion. A selected
relay is then exchanged in both directions in the original model:

```text
M = 1/2 * ([Y(- <- +) - Y(- <- -)] - [Y(+ <- -) - Y(+ <- +)])
```

`Y` is oriented to the evidence-consistent candidate in each world. The primary
outcome is evidence-consistent sequence log-odds; free-generation factual
consistency is secondary.

## Formal estimand and intervention contract

Let `W+` and `W-` be an aligned minimal pair that differs in one declared
constraint. A frozen alignment `phi` maps shared-prefix rows and candidate-token
positions; it is created without H/N labels. For each world, the primary outcome
is the teacher-forced sequence contrast

```text
Y_W = sum_t log p(y_consistent,W,t | prefix_W,t)
      - sum_t log p(y_inconsistent,W,t | prefix_W,t).
```

Candidates must both be grammatical in context and must swap their evidence
status between worlds. Edge matching `pi(e)` preserves layer, physical head,
V/K branch, and aligned source/query coordinates. Unmatched edges are reported
as unmatched mass, not silently removed.

The primary finite witness is a declared post-WO physical-head message delta at
one aligned carrier `(layer, head, query)`:

```text
delta_m(W -> W') = m_W(layer, head, query) - m_W'(layer, head, query).
F(W' <- W, alpha) = Y_W'(post_attention_residual + alpha * delta_m) - Y_W'.
```

`alpha` is evaluated on a preregistered dose grid. This intervention tests the
selected carrier message, not a natural indirect effect. Optional branch tests
exchange a declared K-cache or V-cache vector at one matched upstream site; they
are reported separately and never merged into the post-WO witness.

The current implementation can at most establish a bounded value-channel
causal witness. A full mediation claim is reserved for a later implementation
that covers K/Q routing, residual/MLP paths, multiple carriers, and path
interference with calibrated finite effects.

## Why this is more than generic attribution

- The semantic estimand is fixed externally by a minimal constraint change.
- The last-crossing partition counts each cross-position path once and retains
  signed V-content/K-routing identities.
- Paired worlds distinguish constraint-selective transport from common payload
  transport.
- Bidirectional finite exchange is a bounded witness gate rather than a post-hoc picture.

The novelty claim remains conditional on a dedicated literature check. Integrated
gradients, attribution patching, edge attribution, information-flow detectors,
and circuit-tracing graphs already cover substantial neighboring ground.

## Controls and falsification

- direct-only, V-only, and source-blind alternatives;
- equal-norm random head/position and irrelevant-source exchanges;
- lexical, positional, and paraphrase-matched constraint controls;
- source-balanced held-out evaluation by task and generator;
- tangent-versus-finite-effect sign and local-magnitude agreement;
- sign accuracy, rank correlation, and effect curves across intervention doses;
- individual-world hop closure, last-crossing closure, and paired decomposition
  closure before any result is published.
- explicit coverage for no-event, no-matched-edge, and no-finite-response cases;
  these cases count as failures for the corresponding claim and are not dropped.

Path selection and thresholds are frozen on controlled/train cases before any
held-out H/N enrichment analysis. If the paired selective effect is absent, no stronger than controls, unstable
across sources, or contradicted by finite exchange, the mechanism claim is
rejected. The source-allocation pipeline remains useful only as a conditional
diagnostic for deeper ancestry.

## Implementation boundary

The immediate refactor separates lifecycle/artifacts from the two mathematical
engines. A new paired-constraint implementation is not allowed to consume H/N
labels; explicit alignment and candidate contrasts are scientific inputs. Labels
enter only in offline evaluation of whether a preregistered transport break is
enriched in hallucinated cases.
