# Problem anchor: constraint-preserving response transport

## Bottom line

The research problem is to identify and causally test how an evidence constraint
(entity binding, negation, time, or scope) is carried through prior response
tokens and changes a later evidence-consistent decision. The method must
distinguish faithful reuse of a response carrier from reuse that preserves the
payload while losing or reversing the governing constraint.

## What must be solved

The current code can close an attribution or a local differential accounting
identity, but it does not identify a semantic constraint and does not establish
that a traced path causes a hallucination. A useful method must connect an
externally defined, minimal constraint change to a signed, route-resolved model
effect and then survive a finite intervention test.

## Non-goals

- A general-purpose computation-DAG visualizer.
- Treating every long-range attention shift as a hallucination mechanism.
- Inferring truth from the observed-versus-runner token contrast.
- Claiming that local Jacobian closure is itself causal validation.
- Replacing exact physical-head accounting with a learned GNN or probe.
- Claiming empirical gains before held-out 8B experiments have run.

## Constraints

- Reuse the existing v3 native captures and checkpoint weights where possible.
- Event discovery and operator construction must not consume H/N labels.
- Preserve signed effects, physical heads, V-content/K-routing identity, and
  numerical closure; top-k and head averaging are display-only.
- Separate scientific identity from device, chunking, threading, and rendering.
- Use source-balanced held-out evaluation and controlled minimal pairs.
- Treat disk and memory growth explicitly; never silently reduce coverage.

## Success criterion

On held-out constraint pairs, a preregistered multi-hop relay must show a stable,
directional effect on evidence-consistent sequence log-odds under bidirectional
finite message exchange, exceed direct-only and matched random/irrelevant-source
controls, and agree in sign with the route-resolved local prediction. Failure of
semantic selectivity, closure, intervention agreement, or cross-source stability
falsifies the proposed mechanism claim.
