# Design audit

## Scientific question

Before introducing a GNN, hypergraph walk, edge gate, or reconstruction model,
we need to know whether correct and hallucinated response tokens differ in the
structure of their causal attention graphs.

## Why this experiment precedes CaSH-GR

Previous experiments mixed three unknowns:

```text
whether raw attention contains signal
whether graph compression preserves it
whether an unsupervised loss aligns with hallucination
```

The graph structure audit removes the learned bottleneck. It preserves exact
endpoints, layer/head incidence, source hyperedges, and causal paths, and tests
recoverability directly.

## Competing recoverability hypotheses

```text
H_normal:    hallucinated graphs are harder to recover
H_attractor: hallucinated graphs are easier to recover
H_null:      graph recoverability is unrelated to hallucination
```

No score sign is selected before labels are opened.

## What counts as graph structure

A quantity is structural when changing exact endpoint incidence while preserving
local edge values can change it. Examples include source-hyperedge consumer
sets, source-pair co-use, prompt-to-response relay paths, head-source bipartite
components and four-cycles, and masked endpoint recovery from prefix topology.
Prompt mass, response mass, diagonal mass, edge count, and causal position are
retained only as nuisance/baseline variables.

## Failure criteria

The topology line is downgraded when recoverability is indistinguishable after
source-level bootstrap, endpoint metrics do not exceed role/mass baselines,
effects disappear under same-response position/density matching, or channel
recovery is explained entirely by global layer/head frequency.

## Permitted claim

A positive audit supports only:

> Correct and hallucinated tokens differ in specified causal attention-graph
> structures or in the recoverability of masked graph relations.

It does not establish that attention caused the generated content or that MLP
updates were directly observed.
