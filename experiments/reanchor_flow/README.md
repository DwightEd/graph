# Claim Re-Anchor Flow Discovery

This experiment tests one graph-specific observation before defining another
hallucination detector:

> Ordinary generation gradually shifts from prompt sources to response history.
> A new factual claim should interrupt that drift: evidence is reread at the
> claim boundary, the first claim tokens become carriers, and evidence-seeded
> paths remain connected to the claim sink. Hallucination may appear as a
> missed re-anchor.

The same frozen model produces every graph view and every path intervention. A
single teacher-forced forward simultaneously records the functional graph and
its attention-only control; optional path cuts are rerun in that model.

## One global event graph

For response token position `p`, the predictor is `q=p-1`. The route row at `q`
is lifted into the causal edge

```text
source token s -> predicted token p
```

with capacity

\[
C_{s,p}=\operatorname{mean}_{l,h}
A_{l,h,p-1,s}\,\|W_l^{O,[h]}V_{l,g(h),s}\|_2.
\]

This coordinate includes the immediately preceding response token without
creating a query self-loop. The attention control uses the same graph and model
but replaces the capacity by `A`. Each target column is normalized over causal
sources.

For a claim ending at sink `t`, the code computes the FlowTracer-style backward
potential

\[
h(i)=\sum_{k>i}W_{i,k}h(k),\qquad h(t)=1.
\]

Let `B` be the first `anchor_width` claim tokens. A second dynamic program sums
all evidence-to-sink path products whose first visit to `B` occurs before the
sink. This is `evidence_reanchor_flow`. It is genuinely global: a chain
`evidence -> boundary -> later response -> sink` can score highly when the
direct evidence-to-sink edge is zero.

## What is frozen before labels

For every response token, artifacts save functional and attention-only incoming
shares from evidence, other prompt tokens, and response history. For every
label-free sentence-like claim proxy, they save:

- total evidence path mass reaching the claim sink;
- evidence path mass crossing the first claim tokens;
- prior-response-seeded path mass through the same boundary;
- direct evidence-to-sink mass;
- bag-of-edges evidence mass inside the claim;
- the boundary reread pulse;
- a role/lag-preserving endpoint rewire;
- an all-layer graph and a middle-third functional control from the same pass.

The pilot boundary splitter uses punctuation/newlines only. It is not a semantic
claim parser. Controlled atomic-claim annotations can replace it without
changing the graph algorithm.

## Graph necessity audit

On a small label-blind, source-disjoint subset, the strongest evidence-connected
claim in each audited sample receives four real reruns:

1. remove the dominant functional global-flow backbone;
2. remove the attention-only backbone;
3. remove the same number of individually largest-capacity edges without using
   path connectivity;
4. remove source endpoints matched by source role and approximate lag.

Each token edge `s -> p` is mapped back to native attention coordinates
`query=p-1, source=s`. The gate is applied after softmax and before the Value
sum, without renormalization, and every later Transformer operation is rerun.
The graph claim survives only if the functional backbone causes a larger
absolute margin change than the attention, edge-bag, and matched-endpoint
controls.

## Evaluation and visualizations

Hallucination labels are opened only after graph artifacts and path cuts are
saved. The preregistered discovery ranking is

\[
-\operatorname{evidence\_reanchor\_flow},
\]

with separate attention, middle-layer, direct, bag, reread, and rewired
controls. QA, Summary, and Data2txt are reported separately. Figures show:

- the normal evidence-to-history transition;
- claim-aligned evidence reread and history curves;
- claim-level global re-anchor flow versus controls;
- the full source-to-predicted-token DAG and the audited backbone.

This is a mechanism-discovery experiment, not a final detector. Stop the graph
claim if global flow does not outperform direct/bag/rewired controls, or if its
selected backbone is not more influential under real reruns.

## Run

From the repository root:

```bash
bash experiments/reanchor_flow/run_all.sh --smoke
```

Pilot:

```bash
bash experiments/reanchor_flow/run_all.sh \
  --limit 20 \
  --audit-limit 3 \
  --plot-limit 3 \
  --output experiments/reanchor_flow/outputs/pilot
```

A full test-split run omits `--limit`. Outputs are written to:

```text
<output>/results/<Task>/<sample>.npz
<output>/figures/<Task>/<sample>.png
<output>/reports/<task>/report.json
<output>/reports/<task>/claim_aligned_reanchor.png
<output>/run_manifest.json
```

## Tests

```bash
python -m pytest -q experiments/reanchor_flow/tests
python -m compileall -q experiments/reanchor_flow
bash -n experiments/reanchor_flow/run_all.sh
```
