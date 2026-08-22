# Graph Structure Audit

This experiment comes **before** a learned hallucination detector. It builds one
causal multiplex attention graph per `prompt + response` sample, then asks:

1. Do correct and hallucinated response tokens occupy different graph motifs?
2. Are masked endpoints and layer-head incidences differently recoverable from
   the causal prefix graph?

No direction is assumed: correct may be more recoverable, hallucination may be
more recoverable as a simple/stable erroneous regime, or there may be no
difference.

## Graph

For prompt length `P`, response length `T`, `L` layers, and `H` heads, one sample
is a directed causal multiplex graph. Nodes are prompt and response tokens. An
incidence is `source s -> response token t` with `(layer, head, weight)` and
`source < target`. Self-attention diagonal is stored separately. Parallel edges
at different layers and heads are not averaged before the audit. Every response
token is scored against its prefix graph `G_{<=t}`; prefix state is updated only
after that token is scored.

## Structural families

### Dynamic source hyperedges

Each source token defines a hyperedge containing earlier response tokens that
used it. The audit measures historical hyperedge size/span, novel-source mass,
source-pair co-use, connected components in the induced source-coalition graph,
and overlap between sources' previous consumer sets.

### Causal path structure

A prompt-bin distribution is propagated through exact response endpoints. The
audit reports prompt reachability, path entropy/effective support, widest prompt
path, path redundancy, expected response hops, two-hop prompt relay, and
response-echo mass. This is an attention-derived path operator, not a claim
about residual-stream causality.

### Layer-head topology

For each target token, every layer induces a bipartite graph between heads and
source tokens. The audit measures adjacent-layer source-distribution change,
support Jaccard, connected components, giant-component fraction, normalized
four-cycle density, and similarity to each source's historical channel profile.

## Recoverability

### Masked endpoint recovery

A fraction of current exact sources is hidden. The true source is ranked among
same-role causal candidates using source-hyperedge popularity, historical co-use
with observed sources, layer-head profile compatibility, and prompt-path profile
compatibility. Outputs include MRR, Hits@1/5, percentile rank, and recovery
error.

### Masked layer-head recovery

For a token-source pair, active layer-head incidences are hidden. Channels are
ranked using the source's historical channel profile, the current token's
remaining channel profile, the global prefix channel profile, and adjacent-layer
continuity for the same head.

## Label protocol and outputs

`extract` never opens hallucination labels. It saves one NPZ graph per sample,
one population token artifact, frozen structural metrics, recovery metrics, and
raw graph arrays. `evaluate` opens labels only after these artifacts are frozen.
It reports raw AUROC/AUPRC, effect sizes, source-level bootstrap intervals,
multiple-testing-adjusted rank tests, same-response matched effects, and explicit
recoverability conclusions. Diagnostic direction is not final detector
performance.

Smoke test:

```bash
ROOT=/path/to/attention_cache OUT=experiments/graph_structure_audit/outputs/smoke \
LIMIT_SAMPLES=20 BOOTSTRAP_REPLICATES=50 \
  bash experiments/graph_structure_audit/run.sh
```

Full audit:

```bash
ROOT=/path/to/attention_cache bash experiments/graph_structure_audit/run.sh
```

Outputs:

```text
<out>/audit/manifest.json
<out>/audit/tokens.npz
<out>/audit/graphs/sample_<hash>.npz
<out>/evaluation/feature_metrics.csv
<out>/evaluation/matched_effects.csv
<out>/evaluation/recoverability_hypotheses.csv
<out>/evaluation/evaluation.json
```

A learned graph model is justified only when endpoint/channel recoverability,
exact source-coalition/path structure, or layer-head topology shows stable signal
beyond position, retained mass, and edge-count controls.
