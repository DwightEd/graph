# HoloRoute Experiment Plan

## Primary question

Does a neural dual-axis attention event graph learn position-independent normal
routing geometry that separates hallucinated tokens without hallucination
labels?

## Frozen hypotheses

### H1: neural graph utility

The full event graph improves held-out self-supervised completion and
label-posthoc detection over both `event_only` and the mandatory `flat-1024`
baseline.

`flat-1024` receives the exact same `L x H` values for every token pair, the
observation mask and matched metadata, but no depth, relay, query-set or diamond
adjacency. It masks complete layer blocks and uses the same source split,
conditional density and evaluation. The comparison therefore tests graph
organization rather than access to more attention numbers.

### H2: causal-path utility

`full` improves over `no_path`, and real relay paths improve over a matched
middle-token rewire.

### H3: query-coalition utility

`full` improves over `no_query_set`; query prediction has non-zero coverage and
held-out gain.

### H4: typed transport utility

`full` improves over `identity_transport`.

### H5: depth/relay compositional footprint

Holonomy residual has sufficient coverage and provides label-posthoc increment
beyond event reconstruction. Failure removes the holonomy claim but does not
invalidate the graph encoder.

## Mandatory baselines

1. absolute token position;
2. relative token position;
3. direct Lookback-style prompt/response score;
4. `flat-1024`: no-topology MLP on the complete token-pair `L x H` tensor;
5. layer-mean and single-layer views;
6. one-hop token/event GNN without relay paths;
7. HoloRoute structural ablations;
8. the strongest frozen RR spectral residual.

Every neural comparison reports parameter count. A smaller/larger flat hidden
dimension may be used only to match the HoloRoute parameter budget and must be
chosen without hallucination labels.

## Position gate

The former rupture score correlated 0.974 with token position. HoloRoute passes
only when:

- its score beats both position baselines;
- position-conditioned feature residuals remain useful;
- the score-position Spearman correlation is substantially below the former
  rupture baseline and is reported for every task.

## Graph-contribution gate

The graph claim requires both:

\[
\mathcal L_{full}^{val}<\mathcal L_{flat1024}^{val}
\]

and

\[
\mathrm{AUPRC}_{full}>\mathrm{AUPRC}_{flat1024}
\]

under matched source groups, masking semantics and comparable parameter counts.
The AUPRC difference is reported with paired source bootstrap uncertainty.

## Evaluation

- QA, Summary and Data2txt separately;
- at least five source-split seeds;
- same-token post-hoc and shifted next-token evaluation;
- `source_id` cluster bootstrap;
- token-ID and manifest verification;
- mechanism-feature coverage;
- per-layer/event sidecars in later visualization work.

## Stopping rules

- Full does not beat flat-1024: remove or redesign the graph claim.
- No path gain: remove De Bruijn relay propagation.
- No query gain: replace SetMixer with a simpler local encoder.
- No transport gain: use identity transport and drop sheaf-style claims.
- No holonomy increment: retain completion model but remove curvature language.
- Score remains position dominated: stop attention-only work and collect richer
  internal states rather than adding another cumulative statistic.
