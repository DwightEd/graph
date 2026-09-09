# Reanchor Query Token profile

## Names and scope

A **Reanchor Query Token (RQT)** is an ordinary generated token at position
`q` whose native attention row contains at least one physical layer/head event
that passes the frozen local-to-remote rule when compared with `q-1`.

A **Reanchor Source Token (RST)** is a past ordinary token that receives a
positive remote attention increment at that event. `peak_source` names the
largest increment for display and cross-head agreement. The full event message,
not only the peak source, is used by the downstream message DAG.

Special tokens are retained in the model input but cannot be an RQT or RST.
Their attention mass is not renormalized into a factual source. The first row
without an adjacent response comparison, the terminal row without a future
target, and rows adjacent to a special query are marked `excluded`, not `none`.

RQT is an operational name. It does not assert that the model consciously
checks a condition. Repetition, copying, entity retrieval, discourse repair,
formatting and long-range syntax can all produce the same structural event.
Constraint confirmation requires either an independently annotated evidence
span or the reviewed counterfactual protocol.

## Frozen event rule

For the current query `q`, both the current and previous attention rows are
renormalized over the same legal ordinary-source set. This prevents a source
from becoming “remote” merely because the window moved by one token. A
layer/head read is active when:

```text
previous local mass >= local_floor
current remote mass - previous remote mass >= gain
remote means q - source > window
```

The event rule never reads H/N labels. The default is `window=10`,
`local_floor=0.5`, and `gain=0.1`.

## Three morphology axes

The profile does not collapse “global” and “far” into one scalar.

1. **Head breadth**: fraction of physical heads active in the dominant layer.
   The dominant layer is the layer with the largest summed positive remote
   increment. `all_layer_head_fraction` and `active_layer_fraction` retain the
   cross-layer extent.
2. **Within-head focality**: for each active head, the largest positive remote
   increment divided by all positive remote increments. `focal_head_fraction`
   records the unweighted share of active heads above the focality threshold;
   this head vote, rather than a few unusually large gains, defines whether the
   morphology is focal. Gain-weighted mean focality and effective source count
   `exp(entropy)` remain continuous diagnostics.
3. **Cross-head source agreement**: the unweighted share of active heads whose
   peak RST is the same absolute source position. This separates a majority of
   heads converging on one source from heads retrieving different distant
   sources. `gain_source_agreement` separately reports magnitude-weighted
   agreement.

Mean gain-weighted distance is reported independently, so “diffuse” does not
automatically mean “farther”. Peak-source overlap with evidence, other prompt,
and prior response is also retained. These are unweighted head-vote fractions,
not total semantic information mass.

The fixed descriptive taxonomy uses 0.5 on each axis:

| Type | Dominant-layer heads | Within-head change | Across-head target |
|---|---|---|---|
| `broad_diffuse` | at least half active | diffuse | unrestricted |
| `broad_convergent` | at least half active | focal | same source dominates |
| `broad_diverse` | at least half active | focal | different sources |
| `sparse_focal` | fewer than half active | focal | specialist heads |
| `sparse_diffuse` | fewer than half active | diffuse | specialist scan |

Continuous axes are primary. The five labels are bins for readable comparisons
and should receive threshold-sensitivity checks before a scientific claim.

## Statistical design

The first estimand is label-free incidence over every eligible ordinary query
with an observable next position:

```text
P(RQT type k at q | eligible ordinary q -> q+1 opportunity, source)
```

It is then decomposed by the next-token label without redefining the event.
Special or unknown next-token targets stay in the overall incidence denominator
but not in either label-conditioned denominator:

```text
P(RQT type k at q | next token is H/N, source)
```

Counts are pooled within each generator source, then source rates receive equal
weight. This is the incidence stage. It prevents a comparison of event-only
tokens from hiding that one outcome class simply has fewer reanchor events.

The second stage is explicitly conditional and descriptive:

```text
E(future hallucination rate or negative log-probability
  | RQT morphology k, horizon, source)
```

Horizons are 1, 2–4 and 5–16 tokens after the RQT. Tokens are first averaged
within each anchor/horizon, anchors within a source, and finally sources. Thus
an early RQT with more observable future tokens does not receive more weight
than a late RQT. Reports retain both anchor and target-token counts. Unknown or
special future targets remain missing rather than becoming normal tokens. Each
reanchor morphology also receives a paired, within-source difference from
`none`; only sources supporting both sides enter that contrast.

This addresses event conditioning, head/token pseudo-replication, unequal
follow-up length, special-token leakage and zero imputation. It does not turn
the association into causality. The counterfactual message-exchange witness is
the separate causal faithfulness gate.

## Artifacts and execution

After a scan/evaluation run:

```text
samples/<split>/<task>/<id>/reanchor_profile.npz
reanchor_summary.json
```

The per-sample profile contains one row per captured query, including excluded
and non-reanchor rows. The summary contains source-balanced incidence,
morphology axes and fixed-horizon outcomes. Labels are joined only while
creating the summary.

Existing scans made before the morphology metrics are upgraded from the saved
Q/K and history during `--phase scan`; their frozen event coordinates must
remain identical. An evaluate-only run without those metrics reports
`not_available_rescan_required` instead of guessing the missing morphology.
