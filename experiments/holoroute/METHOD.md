# Method

## 1. Event graph

One node is an exact layer-specific attention event

\[
e=(s,t,l),\qquad x_e=[A_{t,s}^{l,1},\ldots,A_{t,s}^{l,H}].
\]

`graph.py` creates four structural objects:

- `depth_edges`: the same `(source, target)` pair in adjacent layers;
- `relay_edges`: a depth-respecting walk `(u -> s, l-1) -> (s -> t, l)`;
- `queries`: all events entering one `(target, layer)`;
- `diamonds`: two valid depth/relay compositions with common endpoints.

The observation mask remains separate from the values because a missing head is
censored below the attention floor, not measured as zero.

## 2. Encoder

`HeadEncoder` learns one event representation from the complete head profile.
Each `HoloRouteLayer` then performs three operations:

```text
depth context = transport + aggregate(depth predecessors)
relay context = transport + aggregate(causal predecessors)
query context = attend to other events in the same query set
state         = gated fusion(local, depth, relay, query)
```

Depth, prompt-relay and response-relay messages use different low-rank
transport parameters before aggregation.

## 3. Self-supervision

`learning.py` defines four separate graph views. They are not mixed into one
corrupted graph.

- `event_view`: hide complete event head profiles;
- `depth_view`: hide events with a valid depth predecessor;
- `query_view`: leave one event out of each query set;
- `relay_view`: remove relay relations while keeping another predecessor.

The decoder predicts both head support and retained values. Censored heads are
constrained only to remain below the attention floor.

## 4. Detection

`detection.py` produces one residual matrix

\[
R_t\in\mathbb R^6
\]

with columns for event, depth, relay, query, depth-relay disagreement and
holonomy. No prefix sum or CUSUM is used. A train-only reference conditions the
residuals on position, response length, graph size, retained mass, censoring and
feature coverage. The final score is an empirical tail probability of the
conditional residual energy.

## 5. Flat control

`baseline.py` groups the same events into one exact pair tensor

\[
X_{s,t}\in\mathbb R^{L\times H}
\]

and trains a residual MLP with no adjacency. HoloRoute supports a graph claim
only when it improves both held-out completion and label-posthoc detection over
this baseline.
