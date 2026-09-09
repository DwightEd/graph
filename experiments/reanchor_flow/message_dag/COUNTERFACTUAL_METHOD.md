# Counterfactual constraint-routing study

## Question and identification boundary

The original observation is a local-to-remote attention transition during
generation. A plausible interpretation is that the model revisits a prompt
constraint, proposes downstream routes, filters those routes, and integrates
the surviving message. Hallucinated tokens appear more local and show smaller
attention changes.

That observation alone has a selection trap. Comparing transport only at
detected transitions conditions on an event whose incidence can differ between
hallucinated and non-hallucinated tokens. A larger conditional transport score
can therefore coexist with fewer transitions, and treating missing transitions
as zero silently changes the estimand. The observational report now separates:

```text
P(prior lookback event | H/N, source)
E(transport | prior lookback event, H/N, source)
```

It pools within generator source before giving sources equal weight. Missing
conditional transport is never zero-imputed. This diagnoses the observation;
it still does not identify what semantic information was routed.

The counterfactual study addresses the semantic question without using H/N
labels for pair construction, event discovery, graph tracing, or carrier
selection. It is a controlled mechanism screen, not a hallucination detector
and not a natural indirect-effect estimator.

## Execution path

```text
reviewed minimal pair
-> freeze token-aligned worlds and candidate contrast
-> capture both native trajectories
-> detect the union of constraint-linked lookback events
-> trace closed last-crossing V-content/K-routing edges in both worlds
-> split aligned transport into common and constraint-selective components
-> freeze a small carrier set
-> exchange physical post-WO head messages in both directions over a dose grid
-> report the complete selection funnel and finite response
```

The stages have separate modules and durable artifacts. `screen` and
`evaluate` are model-free. Every phase is resumable in the same output
directory, provided the frozen scientific settings and pair identities are
unchanged.

## Stage definitions

### 1. Reviewed minimal pair

`counterfactual_pairs.py` requires one declared constraint substring to be
replaced while all surrounding prompt text stays byte-identical. The tokenizer
must preserve equal length; all changed token coordinates are frozen. Both
worlds share the same teacher-forced response prefix and two fixed candidate
sequences. Supported declared edit types are entity, negation, time, and scope.

Textual minimality and token alignment are checked mechanically. Semantic
equivalence outside the intervention and candidate validity remain human-review
requirements, recorded by `reviewed=true`.

### 2. Constraint-linked lookback union

Each world is scanned independently with the existing label-free event rule.
The current ordinary response carrier is named the Reanchor Query Token (RQT),
and a past ordinary source receiving remote gain is a Reanchor Source Token
(RST). See `REANCHOR_PROFILE.md` for the non-counterfactual morphology audit.
The frozen event set is the union of events in both worlds, restricted to sites
whose strongest remote gain points into the changed constraint coordinates in
at least one world. Taking the union prevents one world from defining which
events are even observable in the other. Rows are ranked by the larger remote
gain across worlds and bounded by `event_row_budget`; zero means all rows.

This step establishes constraint contact, not successful semantic use.

### 3. Closed paired transport

For every frozen event row, the native JVP is propagated in each world. Every
one-or-more-hop response path is partitioned by its unique final cross-position
edge into physical V-content and K-routing contributions. Individual-world
closure is checked before comparison.

For an aligned edge `e`:

```text
selective(e) = (plus(e) - minus(e)) / 2
common(e)    = (plus(e) + minus(e)) / 2
plus(e)      = common(e) + selective(e)
minus(e)     = common(e) - selective(e)
```

Unmatched chunks are reported separately and are not imputed as zero in the
matched decomposition. The screen chooses response carriers by absolute
constraint-selective signed response, with deterministic tie breaking and a
small `carrier_budget`.

### 4. Finite message-exchange witness

The screen only proposes a carrier. For a frozen `(layer, head, response
position)`, `message_exchange.py` captures the head output in both native
worlds, applies the corresponding output-projection block, and obtains the
physical post-WO residual-stream message. Each world then receives a dose
`alpha` of the other world's message difference at that exact layer and
position.

Both exchange directions are scored on full teacher-forced candidate sequence
log-probabilities. Candidate odds are oriented so positive always means the
evidence-consistent candidate. Required checks and reported evidence are:

- dose zero reproduces an ungated native forward within dtype tolerance;
- the local selective screen sign agrees with the finite selective endpoint;
- cross-world exchange weakens evidence-consistent odds in both worlds;
- the full dose curve is retained rather than reducing it to one endpoint.

This is deliberately called a bounded post-WO value-channel witness. It does
not separately intervene on Q/K routing, MLP paths, residual alternatives, or
free-running generation.

## Cost controls

The counterfactual protocol is more expensive than an observational scan, but
the expensive work is bounded and late:

- two reusable native captures per pair;
- at most `2 * event_row_budget` paired cut results, with rows sharing each
  loaded layer within a world;
- at most `carrier_budget` finite witnesses per pair;
- for `D` doses, one witness uses `2` message-capture forwards plus
  `4 + 4D` baseline/intervention forwards. The default `D=3` costs 18 forward
  calls per selected carrier.

Pairs that reach no event, no edge, or no selective response never pay the
finite-witness cost. `prepare`, `capture`, `trace`, `screen`, `witness`, and
`evaluate` can be scheduled separately. The final report records realized
witness forward calls.

## Input and command

One JSON object is supplied per line:

```json
{"pair_id":"entity-001","constraint_kind":"entity","prompt_plus":"Facts: Alice owns red. Question: what does Alice own?","prompt_minus":"Facts: Alice owns blue. Question: what does Alice own?","constraint_plus":"Alice owns red.","constraint_minus":"Alice owns blue.","response_prefix":"Based on the facts, Alice owns","candidate_plus":"red","candidate_minus":"blue","reviewed":true}
```

Run all phases:

```text
python -m experiments.reanchor_flow.message_dag counterfactual \
  --pairs pairs.jsonl --model /path/to/local-llama --output outputs/cf-study
```

For a bounded pilot, set `--event-row-budget 1 --carrier-budget 1 --doses
0,0.5,1`. Use `--phase` to run or resume one stage. Offline reporting is also
available through `python -m experiments.reanchor_flow.message_dag evaluate
OUTPUT`.

## Interpretation gates

No routing-mechanism claim follows from a selected carrier alone. A credible
claim additionally needs paired closure, adequate matched-edge coverage,
local-to-finite sign agreement, bidirectional evidence weakening, robustness
to lexical/position/paraphrase controls, and replication across tasks and
generator sources. No-event, no-edge, and no-response pairs stay in the total
pair denominator. Negative or null outcomes falsify the proposed route for
that scope; they are not discarded.
