# Current research status

## Active hypothesis

A language model's attention computation is treated as a dual-axis routing process:

- the same token-pair relation evolves along Transformer depth;
- information can be relayed through ordered token paths `u -> s -> t`;
- events entering the same target-layer form a query-specific source coalition.

The active hypothesis is not that hallucinations merely have lower prompt attention or higher entropy. It is that hallucinated tokens may violate the normal structural completion laws learned from these multilayer attention events.

## Active graph

One prompt-response sample produces one independent `EventGraph`.

A node is an exact event:

```text
(source token, target token, layer)
```

Its attribute is the complete attention-head profile at that layer plus an observation mask for cache-censored heads.

Relations:

```text
depth:  (s,t,l) -> (s,t,l+1)
relay:  (u,s,l) -> (s,t,l+1)
query:  all (s,t,l) entering the same (t,l)
diamond: two depth/relay compositions connecting the same endpoints
```

## Active model

HoloRoute contains:

- a head-profile encoder;
- relation-specific low-rank transport;
- destination-conditioned depth and relay aggregation;
- leave-one-event-out query-set attention;
- gated fusion of local, depth, relay, and query contexts;
- whole-event, depth, query, relay, and holonomy self-supervision;
- a position-conditioned one-class reference for token scoring.

The detector does not use CUSUM, prefix sums, Markov state tables, prompt-ratio scores, or manually defined lock-in products.

## Strong baseline

Flat-1024 receives the same exact token-pair `layer x head` tensor and observation mask, but no adjacency, query groups, relay paths, or diamonds. It masks complete layer blocks and uses the same data split, calibration, and evaluation protocol.

## Evidence currently available

The attention-structure audit found:

```text
depth transport gain     +0.05663
query-set gain           +0.02642
relay transport gain     +0.00590
exact-path rewire gain   +0.00569
diamond token coverage    99.56%
```

These results justify implementing the graph. They do not yet prove that graph anomalies identify hallucinations.

## Required acceptance gates

A graph contribution is allowed only when all of the following are reported:

1. full HoloRoute improves held-out self-supervised loss over Flat-1024;
2. full HoloRoute improves token AUPRC over Flat-1024 with paired source bootstrap;
3. depth and query each provide incremental value;
4. real relay paths outperform matched rewiring;
5. holonomy improves beyond depth and relay prediction, otherwise the curvature claim is removed;
6. the final score beats absolute and relative position baselines;
7. score-position correlation is reported and remains far below the rejected cumulative rupture detector;
8. same-token and shifted next-token metrics are both reported;
9. QA, Summary, and Data2txt are evaluated separately over multiple seeds.

## Stop rules

```text
full ~= Flat-1024       -> remove the graph contribution claim
no depth increment      -> remove multilayer transport claim
no query increment      -> simplify to local/depth event modeling
no relay/rewire gain    -> remove causal-path branch
no holonomy increment   -> retain graph completion, remove curvature language
position dominates      -> reject the detector regardless of raw AUROC
```

## Next experiment

Run the full QA comparison from the same commit and dataset:

```bash
bash experiments/holoroute/run_qa.sh
```

Then run Flat-1024 with the same sample scope and training budget. Archive the resulting `evaluation.json`, position table, residual table, parameter count, and commit SHA before changing the method.
