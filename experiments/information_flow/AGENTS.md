# Information-flow experiment contract

Before changing this directory, follow:

```text
experiments/grounded_route/iclr/ENGINEERING_RULES.md
```

Additional rules for this experiment:

- do not call raw attention a functional contribution;
- keep the frozen GCN embedding and flow transport separate in saved artifacts;
- preserve exact layer/head/endpoint structure until transport;
- do not introduce hand-written hallucination features;
- keep all downstream readers node-only;
- compare every new representation with the same GCN rows and the same evaluator;
- value-aware claims require new caches containing OV contributions or residual states.
