# Experiment plan

This disposable experiment answers two separate questions.

1. Which token-local mechanisms differ between hallucinated and correct RAGTruth tokens after controlling for response position, prompt/response length, task type, and data source?
2. With identical node mechanisms and scoring protocol, which graph summaries matter: absolute mass, support, within-support weights, position/lag/predecessor summaries, RP descriptors, or RR descriptors?

The execution order is fixed:

1. `screen` train and test attention without opening labels. Each sample is processed on GPU and immediately written as a compact CPU artifact.
2. `evaluate-mechanisms` opens labels only after both artifact splits load. Train-token selection happens before concatenation, univariate orientation is learned only on train, and only the prespecified `global_mean` view receives a source-bootstrap interval. It reports `nuisance_only`, `single:*`, `full`, and `leave_out:*`, plus overlap audits.
3. `build-graph` keeps the same response-node mechanisms for every graph variant and operates on atomic `(target, source, layer/head, RP/RR, value)` traces.
4. `evaluate-graphs` separates representation sufficiency (retrain each variant) from decoder sensitivity (fit exact once and apply it to all variants). Graph diagnostics are `nuisance_only`, `node_only`, `graph_only`, and `full`. The important comparisons are:

   - `exact` vs `no_edges`: edge information incremental to fixed node summaries;
   - `exact` vs `unit_mass`: absolute RP/RR mass;
   - `exact` vs `uniform_on_support`: relative edge weights;
   - `exact` vs `weight_shuffle`: weight-to-endpoint assignment;
   - `exact` vs `source_rewire`: source position, lag, and predecessor summaries jointly;
   - `exact` vs `source_free`: endpoints beyond row marginals;
   - `rp_only` vs `rr_only`: current asymmetric graph-descriptor sufficiency.

This test set has already informed method design, so every report is marked `post_hoc_exploratory`. A paper-level confirmatory claim requires a new source-disjoint holdout after formulas, feature directions, and hyperparameters are frozen.
