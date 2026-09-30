> 2026-09-30: Current implementation and status are in [the architecture document](../../docs/ARCHITECTURE.md). Reconstruction graph commands described below are retired; their cached results remain historical evidence. Current commands: `python main.py --help`.

# Independent token backtraces and exposed-case iterations

This experiment did **not** meet the detection objective. It preserves the
negative results and does not change the default detector.

`trace.py` uses a separate actual-token log-probability objective for every
target, backpropagating through native Q/K, RMS and SwiGLU to all original input
embeddings. Roots are embedding-gate sensitivities, not semantic evidence or a
conserved attribution. The model is a Llama-3.1-8B observer of existing answers.

The first readout iteration removes the historical unit broadcast and route
window. For each token, let `a` and `s` be its probabilities without and with
the source. `odds_full` uses `logit(a) - logit(s)` under the complete original
answer history. It combines a source-balanced empirical midrank of this
contrast with the raw route midrank using the historical fixed 3:1 weight.
`token_pair` and `odds_pair` also use a saved variable-unit history reset as
ablations; they never average neighbouring risks. The historical removal of
the source changes prompt positions. FP32-saturated complementary mass is
regularized at float32 epsilon, so this is not exact odds in that region.

The second iteration adds restricted copular polarity and duration interval
constraints with source witnesses. Unknown cases keep the native score;
recognized support/contradiction sets the corresponding token scope to 0/1.
The duration event alignment is heuristic, and clause support is not a proof
that the entire answer is correct. `logic_only` is a selective diagnostic:
unknown means no alarm, not verified correctness. Its fixed threshold is 0.5.
The hybrid inherits the native threshold and has no 5% false-alarm guarantee.

Both designs were informed by exposed examples. No natural-label classifier or
threshold fitting is used, but development is not label-blind. Four of the
eight cases belong to the official test set, which was already exposed in
earlier experiments. The full test is not a fresh holdout.

## Observed results

The pilot has 8 answers, 1487 tokens, and 134 official error-span members.
Historical fixed: 19 TP / 58 FP. Token pair: 23 / 77. Full-history odds:
25 / 111. Adding restricted logic: 42 / 111. Logic alone: 18 / 0 with 116
missed span members. These counts are not a separately defined key-token metric.

All 2700 official test answers / 424408 valid tokens were newly scored using
the cached native scalar measurements. Only the pilot has new root gradients.

| Task | Historical fixed AUROC | Odds AUROC | Odds + logic AUROC |
|---|---:|---:|---:|
| QA | 0.890665 | 0.799441 | 0.799592 |
| Summary | 0.752177 | 0.689958 | 0.690073 |
| Data2txt | 0.758040 | 0.620336 | 0.620343 |

The logic parser recognized only 302 tokens on the full test. Its 4 positive
alarms were all outside official spans. Full results include AP, false alarms,
within-answer AUROC, source-cluster bootstrap intervals, and custom exact
character-union/span-coverage metrics. Any-overlap span recall must not be
confused with complete-span coverage.

## Reproduction

Use the existing `research` Python environment from `.aris/compute/local.md`,
with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`,
`HF_HUB_OFFLINE=1`, and `TOKENIZERS_PARALLELISM=false`. Outputs below already
exist; preserve them and use fresh paths for new runs. The diagnostic currently
uses the recorded trace directories in `diagnose.py`.

```bash
python -m experiments.token_backtrace.trace --output outputs/token_backtrace_20260930_pilot --keys 15604 219
python -m experiments.token_backtrace.trace --output outputs/token_backtrace_20260930_controls --keys 11907 12015 12045 12219 9022 7305
python -m experiments.token_backtrace.validate --trace outputs/token_backtrace_20260930_pilot
python -m experiments.token_backtrace.contrast --output outputs/token_backtrace_contrast_20260930_v1
python -m experiments.token_backtrace.benchmark --stage fit --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.benchmark --stage pilot --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.diagnose --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage pilot --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m experiments.token_backtrace.diagnose --output outputs/token_backtrace_readout_20260930_v2 --logic
python -m experiments.token_backtrace.benchmark --stage score-test --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage score-test --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m experiments.token_backtrace.benchmark --stage evaluate-test --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage evaluate-test --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m pytest -q experiments/token_backtrace/test_readout.py experiments/token_backtrace/test_logic.py
```

`pilot.html` and `pilot_tokens.csv` inspect every original token. The raw
`trace.npz`, numerical checks, counterfactual records, constraint witnesses,
frozen test NPZs, and `test_results.json` retain separate measurement scopes.
The shared handoff is `lys/codex/research/refine-logs/token_iteration_20260930/`.

## Whole-answer graph anomaly pilot, 2026-09-30

The existing `benchmark.py` now supports `graph-score` and `graph-evaluate`.
These reuse all eight answers' independent root gradients and saved final
normalized 4096-dimensional hidden states. There are no new LLM forwards.
Every original token receives a score. This is a cached-observer graph
anomaly experiment, not the full native-path/finite-intervention model.

The primary model is signed, directed conditional ridge reconstruction.
Node targets comprise a fixed seed-17 Gaussian projection of hidden states
to 32 dimensions plus NLL, entropy, negative margin, log root-gradient norm,
and positive/negative source fractions. Source-excluded standardization is
followed by equal weighting of hidden and scalar reconstruction losses.
Predictors include position, lexical format, task, signed history mass,
and separate positive/negative weighted ancestor attributes. The weights
are total root sensitivities; they are used for one-hop feature context,
never recursively multiplied as if they were direct neural edges. Feature
aggregation does not broadcast neighbouring risk scores.

Controls omit endpoint identities: conditional ridge retains signed history
mass covariates, and Isolation Forest uses node attributes only. Further
controls use expected endpoints within log2-lag/repeated-token-ID strata and three randomized
endpoint assignments (seeds 17/29/43). Nulls preserve receiver-stratum signed
mass; randomization also preserves weight multisets, but not every sender's
outgoing degree. They test endpoint identity beyond the retained lag/repetition
structure, not whether all graph structure is useless.

Fitting holds out the entire source. Inner source-held predictions from the
remaining sources calibrate an equal-source empirical percentile. Alarm is
percentile > .95; this is not a calibrated error probability or a guaranteed
normal FPR. Inner fits have six sources and final outer fits seven, so their
score distributions can differ. Only eight historically exposed sources
are available, including just one Summary source; task shift and small-data
effects remain. Natural labels are opened only in `graph-evaluate`.

```bash
python -m experiments.token_backtrace.benchmark --stage graph-score --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m experiments.token_backtrace.benchmark --stage graph-evaluate --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m pytest experiments/token_backtrace/test_readout.py -q
```

Run commands in the existing research environment with four BLAS/OpenMP
threads. Scoring refuses to overwrite its output. `graph.npz` preserves
full history/prompt root responses, attributes and predictor matrices;
`tokens.csv` and `TOKEN_GRAPH.html` expose every token, method score, raw
error, official span membership and top history/prompt roots. The common
79-token alarm budget is a ranking diagnostic separate from frozen alarms.
Shared protocol, execution, results and audit are in the existing
`lys/codex/research/refine-logs/gsm8k_states_20260929/TOKEN_GRAPH_*` files.

The exposed pilot contains 1487 valid tokens (134 official span members):
`graph_ridge` AUROC/AP = 0.516395/0.098217, TP7/FP60; `node_ridge`
0.520921/0.101166, TP8/FP58; historical `base` 0.808717/0.279095,
TP19/FP58. Neither a useful detector nor an endpoint-structure advantage
was established. All graph-vs-node/temporal/shuffled source-bootstrap
intervals include zero. Official span membership is not independently
annotated key-token truth.

One explicitly posthoc ablation preserves raw signed edge mass instead of
normalizing each channel. All five graph/null variants are reported, with
no label-based selection. `graph_mass` AUROC 0.541365 (TP9/FP61), expected
endpoints 0.535317, and randomized endpoints 0.538816/0.540507/0.539338 do
not establish an endpoint advantage either. Original files remain frozen.
The original seven methods' scores reproduce bit-exact after this optional
parameter refactor; see `refactor_verification.json`.

```bash
python -m experiments.token_backtrace.benchmark --stage graph-mass-score --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m experiments.token_backtrace.benchmark --stage graph-mass-evaluate --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
```

Supplemental outputs use `mass_` prefixes in the same directory.
`graph_structure_diagnostics` computes exchangeable edge mass and rank
correlations; its tiny-mass cutoff is descriptive and never used to score.
Only five QA, one Summary and two Data2txt sources are available (four
official train and four official test). Source-held cross-fitting here is
an exploratory protocol, not the official held-out benchmark. The baseline
uses a larger historical calibration set; its comparison is same-token,
not equal-training-budget. No new full-test run or default replacement.
