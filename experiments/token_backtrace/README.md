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
