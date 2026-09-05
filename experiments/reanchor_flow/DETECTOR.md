# Grounded Re-Anchor Failure Detector

## 1. Detection target

The mechanism audit asks whether average signals differ at labelled hallucination onsets. The
detector asks a stricter question for every response token, without reading train labels:

> When the predictor changes route, does external context enter its computation, alter the
> vocabulary candidates, and support the emitted token?

Schema v8 supports this target with head-resolved `A ||W_OV||` routing and an all-sample context
cut. It does **not** contain source-specific signed message vectors, so this detector is not called
graph propagation and does not claim exact fact transport.

## 2. Raw signals

For prediction event `t`, the detector keeps layer/head identity until the final reduction:

- `route_demand`: RMS head route-change JS;
- `evidence_entry_deficit`: the positive loss from each head's rolling evidence-route median,
  weighted by that head's current route change;
- `context_opposition`: negative context-cut target log-probability gain;
- `context_distribution_js`: full-vocabulary distribution change under the context cut;
- `adoption_deficit`: negative target-versus-best-context-candidate margin;
- `context_target_log_rank`: rank of the emitted token under context support;
- `late_evidence_route_loss`: middle/late evidence-route peak lost by the final layer;
- `predictor_reuse` and `emitted_token_anchor`: later use of predictor `q=p-1` and token state `p`.

Every raw signal is oriented so that a larger value means a stronger failure candidate.

## 3. Unlabelled conditional calibration

Calibration uses only train captures. Any train source ID also present in test is removed. A
separate detector is fitted for QA, Summary and Data2txt.

For each signal `x`, `Q(x)` is a source-balanced empirical CDF conditioned on:

- eight response-position bins;
- four train-quantile entropy bins;
- four train-quantile target-log-probability bins.

Each source has equal total CDF mass, independent of its token or sample count. A conditional cell
needs at least 64 finite tokens from three sources; otherwise it uses the task-global
source-balanced CDF. No labels, classifier weights or test statistics select these transforms.

## 4. Registered scores

Let `q_X = Q(X)`. AND is represented by `min`, so every required signal must be anomalous; OR is
represented by `max`, so either registered failure path may fire.

\[
S_{entry}=\min(q_{demand},q_{entry\ deficit},q_{context\ opposition})
\]

\[
S_{adoption}=\min\left(q_{context\ JS},
\max(q_{adoption\ deficit},q_{target\ log\ rank})\right)
\]

The primary score is available from the causal generation prefix:

\[
S_{online}=\max(S_{entry},S_{adoption}).
\]

The secondary score asks whether a failed token later becomes an internal anchor:

\[
S_{override}=\min\left(q_{late\ route\ loss},
\max(q_{predictor\ reuse},q_{token\ anchor}),
\max(q_{context\ opposition},q_{adoption\ deficit})\right),
\]

\[
S_{offline}=\max(S_{online},S_{override}).
\]

`S_offline` is an after-the-fact diagnosis, not an online detector. The fixed alert threshold is
`0.95`; ranking metrics use the continuous scores.

## 5. Leakage protocol and evaluation

The command performs the following order:

1. read schema-v8 train/test captures without labels;
2. remove overlapping train source groups and fit task-specific conditional CDFs;
3. score all selected test tokens and atomically write `detection/token_scores.npz`;
4. freeze the score file digest and bind it to the canonical test manifest/sample/token geometry;
5. only then open test labels and write `detection/detection_report.json`.

The report contains token and span-onset AUROC/AUPRC, the fixed-threshold result, source-cluster
bootstrap intervals, all three failure modes, the offline score, and controls for context JS,
context opposition, route demand, model surprisal and position. The main innovation is supported
only if `online_failure` improves over the single-signal controls on held-out test sources.

## 6. Commands

Run the detector on an existing complete train/test capture:

```bash
python -m experiments.reanchor_flow.run detect \
  --output experiments/reanchor_flow/outputs/reanchor_v8_all
```

The complete `run_all.sh --split all ...` pipeline now captures both splits, freezes and evaluates
the detector, then produces the descriptive mechanism reports.

## 7. Boundary of the claim

The current score establishes a transport/adoption detector over the full external-context mask.
It cannot yet say which exact supporting fact travelled through which source edge. A later capture
schema must store source-specific signed `W_O V` messages and validate selected paths with targeted
cuts before a layered operator graph or graph message-passing score is scientifically warranted.
