# Unlabelled graph fitting: completed experiment, 2026-09-28

The user requested implementation, exposed-case iteration, and complete testing,
prioritizing effectiveness over novelty. `main.py graph-anomaly` reuses existing
native scalar packs; no additional LLM forward was run.

## Implementation and separation

Fresh label-free transforms produce 123 engineered features from the original
6 context and 11 observation scalars. Isolation Forest is a node-only baseline.
The graph contrast model encodes nodes and mean temporal neighborhoods with
two 123→64→32 MLPs. It distinguishes original pairs from source-excluded,
generator/position-matched negatives. A separately trained shuffled-topology
model controls for the proposed temporal edges. This is CoLA-inspired, not a
faithful CoLA reproduction or a native attention graph.

All parameter fitting is label-free on the original source-disjoint fit split.
`fixed_unsupervised` uses .75 source-pair rank +.25 route-window rank, fixed before
evaluation; its threshold is unlabelled development mixture95. `selected_dev`
uses development labels to select among standalone scores and 45 fixed convex
source/auxiliary mixtures and to set a normal-token95 threshold. This second
variant is explicitly label-assisted selection. Neither trains a truth classifier.

Two development rounds were completed: standalone scores and known fit/dev
cases, then finite source-first mixtures and case re-evaluation. Test cases were
only inspected after every task's test predictions froze. All six generators,
900 responses and 150 sources per task were evaluated: 2700 responses,
450 sources and 424408 valid tokens total. This historical test is exploratory.

## Results: token AUROC

| Method | QA | Summary | Data2txt |
|---|---:|---:|---:|
| Graph contrast | .411077 | .407374 | .399982 |
| Shuffled graph contrast | .529653 | .492940 | .459310 |
| Isolation Forest | .655841 | .637798 | .516654 |
| Fixed unsupervised | .890665 | .752177 | .758040 |
| Development-selected | .890665 | .767623 | .785537 |
| Previous source_refine | .880220 | .763045 | .788506 |
| Previous supervised conditioned | .918765 | .795918 | .828858 |
| Previous supervised HGB | .916670 | .797821 | .830690 |

The chosen methods are fixed_unsupervised for QA, full+.25isolation for Summary,
and pair+.1token_source for Data2txt. Selected AP is .353886/.157658/.162144.
Selected-minus-refine source-cluster300 bootstrap95 intervals are respectively
[.00350,.01742], [-.00136,.01038], [-.00439,-.00151]. These intervals are exploratory
and unadjusted. Graph-minus-shuffled intervals are negative in all tasks.
The graph proxy failed; this does not establish that every graph approach fails.

## Failure cases and alarms

At frozen labelled-development normal95 thresholds, selected normal-answer
any-alarm is 16.89%/46.26%/47.35%. These differ from the fixed unsupervised
mixture-threshold results; the threshold regimes must not be conflated.

The pension claim15604, duration219 and hours/intimate/WiFi9022 still miss.
7305 detects16/16 wrong tokens but alarms8/8 neighboring normal tokens.
12045 detects0/15 and25/53 in its two spans. 12219 detects13/13 but neighboring
normal FPR is50%. Normal control11907 has34/206 false alarms under selected_dev
and0/206 under fixed_unsupervised;12015 has0 for both. These exposed cases are
regression diagnostics, with fit/dev/test clearly separated, not independent
confirmation. Do not replace the previous supervised detector with this run.

## Reproduction and verification

Commands/configuration: `experiments/unsupervised_graph/README.md`.
Raw artifacts: `outputs/unsupervised_graph_20260928/`, including fit/score logs,
models, development choices, all56 test score arrays per task, full metrics,
generator tables, bootstrap intervals and three separate case reports.
Existing41 related tests and8 new invariants passed. An independent same-agent
implementation reconstructed labels from official spans, checked the complete
official roster, all168 new score arrays and156 ranking/alarm metrics, with
zero metric discrepancy. Frozen model SHA checks passed. This was not a fresh
external review and did not independently rerun all bootstrap intervals.

Shared handoff: `codex/research/refine-logs/unsupervised_graph_experiment_20260928/`.
The run is complete; preserve failures and do not restart historical experiments.
