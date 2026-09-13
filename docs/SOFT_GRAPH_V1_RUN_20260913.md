# Soft graph v1: frozen full 36-response natural development batch

Existing research environment, no install or rebuild. Mainline graph detector, not a new cloze-parser experiment. Model structure and unresolved claims: SOFT_GRAPH_METHOD_20260913.md. The six-source roster has already been used by v1/v2/v3 and is development data. This run compares four synchronized predictions on all original words; it does not claim independent generalization or original-generator internal traces.

## Prepared by the author on CPU

Working directory `/share/home/tm902089733300000/a903202310/lys/research/graph`.

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m route_graph.soft_graph_runner --inputs outputs/native_audit_design_20260913/inputs.jsonl --output outputs/soft_graph_v1_20260913 --evaluation-manifest outputs/native_audit_design_20260913/evaluation_manifest.json --stage prepare

The fresh witness does not repeat author preparation. Inputs: 36 responses / 6 train sources / 4733 words; input SHA256 `c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264`. Model/tokenizer bytes, runtime, all executed Python sources and protocol are frozen in settings.json. Settings are published only after a complete verified executed_code snapshot. The documented scheduler performs its own identical prepare validation before any population signal.

CPU checks: 70 scoring/operator/structure/feature/integration tests passed, plus 2 snapshot-publication/resume tests passed. Independent final runner/pipeline verification: 9 passed, no Critical/Required findings. These are engineering checks, not method effectiveness.

## Fresh witness: execute this command exactly once

Working directory `/share/home/tm902089733300000/a903202310/lys/research/graph`:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_soft_graph.py --population-pid 153564 --output outputs/soft_graph_v1_20260913 --journal ../reanchor/runs/soft_graph_v1_20260913.json

Use a persistent exec session and retain its ID. Report the actual session ID, later child/wrapper exit codes, phase counts, actual native forward count and original population restoration evidence. Do not terminate the wrapper, edit source/protocol/doc, improvise another command, retry or load another GPU job. If the command diverges, report the divergence to the parent; do not repair it yourself. Check elapsed work with waits no longer than 60 seconds per call.

The scheduler retains its run lock through CPU preparation and child launch, checks the current population PID/command/frozen metadata, uses Linux pidfd for a graceful pause and holds the population output lock until the graph job exits. It snapshots all previously completed population manifest hashes. In finally it verifies old hashes and resumes the original frozen population, with a 900-second verification deadline and actual PID/progress/held-lock checks. A started launcher alone is not restoration. Do not report overall witness success until graph execution and verified restoration both succeed. The previous v3 journal remains untouched.

## Fixed phases and interpretation

A: Qwen structural pointer extraction; Data2txt literal structure; all response leaves preserved.
B: frozen Llama high-dimensional source/response features, event/role candidate matching, structurally matched controls and all-prefix history candidates. Actual feature forwards counted separately.
C: Qwen fullsource S/C/N/U, local S/C/I/U, and history R/C/Q/T/U finite option probabilities. Fixed-pool controls filtered before D. No free-text source answers or annotation labels.
D: native measurements for all response leaves; fixed maximum 2 source edges and 1 history edge, 80 actual forwards per leaf. Queries cover the original preceding response, six new query-search calls per edge; only layers7/15/23 are measured. Record position effects, exact shams, raw embedding→V→recipient effects, repeat/half strength, two matched controls, within-role route diagnostic, unsearched intervals and skipped edges. Without two eligible controls, only position diagnostics are emitted. No control backfill. Negative effects do not imply correctness.
Merge: full-word scores for qwen_fullsource_no_graph, qwen_matcher_no_native, graph_no_history and full_graph. Unknown contracts scores toward0.5; local not-stated is U, literalNone is U, and only forward R>=0.8 can propagate previous risk. Graph regions retain nonadjacent factual-reuse edges. The strict A/B certificate overlay is separately preserved and not invoked by this target-only batch: no claim of identifying a wrong route or aggregation error solely from observed-target dependence.

No RAGTruth hallucination labels train models or tune thresholds, weights, candidate temperatures, control selection or checkpoint. Invalid structures/unknowns/untested edges stay in denominators. Full scores do not mean all words received reliable evidence; report neutral scores and native coverage separately.

## Read-only monitoring

- Graph progress: outputs/soft_graph_v1_20260913/progress.json
- Child log: ../reanchor/runs/soft_graph_v1_20260913.audit.log
- Scheduler journal: ../reanchor/runs/soft_graph_v1_20260913.json (large manifest list; parse selected fields, never dump whole file)
- Restored population log: ../reanchor/runs/soft_graph_v1_20260913.population.log
- Population progress: ../reanchor/outputs/ragtruth_population_20260912/progress.json

## Evaluation after complete, parent only

From the graph working directory after graph progress status is complete and immutable predictions exist:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.evaluate_soft_graph --run outputs/soft_graph_v1_20260913 --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl --output outputs/soft_graph_v1_20260913/evaluation.json

The evaluator rejects modified snapshots/live source before reading labels. Preserve code while the batch runs. Report all-word source-balanced AUROC/AUPRC and the fixed0.8 threshold precision/recall, neutral scores, candidate/control failures and actual native coverage. Six development sources are insufficient for a strong generalization or unique-lookback-accuracy claim. Follow experiment-audit and result-to-claim after completion; graph improvement must be measured against the identical reader without graph.
