# P7 future fixed128 confirmation — prepared, NOT yet authorized by a freeze

The interface is prepared while P7 development is running; no P7 development
efficacy result or new128 annotation has been read at this preparation.
The original scorer remains unchanged. This document is not evidence of a run.

## Gate before any model scoring

Complete all32 original development answers, then evaluate their full5170 tokens
with development_assess_scoped and obtain an independent bounded integrity review.
The candidate-selection gate specified here is development all-token AUROC>0.8;
that gate is not confirmation or overall goal achievement. Report comparisons
against P6/P5 and uncertainty even if they do not favor P7. Main must read and
resolve material integrity findings before freezing; existence of review JSON
does not by itself mean that an LLM verdict has automatically approved the run.

If retained, main executes next_iteration.freeze_p7_confirmation with
--integrity-record pointing to the preserved independent development review
metadata AND --main-decision pointing to a separate explicit main-agent decision
receipt. The latter must say authorize_confirmation/complete_development_audit,
bind exact development32 IDs,32responses/16sources/5170tokens,final manifest and
evaluation hashes,independent metadata and response hashes,completed independent
recomputation,review_read_by_main=true,empty unresolved_material_findings, and a
nonempty finding_disposition map. The independent verdict remains advisory; it
does not automatically authorize a run. No mere nonempty or pending JSON passes.
It refuses incomplete/wrong32 development, changed scorer, failed gate,
missing metadata and pre-existing freeze/confirmation output. It pins all score,
metric, roster, protocol and parent identities before the new128 run. The original
P6 freeze and results are never changed. The new output is
outputs/P7_CONFIRMATION_FREEZE_20260914.json.

Pre-freeze hardening after independent static review: the builder validates exact
DEV_IDS,16sources,5170actual token IDs,and scoped evaluator/annotation provenance.
It pins the old population root/settings/COMPLETE plus all128 selected parent
manifest hashes (label-free). Both exporter and assessor enforce those identities,
zero-new-forward provenance,and exactly ten expected population score names.
The scorer pins its output to the freeze and records the exact freeze SHA/time.
The assessor requires this binding and exactly primary+ten controls before labels.
The v1 static WARN/CPU receipt are preserved; v2 checks are recorded separately in
outputs/P7_CONFIRMATION_INTERFACE_CHECK_20260914_v2.json. Running development code,
word-score formula,prompts,sampling,numerical metric estimators remain unchanged.

## Fresh document witness command after the freeze

Reuse research@03909e02 (env-spec hash, not Git commit), one idle RTX4090, existing
Qwen3-8B and same precision/budgets/batch4. No install, weight edit, downscaling,
input truncation, rejected answers or retry into an existing directory.
Check output/log/launch record absent, then run exactly ONCE:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_reasoned_confirmation.sh /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p7_reasoned_confirmation_20260914_v1
```

Log: graph/runs/p7_reasoned_confirmation_20260914_v1.log. Capture actual child
returncode and UTC start/end in graph/outputs/P7_CONFIRMATION_LAUNCH_20260914.json
using a detached Python supervisor, not CRLF shell wrappers. Both launcher and
scorer refuse before the freeze. The scorer also pins parent implementation,
input hash,128responses/64sources,official-test split,batch4 and model directory.

All128 responses are mandatory:48QA,48Summary,32Data2txt, from64 distinct sources.
The roster excludesallR04 andspentP6 IDs/exactsource texts, was SHA-selected before
P7 efficacy, and is not globally pristine because old population measurements
exist. No labels or score peeking until all128 have finished and their hashes are
frozen. Failure or OOM must be retained, not repaired by dropping harder samples.
On resume inspect manifest/liveprocess/launch record; do not start a duplicate.

## CPU-only commands after successful full128 scoring

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.reasoned_confirmation_reference_export --inputs outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl --population ../reanchor/outputs/ragtruth_population_20260912 --phase validation --output outputs/p7_population_reference_confirmation_20260914_v1
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.reasoned_confirmation_assess --predictions outputs/p7_reasoned_confirmation_20260914_v1 --reference-predictions outputs/p7_population_reference_confirmation_20260914_v1 --annotations /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl --primary reasoned_source_risk --output outputs/p7_reasoned_confirmation_20260914_v1/evaluation.json
```

The first command only exports10 preexisting observer scores with source/token
identity and parent-file hash checks; no model forward or annotations. The second
requires the frozen method, unchanged annotation-file identity, exact full128
roster alignment and scoring start after freeze, then decodes only wanted128 ID-first
annotation rows. Numerical metrics remain the same full-token AUROC/AP,within-answer,
source-balanced,task metrics and500source-bootstrap, seed20260914. No change of
primary, bootstrap, sign, task subset or thresholds after seeing results.
Independent recomputation is required before an efficacy/goal claim. Even a pass
would establish only this frozen retrospective detector/cohort endpoint, not a
universal hallucination mechanism, online onset warning or graph necessity.
