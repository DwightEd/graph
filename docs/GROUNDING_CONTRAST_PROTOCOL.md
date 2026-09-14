# P1: prefix-grounding contrast — frozen pilot protocol

2026-09-13. Supersedes the requirement that the next candidate must use a graph, per the user's latest instruction. Does not delete or mark R05/R06 complete.

Question: can the frozen observer distinguish context-supported continuation from a fluent continuation supported only by its own response history? This is a zero-additional-training verifier baseline, not a novel method or a recovered native causal circuit.

Inputs: immutable R04 inputs.jsonl (SHA256 3258b79c7886dc2f6f93ff6c6e9e38c695a773b7dab2ddfc4cba271a3baac38f). All original source text is retained. Initial timing pilot: fixed first source 12328, both responses 17019/17020. Next: 16 development sources / 32 responses; validation is unopened until the decision rule and code are frozen. These are official-train examples, previously present in population measurements, not a pristine official test.

Frozen model: existing Meta-Llama-3.1-8B-Instruct, bf16 SDPA, no fine-tuning, no label-based calibration or feature selection. Baseline is exact R04 pre-token replay. This is an observer of Llama2 answers, not their original generator.

At every response token, construct a verifier input using only the response token prefix (never the future suffix). The current sentence prefix is the target; preceding history resolves references but is explicitly not evidence. A means supported/compatible or not yet a factual assertion; B means an unsupported factual assertion, including a copied value bound to the wrong object, time, action, or condition. Score is z(B)-z(A), without threshold optimization. Track the full-vocabulary mass of A/B so invalid answer-format behavior is visible.

Primary: causal source-verification log odds. Fixed control: history-only verification (source removed; prior assertions accepted only for this deliberately invalid evidence-control arm). Secondary causal contrast: source log odds minus history log odds. Native controls: NLL, entropy, negative top-two margin. Scores are not probabilities of natural hallucination.

Retrospective secondary: current-sentence completion score is assigned to earlier tokens in that sentence, with explicit lookahead token counts. It may not be reported as causal/early detection. Sentence segmentation is punctuation/newline based and label-free. Neither raw attention nor verifier attention is claimed to establish use or necessity.

Eight program-defined canaries cover same number / different stage, wrong object, missing duration, and nonfactual prefix. They are functional diagnostics, not natural accuracy or training examples. No canary-based prompt tuning in this run.

Save every token's verifier label logits, full log normalizer, top token ID, answer-label mass; baseline token logp/entropy/margin; all token IDs; causal prefixes and token endpoints reconstructible from IDs; immutable settings/code/inputs hashes; per-response atomic files and completion manifest. Repeats use fresh directories, no silent overwrites. No natural labels can be loaded by the scoring module. Evaluation joins official labels only after a complete prediction manifest exists, verifies response identity, and reports every token including initial errors, source-balanced/micro AUROC/AP, full stream onset, through-first-error, and per-task results.

Execution: first fixed two responses are a latency/format pilot, not a success claim. Development expansion only if the runner is finite, complete, answer-label mass median >=0.5 and throughput makes the 32-response run <=30 minutes. No GPU retries on OOM; preserve failure and lower only batch size in a separately documented run. If primary development AUROC fails to beat native controls, record a negative result; no validation fishing or sign reversal. If development succeeds, freeze the exact score before running validation, and require both AUROC and AP gains with source-cluster uncertainty before a success claim. Mechanism claims require subsequent bounded interventions on independently selected cases; P1 alone is observational prompted verification.

## Reproducible invocation and environment

SSH alias: szu-gpu. Reuse the graph `.aris/compute/env-spec.json` unchanged (existing environment, canonical spec hash 03909e02); no dependency installation. PyTorch 2.8.0+cu126 / Transformers 4.57.1 / RTX 4090 24 GiB. The launcher sets offline model use and four CPU threads. No screen binary is installed, so launch in a detached subprocess with a durable exclusive-created log.

Run on the remote machine (or use these exact argv with Python subprocess.Popen, stdout/stderr redirected to an exclusive-created log, start_new_session=True):

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_grounding_contrast.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p1_prefix_pilot_20260913_v1
```

The output directory must not exist. For the gated next phase replace `pilot` with `development` and the directory basename with `p1_prefix_development_20260913_v1`. Save protocol and launcher snapshots with each run. A pilot completion is a runtime/format check only; the evaluator refuses natural-label joins for the pilot.
