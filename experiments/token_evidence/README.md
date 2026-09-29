# Unsupervised token evidence

Predict each original token from source-present and source-key-masked contexts, under full causal history and a fresh last-16-token history branch. Prompt tokens and absolute RoPE positions remain fixed. Each branch starts from clean prompt KV. No sentence-score broadcast, neighboring-risk average, correctness classifier or manual candidate list is used.

The primary `reset_cad_tail` uses q = softmax(2 z_source − z_masked), then scores the negative log probability mass of vocabulary items no likelier than the observed token (half mass for ties). The full vocabulary contributes to the score; saved top-eight candidates explain the output but do not define the score. This is a model-relative anomaly, not a calibrated probability of factual error. The contrast follows [Context-aware Decoding](https://aclanthology.org/2024.naacl-short.69/); its detection use and history reset are hypotheses tested here.

All scores freeze before labels are read. Task thresholds use the unlabeled development mixture's 95th percentile, with equal source/answer weight. This does not guarantee 5% normal-token FPR. Existing supervised models are not used. The small historical evaluation subset is exploratory.

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
research_python=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
"$research_python" -m unittest experiments.token_evidence.test_token -v
"$research_python" -m experiments.token_evidence.run --stage prepare --output outputs/token_evidence_new
"$research_python" -m experiments.token_evidence.run --stage capture --output outputs/token_evidence_new
"$research_python" -m experiments.token_evidence.run --stage evaluate --output outputs/token_evidence_new
"$research_python" -m experiments.token_evidence.report --output outputs/token_evidence_new
```

Run from the graph repository. The output must be fresh. No packages or model weights are installed. Capture evaluates every original token; GSM step labels are only used for separate step-level evaluation. See `PLAN_ZH.md` and `RESULTS_ZH.md` for scope, failures and actual results.
