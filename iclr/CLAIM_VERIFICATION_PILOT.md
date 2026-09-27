# Evidence-linked claim verification pilot

This is an exploratory external-reader baseline, inspired by CLATTER, FENICE,
DAE and FactCC. It is not a claim of a new factuality architecture, a causal
mechanism in the original generator, or an improvement until actual results
are evaluated. Existing source-first/refine results remain unchanged.

## Computation

1. Freeze original answers, source text and Llama-3.1 token offsets without using
   labels. The historical 32-answer development roster plus the recent four
   answers form the 36-answer regression set. Select four other official-test
   sources per task by SHA256(`binding42:{source_id}`), then the first two
   generator/ID-sorted answers per source. This gives 24 source-disjoint
   exploratory checks; the old aggregate test has already informed research.
2. A frozen Qwen3-8B reader generates atomic claims, unique exact answer quotes,
   source quotes, and optional source-derived corrections. No natural-label
   training. Greedy nonthinking generation, at most 3072 new tokens.
3. Independently score each original sentence and extracted claim against the
   complete original source. The generated audit or its proposed verdict is
   not evidence. Full answer context is available solely for reference
   resolution, making this an offline detector.
4. Reverse the A/B support-label mapping and average the two properly signed
   logit margins. Positive risk means the reader favors unsupported. The
   margin is not a calibrated probability; the frozen alarm threshold is zero.
5. `direct_sentence` broadcasts sentence scores. `atomic_raw` overwrites only
   tokens overlapping an exact unique claim quote; overlapping claims take the
   maximum risk. All other tokens retain direct-sentence scores. Invalid JSON
   or quotes therefore remain in evaluation through the baseline fallback.
6. `atomic_flat` uses exactly the same extracted claims but presents Data2txt
   evidence as JSON leaf paths with unchanged values/types. Its fallback is
   still the raw-source sentence baseline. This is a representation ablation,
   not a complete independently flattened pipeline. Other tasks are identical
   to `atomic_raw`.

Alternative claims are scored separately and retained for repair diagnostics;
they do not replace the primary detector. Exact source-quote validation is
reported, not taken as evidence of semantic entailment. A valid character
quote does not guarantee complete or faithful claim decomposition.

The native Qwen3 model is loaded directly because the teaching intervention
adapter intentionally supports only Llama/Mistral/Qwen2 layouts. No adapter,
attention or weight modifications are made. SDPA handles the native forward;
the final A/B head rows are evaluated in FP32. BF16 hidden-state numerics still
apply. Model batch size and all raw label-order margins are recorded in commands
and per-answer audit files.

## Run

From the graph directory, with torch 2.8.0 and transformers 4.57.1:

```bash
export PYTHONPATH=.:teaching/state_audit/src
python -m experiments.native_support.claim_verification.run --stage prepare \
  --output outputs/claim_verification_20260927_v1 \
  --dataset /path/to/RAGTruth/dataset \
  --historical-roster ../reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl \
  --model /path/to/Qwen3-8B --tokenizer /path/to/Meta-Llama-3.1-8B-Instruct
python -m experiments.native_support.claim_verification.run --stage score \
  --output outputs/claim_verification_20260927_v1 --group all --batch-size 2
python -m experiments.native_support.claim_verification.isolate \
  --output outputs/claim_verification_20260927_v1 --batch-size 2
python -m experiments.native_support.claim_verification.blind \
  --output outputs/claim_verification_20260927_v1 --batch-size 2
python -m experiments.native_support.claim_verification.run --stage evaluate \
  --output outputs/claim_verification_20260927_v1 --group all
```

Scoring resumes completed per-answer files without overwriting them. Use a new
output directory for changed prompts/model/scoring; no cross-version resume is
supported. `--ids` permits an engineering sanity run but is forbidden for
evaluation. Evaluation requires `--group all` and every score in the complete frozen roster,
then reads official labels. AUROC/AP, pair-weighted within-answer AUROC,
zero-threshold precision/recall/FPR, span coverage, correct-answer false alarms,
and extraction failures are saved. Baselines use identical original token IDs;
comparisons against refine also include an explicitly matched subset because
the refine cache does not include every train answer.

## Literature

- [CLATTER](https://arxiv.org/html/2506.05243v1), Eliav et al., 2025 preprint.
- [FENICE](https://arxiv.org/html/2403.02270v3), Scirè et al., Findings ACL 2024.
- [DAE](https://aclanthology.org/2020.findings-emnlp.322/), Goyal and Durrett, 2020.
- [FactCC](https://aclanthology.org/2020.emnlp-main.750/), Kryściński et al., 2020.
- [MiniCheck](https://arxiv.org/html/2404.10774v2), Tang et al., EMNLP 2024.

The next research question is whether relation- and condition-preserving
counterfactual comparisons improve over the same claim extractor plus a strong
direct verifier. That extension is not established by this pilot.

## Pre-evaluation additional comparisons

Before reading aggregate labels/metrics, a qualitative sanity review motivated
`isolate.py`: same claims and sources with empty answer context, saving
`direct_isolated` and `atomic_isolated` separately. This changes input length
and positions too; it is not an isolated semantic causal intervention.

CoVe and MARCH additionally motivated `blind.py`: a proposer sees only the
original answer and creates open questions with exact answer-value quotes;
the checker sees only the source and questions; a final A/B comparison tests
whether the original value follows from the reconstructed answer. The noisy
reconstruction is a prediction feature, never evaluation ground truth. It is
not a reproduction of MARCH training or a novel method. Questions can still
leak the proposed value; direct quote containment is recorded, and semantic
leakage is not fully checked. Invalid extraction/answer parsing falls back to
the isolated direct-sentence baseline without excluding original tokens.

Run these sequentially after the original scorer exits:

```bash
python -m experiments.native_support.claim_verification.isolate \
  --output outputs/claim_verification_20260927_v1 --batch-size 2
python -m experiments.native_support.claim_verification.blind \
  --output outputs/claim_verification_20260927_v1 --batch-size 2
```

When their protocol markers exist, evaluation requires every additional score
for all 60 answers before loading official annotations. The initially declared
primary remains `atomic_raw`; additions are reported as separate candidates.

Additional sources: [CoVe](https://aclanthology.org/2024.findings-acl.212/),
[MARCH](https://arxiv.org/html/2603.24579v1), and
[ContextCheck](https://aclanthology.org/2026.findings-acl.658/). ContextCheck
supports the opposing possibility that context helps disambiguation; deleting
context is not assumed to be universally beneficial.

Earlier direct precedents also include [QAGS](https://aclanthology.org/2020.acl-main.450/),
[FEQA](https://aclanthology.org/2020.acl-main.454/),
[QAFactEval](https://aclanthology.org/2022.naacl-main.187/), and
[Q²](https://aclanthology.org/2021.emnlp-main.619/). Question-based source
reconstruction and semantic answer comparison are established baselines.

## Post-evaluation development refinement

`focused.py` is a separately frozen, post-evaluation development iteration on the
36 regression answers only. It asks for minimal answer values, checks potentially
false question premises, and uses a question-conditioned answer comparator. The
source-only checker still cannot see the proposed value. Invalid generation retains
the original tokens through the isolated sentence fallback. `focused_eval.py`
requires completion of the entire frozen development roster before evaluating.
This iteration is informed by the first-round failures; it is not unseen validation.

```bash
python -m experiments.native_support.claim_verification.focused \
  --output outputs/claim_verification_20260927_v1 --batch-size 2
python -m experiments.native_support.claim_verification.focused_eval \
  --output outputs/claim_verification_20260927_v1
```

Create `focused_protocol.json` once after recording this development decision,
and before launching the focused stage (refuse to overwrite an existing protocol):

```python
import json
from pathlib import Path
output = Path("outputs/claim_verification_20260927_v1")
manifest = json.loads((output / "manifest.json").read_text())
protocol = {"ids": [r["id"] for r in manifest["records"] if r["group"] == "regression"],
            "post_evaluation_development": True, "threshold": 0, "seed": 42}
with (output / "focused_protocol.json").open("x") as handle:
    json.dump(protocol, handle, indent=2)
```

The focused stage requires that protocol with the frozen development IDs;
use a separate output directory/protocol for any future changed prompt or model.
No candidate replaces the existing source-first/refine defaults automatically.

First-round results (60 answers): the declared primary `atomic_raw` underperformed.
Removing complete-answer context improved the same reader but produced substantial
false alarms. Blind reconstruction increased recall while severely degrading
precision. The 24-answer exploratory source holdout has only 33 positive tokens;
no broad generalization claim is supported. All candidates and failed outputs are
retained. The focused iteration is evaluated separately from these frozen results.

Focused development also completed (36 answers): AUROC .780925, AP .211977,
precision .178282, recall .607735 and FPR .179660 at zero. It detects the omitted
four-day bound but still misses the two opening-hours contradictions and intimate
polarity. Normal-answer any-alarm is 17/18. This is a negative optimization result
relative to the isolated direct-sentence baseline (.896864/.425917 AUROC/AP).
Nineteen original related tests plus two focused wiring/fallback tests passed.
