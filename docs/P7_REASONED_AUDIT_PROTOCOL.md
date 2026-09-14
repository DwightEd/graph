# P7: bounded reasoning before the same localized grounding decision

Candidate proposed after P6's full frozen confirmation failed its endpoint: all-token
AUROC0.7816538805, not>0.8. P6 outputs, original freeze and scoped evaluation remain
unchanged. No reuse of either old R04 validation or P6 confirmation for new claims.

The development audit exposed false assertions in P6's draft (e.g. attributing a
wage tax rate to private pensions, treating intimate=False as supported). This is
motivation, not evidence that reasoning will fix them, and not an original-generator
causal claim. P7 is a frozen external retrospective verifier with zero additional
training; full source and full response are visible. It is not a new graph method.

## Fixed change and full-token score

All P6 audit instructions, source/task/answer ordering, final word-verifier prompt,
FP32 arithmetic with unchanged BF16 resident weights, cache guard0.005, whitespace
units and original-token overlap mapping remain identical. The only changed
component is draft generation: enable Qwen3 thinking, with1024 generated-token
reasoning budget, then at most384 final-audit tokens. The word verifier still has
thinking disabled and sees only the final fallible audit, not the internal reasoning.
It returns raw z(B)-z(A), primary name reasoned_source_risk. No score blending,
normalization, sign/threshold fitting, token filtering or answer rejection.

Thinking stops at the atomic </think> token or EOS or the1024 cap. Natural closing
is followed by two newlines. If absent, only terminal EOS tokens are removed from
the continuation context, an explicit budget-close message and </think> are added,
and a final audit is generated. Raw generated reasoning IDs, retained IDs and
injected tail IDs are separately recorded; injected tokens are not model output.
Final audits reaching384 are marked truncated and retained, never selectively rerun.
Input context is never truncated. The cap check reserves64 extra closure tokens.

Both generation stages use one sample with temperature0.6,top_p0.95,top_k20,min_p0.
The seed is fixed per(source,instruction,response):20260914+SHA256 first4-byte big
endian, modulo2^31. This makes case ordering irrelevant; no best-of-K/oracle selection.
The two-stage budget design and sampling guidance follow the original Qwen sources:
[model card](https://huggingface.co/Qwen/Qwen3-8B),
[budget example](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md).
Official guidance is not an assurance of hallucination-detection improvement.

Reasoning, sampling and extra compute change together, so a P7-P6 difference cannot
be claimed as the isolated causal effect of reasoning or correct binding. The goal
of this iteration is an effective baseline; mechanism controls remain separate.

## Inputs, gates and execution

Reuse research@03909e02, torch2.8.0+cu126/transformers4.57.1, existing Qwen3-8B files;
no installs, downloads, weight updates, paid services or competing GPU job.
R04 input SHA3258b79c7886dc2f6f93ff6c6e9e38c695a773b7dab2ddfc4cba271a3baac38f.
Pilot is the same fixed IDs17019/17020, no annotation access;8 existing synthetic
canaries are engineering fixtures only. Development is all32 original development
answers,16 sources,5170 original tokens. Complete predictions/hash freeze precedes
the ID-scoped development annotation join. No partial-run metrics.

Fresh document witness executes only this pilot ONCE, after checking GPU idle and
the output/log path absent. Capture actual subprocess returncode, all failure logs,
manifest, generated IDs/closure/truncation, raw-score mapping and cache comparisons:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_local_grounding_reasoned.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p7_reasoned_pilot_20260914_v1
```

Log: graph/runs/p7_reasoned_pilot_20260914_v1.log. Launch record:
graph/outputs/P7_PILOT_LAUNCH_20260914.json. Do not launch development concurrently.
If engineering gates pass, main may run the same command with development and a
new outputs/p7_reasoned_development_20260914_v1 directory. Failure is retained;
do not relax precision, change budget or replace samples while this version runs.

Evaluate all original development tokens via development_assess_scoped.py with
primary reasoned_source_risk and complete P6/P5/population references. Only a frozen
future source-disjoint confirmation can establish>0.8; development never suffices.
The already-observed P6 new64 cohort is now spent for method selection.
