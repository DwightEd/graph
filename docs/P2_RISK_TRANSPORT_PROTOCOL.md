# P2: retain uncertainty along generated dependencies

2026-09-14 Asia/Shanghai. Candidate designed AFTER inspecting P1 development metrics; this is development-informed hypothesis selection, not blind unsupervised model selection. Inference and scoring use no hallucination labels, no fitted detector, and no extra training. Validation has not been scored. This tests the existing H20/H21/H22 propagation hypothesis; no novelty claim is made.

P1's primary all-token AUROC/AP were 0.5491/0.05586 versus entropy 0.6121/0.09042. Through-first-error entropy AUROC was 0.8032, with only 16 positives. That contrast motivates testing retention rather than another semantic verifier prompt. It does not establish that all hallucinations begin with uncertainty.

At prediction query q=P+t-1 collect actual native attention to source keys and earlier response tokens. Use all 32 layers and all 32 heads equally, without label-based head selection. Exclude instruction/BOS keys from the provenance normalization, retain all source and earlier response keys. Let a_tj be mean attention to earlier response token j; s_t is mean source mass; w_tj=a_tj/(s_t+sum_j a_tj), m_t=sum_j w_tj. Empty denominator means m=0.

Fixed primary recurrence:

    R_t = (1-m_t) H_t + sum_{j<t} w_tj R_j

H is native pre-token entropy. This is a directed graph transport operator, not a classifier trained on hallucination labels. Native attention is a predictive dependency proxy, not proof of causal information use; source attention does not certify applicable evidence. P2 therefore cannot by itself solve wrong binding or claim a universal mechanism.

Controls fixed BEFORE P2 scoring: same row mass but all generated weight on immediate predecessor; same row mass spread uniformly over history; same row mass with reversed generated-key order; native entropy/NLL/negative logit margin; causal current-clause maximum entropy. The source reset term and node attributes are identical across edge controls. Actual edge structure contributes only if it beats mass-matched controls; otherwise drop the topology claim. Do not choose whichever control scores best and rename it the primary method.

Resources: reuse existing research conda, same Llama3.1-8B-Instruct and RTX4090. Eager attention is streamed one layer at a time; save mean full-key attention for every prediction query, not all heads/hidden states. This is NOT R05 full attributed-graph completion. Capture weights actually used in A@V; never alter them. Compare recomputed native logits-derived controls with immutable P1 SDPA controls, report discrepancies. Capture all rows/layers, prohibit edges to current/future target tokens, store inputs/code/graph/score hashes. No OOM retry or truncation.

First fixed R04 responses17019/17020, label-free pilot. Gate: finite complete outputs, all32 layers captured, causal edges valid, mean attention row sum error <=0.01, each native NLL/entropy discrepancy from P1 <=0.5, and extrapolated development <=10 minutes. The 0.5 tolerance bounds backend drift, not accuracy; exact deltas must be reported. Development next only if gate passes. No validation unless PRIMARY all-token pooled AUROC AND AP improve over all three native controls AND the chain/uniform edge controls. Freeze exact primary and run untouched16validation sources if eligible. Positive generalization requires paired source-cluster95% intervals above zero; if absent or conflicting no claim. All secondary scores and all negative results remain visible.

Exact remote invocation (detached Popen wrapper, fresh exclusive log as in P1):

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_risk_transport.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p2_transport_pilot_20260914_v1
```

For gated development replace pilot with development and output basename with p2_transport_development_20260914_v1. Save this protocol and launcher with outputs. No scorer reads evaluation.json or annotations. P1 is preserved as COMPLETE_NEGATIVE; R05 remains unrun.
