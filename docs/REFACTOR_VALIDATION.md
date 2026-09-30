# Refactor validation, 2026-09-30

- Full retained repository suite: **412 passed in 14.70s** (`python -m pytest -q`).
- All 19 explicit root CLI stage help routes passed.
- Frozen fixed baseline: exact scores and thresholds for QA, Summary and Data2txt in every fit/development/test partition; all **2700 test answers / 424408 test tokens**, maximum score error **0**. No annotation array accessed, no model forward needed.
- New native message checks: eager and SDPA tiny Llama, causal ports, no-op replay, whole-head finite/VJP agreement, physical site/alignment rejection and hook cleanup. Tiny-model checks do not validate real-8B fidelity.
- Graph core: exhaustive random tiny-graph optima, 240 continuous-case min-marginals, tied cuts, independent direct scores, scope/fork/support exits, continuity and capacity limits, missing evidence, and empirical reference invariants.
- Fresh `experiment-bridge` code review (gpt-5.6-sol xhigh; same-family/provisional): **ACCEPT bounded refactor/core** after fixing unknown-gate coercion, invalid native coordinates, and fork/anchor validation. No remaining blocking/high software finding in reviewed scope.
- Static retained-code audit: no missing local imports or syntax errors; existing untracked routing_likelihood imports remain valid.

Environment: Python 3.11, torch 2.8.0+cu126, transformers 4.57.1, NetworkX 3.6.1; tests execute on CPU. Raw logs are in `outputs/graph_refactor_20260930/`. Shared review/deletion/experiment records remain under `/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/gsm8k_states_20260929/`.

No new natural-data detection efficacy is claimed. The semantic observer, source-disjoint reference assembly and 8B finite fidelity/graph ablations remain pending. See [architecture and theory](ARCHITECTURE.md) and [cleanup scope](CODE_CLEANUP.md).
