# Attention mechanism audit rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory. The rules below are mandatory for this experiment.

- Put the mechanism algorithm first. Keep the implementation direct and as
  small as possible; do not add feature factories, schema layers, compatibility
  wrappers, or defensive checks that do not protect the scientific result.
- Maintain one formal implementation. Remove superseded paths instead of
  adding `new`, `v2`, fallback, approximate, or degraded variants.
- Preserve the exact frozen-model method: dynamic attention/value/output
  messages and causal message deletion. Never silently replace it with an
  attention-only proxy, static operator geometry, or another cheaper method.
- Treat `train` and `test` only as physical attention-cache locations. Capture
  both, then pool every available QA response into one evaluation. There is no
  fitting split, calibration split, or split-level headline result.
- Keep labels sealed during capture and score construction. Open them only to
  compute post-hoc AUROC and AUPRC on the pooled token scores.
- The default report contains one main score (`causal_route_capture`) and
  exactly three mechanism components (`routing_imbalance`,
  `source_dispersion`, and `message_independent_preference`). Additional
  diagnostics must answer a stated scientific question and must not appear in
  the default output.
- Generate population statistics and figures after all data have been pooled.
  Do not automatically emit one JSON or figure per sample. A sample figure is
  an explicit, on-demand operation selected by sample ID.
- Keep one one-click `run_qa.sh`. Do not use `set -e`, `set -u`,
  `set -o pipefail`, or `set -euo pipefail`. Check the exit code after each
  command so the original Python traceback stays visible and later stages do
  not run after failure.
- Test the message equations, score equations, pooled evaluation, and CLI
  contract. Do not grow tests around incidental schemas or error wording.
- When delegating work to another agent, repeat the exact scope and these
  scientific constraints in its task. Do not assume conversation context or a
  sibling directory's `AGENTS.md` will be inherited.
