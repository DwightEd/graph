# P4: delayed decision-state update, development-only CPU iteration

2026-09-14. Designed AFTER seeing P3 development: exact revisit/onset coincidence
is uncommon, but lead-8 coverage exceeds equal-count position-block randomization;
holding entropy AT the revisit failed. The new delay=8 is explicitly selected
from this already-observed development result. It is NOT independently validated.

Primary: a revisit opens a causal 9-position update interval [event,event+8].
At the event reset state to native entropy. During that interval update state
to max(state,H_t); thereafter retain it. Another revisit resets/opens a new
interval. Before the first event use current entropy, no backfill to past tokens.
This tests a reading-to-commit delay, not semantic correctness or a learned model.

Controls fixed before P4 scoring: native entropy; native causal rolling-9 maximum
(no reading signal); the same delayed state machine triggered by causal entropy
events; the same machine triggered every16tokens starting at17. No labels enter
any state or score; no additional model forwards. The comparison with rolling
maximum is essential to distinguish a reading-event benefit from smoothing.

Reuse all32 frozen P3 development predictions. No filtering cases or reading
validation. Freeze P4 predictions before separate annotation join. Report all
token AUROC/AP, all-stream onset, within-answer AUROC and paired source CI.
Do not flip signs or promote a winning control to the primary. Stop if primary
does not improve AP AND within-answer AUC over both native entropy and the
rolling maximum control. This candidate may fail without rejecting revisit
as a mechanism-audit location selector.

CPU invocation, in remote graph repository with existing research Python:

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.read_commit_state --predictions outputs/p3_revisit_development_20260914_v1 --output outputs/p4_read_commit_development_20260914_v1
```

Prediction output must not exist. No GPU, no installs, no test-set labels, no push.
