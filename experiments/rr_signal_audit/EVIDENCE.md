# Evidence registry and decision gates

This registry separates reproduced observations from new hypotheses. Values are
from earlier full-test exploratory runs on the current benchmark and therefore
justify what to audit, not a final independent-test claim.

| mechanism | frozen observation | status in this experiment |
|---|---:|---|
| exact history-edge fraction | AUROC 0.6418 | exact scalar parity baseline plus full layer/head field |
| RR raw residual energy | AUROC 0.6601, AUPRC 0.1344 | `received_topk` robust residual |
| strict causal received support | AUROC 0.6784, AUPRC 0.1375 | target baseline; no sign refit |
| normalized entropy | raw AUROC 0.4086, lower-oriented 0.5914 | exact scalar diagnostic only |
| history mass fraction | AUROC 0.5865 | exact scalar plus separate PR/RR channel fields |
| direct retained Lookback | raw AUROC 0.3899, lower-oriented 0.6101 | exact compatibility baseline only |
| fixed SetWalk | smoke AUROC 0.5603 versus no-walk 0.6367 | falsified; not part of the active representation |
| source-prediction surprise | approximately random | retired baseline |

## Acceptance rules

A channel-preserving block is useful only when all of the following hold:

1. its frozen causal score improves on its matching exact scalar baseline;
2. performance is reported separately for QA, Data2txt, and Summary;
3. source-group bootstrap uncertainty is reported;
4. its joint PPCA score is compared with the same block's independent density;
5. channel shuffling establishes whether cross-channel alignment matters;
6. no test label selects direction, component, rank, or fusion weight.

No equal-weight fusion is a primary method. A new graph propagation operator is
admitted only if it beats both the direct route field and a topology-preserving
rewiring control on the same token rows.
