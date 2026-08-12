# 实验追踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | pipeline sanity | MART | train→test smoke | coverage, finite score | MUST | TODO | labels only in evaluate |
| R002 | M1 | simple confound | relative-position only | train→test | AUROC/AUPRC | MUST | TODO | same evaluation rows |
| R003 | M1 | primary non-GNN | MART full | train→test | token AUROC/AUPRC | MUST | TODO | frozen NPZ checkpoint |
| R004 | M2 | feature attribution | MART deletion suite | train→test | paired ΔAUROC/ΔAUPRC | MUST | TODO | fixed feature groups |
| R005 | M3 | support × message | threshold/typed-mass × 0/2 steps | train→test | paired metrics | MUST | TODO | same source splits/seeds |
| R006 | M3 | topology necessity | source-shuffle best GNN | train→test | paired metric drop | MUST | TODO | preserve target/relation/weights |
| R007 | M4 | robustness | subgroup + length match | test | grouped metrics | NICE | TODO | no model selection |
