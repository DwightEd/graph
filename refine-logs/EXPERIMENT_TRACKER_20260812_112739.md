# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | prefix/no-label audit | canonical CSR | toy | invariance, label access | MUST | TODO | alignment first |
| R002 | M0 | reconstruction overfit | full encoder | toy | support/weight/distribution loss | MUST | TODO | loss must decrease |
| R003 | M1 | scalar baseline | graph features + Isolation Forest | 5-fold OOF | token AUPRC/AUROC | MUST | TODO | all tokens |
| R004 | M1 | edge baseline | edge marginals + robust GMM | 5-fold OOF | token AUPRC/AUROC | MUST | TODO | no labels in fit |
| R005 | M2 | main run | channel-aware GNN + K4 mixture | 5-fold OOF | primary/secondary metrics | MUST | TODO | seed 0 first |
| R006 | M2 | variance | main run seeds 1,2 | 5-fold OOF | mean/CI | MUST | TODO | after R005 gate |
| R007 | M3 | adjacency | relation-preserving rewire/no-message | 5-fold OOF | paired ΔAUPRC | MUST | TODO | proves graph use |
| R008 | M3 | attributes | relation collapse/channel mean | 5-fold OOF | paired ΔAUPRC | MUST | TODO | mechanism isolation |
| R009 | M3 | dynamics | K1/no-GRU/time-shuffle/no-q0 | 5-fold OOF | NLL, ΔAUPRC | MUST | TODO | density claim |
| R010 | M4 | full statistics | final OOF records | all test tokens | clustered CI/forest/LOO | MUST | TODO | labels opened here |
| R011 | M4 | learned representation plots | fold-local embeddings | held-out fold | PCA/trajectory/innovation | MUST | TODO | no cross-fold PCA |
