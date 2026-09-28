# Unlabelled temporal graph pilot

Reuses `probabilistic_detection` packs, fresh label-free 123D expansion, official
source splits, metrics and case renderer. No new LLM forward. This graph connects
valid tokens at +/-4 positions within each answer; it is not native attention.

Isolation Forest: 150 trees, 512 subsamples. CoLA-inspired contrast: separate
123→64→32 node/context encoders and a pair readout. The positive context is the
mean of adjacent nodes, excluding self; negatives match generator and relative
position octile and exclude the complete source. Original pair means observed,
not factually correct. The topology control permutes nodes within each answer,
preserving topology and degree multiset, and trains a separate identical model.
Fixed seed42, 12 epochs ×32 batches ×4096, AdamW lr1e-3/wd1e-4. Both models use
source-balanced sampling. Feature transform/forest use a fresh source-balanced
bootstrap reference of up to 200000 rows. Rank scales use all fit token scores.

Higher negative positive-pair logit is more anomalous; scores are not calibrated
hallucination probabilities. The model can learn topical/positional shortcuts.
Only natural held-out evaluation determines whether the proxy is useful.

`fixed_unsupervised` is .75×source-pair rank +.25×route-window rank, frozen before
this experiment's labels. Its threshold is the 95th percentile of the unlabelled
development mixture; this is not a promise of 5% normal-token FPR.
`selected_dev` explicitly uses development truth to select among base scores
and source/auxiliary convex mixtures (auxiliary weights .1/.25/.5), and sets the
normal-token95 threshold. Thus fitting is label-free but final selection is
label-assisted. Both outputs remain separate; no test-label selection.

```bash
python main.py graph-anomaly --phase fit --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase pilot --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase cases --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase select --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase score --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase evaluate --output outputs/unsupervised_graph_20260928
python main.py graph-anomaly --phase cases --include-test --output outputs/unsupervised_graph_20260928
```

Use the existing research Python environment with torch/sklearn and
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`. Scoring uses CUDA0
for the small detector; fitting never accesses the training archive's labels.
All three tasks must have frozen test predictions before test evaluation/cases.
The historical test has been exposed: these are exploratory results.
The inherited case annotation reader parses dataset JSONL records before ID
filtering; excluded test labels are not returned to development or fitting.
