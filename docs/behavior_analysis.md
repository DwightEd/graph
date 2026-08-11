# Token behavior analysis

`scripts/analyze_behavior.py` studies one canonical split through
`BehaviorAnalysis`. It uses `ResearchDataset` for canonical attention,
provenance-bound graph loading, and evaluation labels; it has no separate
cache or data loader.

The 11 token features in `behavior.token_behavior_features` are defined only
on an **original threshold graph**: the four routing descriptors plus incoming,
prompt, and history edge counts/densities. A `relation_topk` graph changes the
retained-edge cardinality by construction, so it is rejected rather than being
silently interpreted as topology. Omit `--graph-root` to construct that
original graph directly from each canonical attention sample; pass `--tau` to
choose the threshold, otherwise the canonical attention floor is used. If a
graph root is supplied, its manifest must declare `kind: original` and must
already match the canonical split.

## Single response and token t-SNE

```bash
python scripts/analyze_behavior.py single \
  --split-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --sample-id <sample_id> \
  --output-dir outputs/behavior/<sample_id>
```

The optional `--graph-root` is useful when a verified original graph archive is
already available. It is not required for the case study.

For a response of at least four tokens, `token_tsne.png` embeds one point per
response token from only the standardized 11 behavior columns. It does not use
PCA. Perplexity is `min(30, max(2, (R - 1) // 3))`, with a fixed seed. The left
panel adds hallucination labels only after the coordinates were computed; the
right panel colors the same path by normalized response position. Consecutive
tokens are joined so the plot is a trajectory, not a bag of independent
points. `token_tsne.npz` stores coordinates and response positions.

Other single-response outputs are `behavior.csv`, `run_summary.csv`,
`behavior.png`, and `metadata.json`. Add `--control-sample-id` to overlay a
fully correct response at normalized position.

## Onset alignment

```bash
python scripts/analyze_behavior.py align \
  --split-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --radius 12 --run-policy first \
  --output-dir outputs/behavior/onset_test
```

This is a label-conditioned exploratory diagnostic, not an unsupervised or
online detector. It aligns labeled hallucination onsets and optionally compares
each one with a length-matched fully correct response. It writes raw windows,
an aggregate table and plot, matched event records, and metadata.

Canonical attention uses `post_token_query_at_same_position`: features at
response position `t` are extracted after token `t` has been read by the
observer. Therefore neither the token t-SNE nor the onset trajectories support
a claim of online next-token prediction without a separately causal cache.
