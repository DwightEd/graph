# Source–carrier information on token graphs

The default is label-free, per-head source-routing analysis. The old graph
autoencoder is an explicit baseline. The method is described in `PLAN.md`;
this input adapter does not change its information or reanchor equations.
No new RAGTruth detection result is claimed.

## Run directly on existing NPZ files

From the repository root, with the research environment active:

```bash
CACHE=/path/to/canonical_attention \
  bash experiments/unsupervised_token_graph/run_all.sh
```

The foreground script runs tests, analyzes all matching NPZ files and then
attempts evaluation. No annotation path means an explicit evaluation skip,
not a successful benchmark. The default output is
`outputs/source_carrier_information_npz_v2`. Existing v1 outputs are untouched;
use this new directory because the input-identity contract has changed.

`CACHE` may also be a single NPZ. `PATTERN='**/*.npz'` is the recursive default.
Select an attention directory, not a mixture of scores and feature archives.
No LLM inference, GNN training, dense [L,H,N,N] expansion or new data export is
required. Unchanged samples can be resumed with identical input settings.

## Identity and offsets: use what already exists

1. Read NPZ fields directly: `id` / `response_id` / `sample_id`, `source_id`,
   `official_split` / `dataset_split` / `split`, `task` / `task_type`,
   `generator` / `model`, `offsets`, `token_ids`, and `response_sha256`.
   An embedded `response` string supplies its hash automatically. Scalar JSON
   `record_json` is supported too; object arrays are not unpickled.
2. If there is an existing `inputs.jsonl`, `records.jsonl` or `records.json`
   beside the cache directory (or in its parent), reuse it automatically.
   Ordinary ID-based records need no new `cache` or `trace` field.
3. For an index elsewhere, set `POPULATION` to the existing population directory,
   or `INDEX` to an existing index file/directory. Missing offsets/token IDs may
   be read from its existing `<id>.npz` companion (reuse/structural layout).
   The companion is read for identity only, never as full attention.

For example, reuse the population already used by `reuse_detector`:

```bash
CACHE=/path/to/canonical_attention \
POPULATION=../reanchor/outputs/ragtruth_population_20260912 \
  bash experiments/unsupervised_token_graph/run_all.sh
```

This is an EXISTING directory, not a request to create a new metadata file.
Matching recognizes `<id>.npz` and `attention_<id>.npz`. When transferring offsets
by ID, the original full `token_ids` and prompt boundary must match. Same ID or
same token count alone is insufficient. Existing native offsets are also
cross-checked. Feature-only offsets without matching tokenization cannot repair
missing alignment. No retokenization or fabricated source IDs are used.

A six-field canonical file still works for graph analysis on its own. If neither
it nor an existing index supplies identity/offsets, evaluation remains unavailable.
`--metadata` accepts the older explicit cache/trace index for compatibility, but
it is not required and no new metadata file is generated.

## Supported attention layouts

- Canonical CSR: `token_ids`, `response_idx`, `attention_diagonal`,
  `response_row_ptr`, `response_column_indices`, `response_values`.
- Legacy four-dimensional `attention` or `data`; compact legacy rows use q=P-1+r
  unless explicit `query_positions` are stored.
- `AttentionAdjacency.save_layer_head()` export: `adjacency` [R,N], `token_ids`,
  `response_idx`, `layer`, `head`. Rows use q=P+r; physical layer/head IDs survive.

`local_attention` windows or scalar `values` features do NOT contain all prompt
and history edges. Passing them as a graph input raises a clear error; missing
edges are never invented. Native `labels` members remain unread even when they
are present in a cache.

## Evaluate without recomputing

`ANNOTATIONS` is the original RAGTruth `response.jsonl`. If the selected existing
population has `settings.json` with `dataset`, that path is saved automatically;
like `reuse_detector`, relative dataset paths use the repository working directory.
Otherwise supply the existing annotation file:

```bash
CACHE=/path/to/canonical_attention \
ANNOTATIONS=/path/to/RAGTruth/response.jsonl \
  bash experiments/unsupervised_token_graph/run_all.sh
```

This assumes identity and offsets are already in NPZs or auto-detected records.
For evaluation only:

```bash
OUTPUT=outputs/source_carrier_information_npz_v2 \
  bash experiments/unsupervised_token_graph/evaluate.sh
```

Set `ANNOTATIONS` here as well if it was not resolved from population settings.
Evaluation reads labels only after scoring and checks answer/source/split identity.
It reports seven token scopes, coverage, pooled/source-weighted AUROC/AP and
source bootstrap; real/permuted scores are compared on common coverage.

## What is measured

Each physical layer/head is processed separately. A same-channel token-DAG walk
terminates at prompt roots, self boundaries or unknown mass. These masses sum
to one in the declared surrogate; unknown mass is not interpreted as hallucination.
`terminal_entropy_bits = carrier_conditional_entropy_bits + carrier_information_bits`.
`source_mismatch_bits` compares direct prompt roots with prompt roots reached
through response carriers. Missing either branch is NaN, not agreement.

Outputs remain `samples/<relative cache path>.npz`, `settings.json`, `summary.json`,
and `complete.json`. Canonical q=P+r scores align to prediction q+1: the first
response prediction is absent and the final outside-response prediction is not
evaluated. Coverage makes this explicit. Reanchor thresholds use the past only;
`--offline-influence` exports future-use diagnostics separately, never as online scores.

Exact prompt-token roots are default. Cost is roughly O(retained_edges * roots)
per channel and O(queries * roots) state. `--root-bins 64` is optional LOSSY root
address grouping, not exact provenance. `--layers 10 11 --heads 3 7` selects
physical channels without averaging attention. Historical autoencoder access:

```bash
python -m experiments.unsupervised_token_graph.run autoencoder \
  --cache /path/to/cache --output outputs/legacy_gae --phase fit
```

The source-flow method is an attention-routing surrogate, not a WV/WO causal
trace or semantic truth proof. Topic changes/corrections can cause mismatch;
a high-confidence error without both comparison branches can be missed.
