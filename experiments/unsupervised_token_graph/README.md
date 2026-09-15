# Source–carrier information on token graphs

The default is label-free, per-head source-routing analysis. The old graph
autoencoder is an explicit baseline. The method is described in `PLAN.md`;
this input adapter does not change its information or reanchor equations.
No new RAGTruth detection result is claimed.

## Run train and test with the saved server paths

From the repository root, with the research environment active:

```bash
bash experiments/unsupervised_token_graph/run_all.sh
```

No CACHE or POPULATION assignment is needed for the default server layout.
The script runs tests once, then processes these two existing directories in order:

```text
train: /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
test:  /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
```

`PATTERN='**/*.npz'` searches recursively, including an `attention/` child if present.
Each split has its own output and is evaluated with the matching split selection:

```text
outputs/source_carrier_information_npz_v2/train/
outputs/source_carrier_information_npz_v2/test/
```

The train report is a train-set diagnostic, not an official test result. Split
identity still comes from the NPZ/index and is checked against annotations;
the shell script does not relabel records based on their directory.
No annotation path means an explicit evaluation skip, not a successful benchmark.
Existing unsuffixed v1/v2 outputs are not overwritten or moved. Resuming requires
identical cache paths and settings; the new split folders start separate runs.

To run only one directory:

```bash
SPLIT=test bash experiments/unsupervised_token_graph/run_all.sh
SPLIT=train bash experiments/unsupervised_token_graph/run_all.sh
```

`ATTENTION_ROOT`, `TRAIN_CACHE` and `TEST_CACHE` can override the server defaults.
`OUTPUT` is the base output directory in this two-split mode. Extra analysis
arguments, for example `--layers 10 11 --heads 3 7`, are forwarded to each run.
If CACHE was exported in the terminal earlier, use `unset CACHE` to restore the
default two-directory mode.

An explicit `CACHE` keeps the previous single-cache interface (directory or NPZ):

```bash
CACHE=/path/to/canonical_attention SPLIT=test OUTPUT=outputs/custom_graph \
  bash experiments/unsupervised_token_graph/run_all.sh
```

In single-cache mode OUTPUT is the exact result directory, with no split suffix;
set SPLIT to the split being evaluated (default test). Select attention files,
not a mixture of scores and feature archives. All runs stay in the foreground.
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

Only when an existing index is elsewhere, point POPULATION to its actual directory.
No default population path is assumed:

```bash
CACHE=/path/to/canonical_attention \
POPULATION=/path/to/existing_population \
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
# Default: outputs/source_carrier_information_npz_v2/test
bash experiments/unsupervised_token_graph/evaluate.sh

# Train diagnostic: outputs/source_carrier_information_npz_v2/train
SPLIT=train bash experiments/unsupervised_token_graph/evaluate.sh
```

An explicit `OUTPUT` selects any other saved result folder without adding a suffix.
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
