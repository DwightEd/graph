# Source–carrier information on token graphs

Current work follows `PLAN.md`. The new default is a label-free, per-head
routing diagnostic; the old graph autoencoder remains an explicitly selected
baseline. No new hallucination benchmark result is claimed.

## Run on existing attention caches

From the repository root:

```bash
python -m pytest tests/test_reanchor_information.py tests/test_reanchor_pipeline.py -q
python -m experiments.unsupervised_token_graph.run \
  --cache /path/to/canonical_npz \
  --pattern '**/*.npz' \
  --output outputs/source_carrier_information_v1
```

This reads existing NPZ files, one sample and one physical layer/head at a time.
No LLM forward pass, logits, hidden states, GNN training, or annotations are
required. `--resume` reuses unchanged samples with identical settings.
`--layers 10 11 --heads 3 7` selects physical channels without averaging them.
Exact prompt-token roots are the default. `--root-bins 64` optionally groups
prompt addresses to reduce computation; this is a **lossy root quotient**, not
exact provenance. Direct/relay/missing mass is still retained. Exact compute is
roughly O(retained_edges * prompt_roots) per channel, with O(queries * roots)
state; it can be expensive for large prompts and many heads.

The graph consumes all retained CSR edges; it does not add a second threshold
or redistribute missing mass. Diagonal weights terminate at the self boundary;
missing edges terminate at unknown. Prompt roots are boundary identities, not
recovered prompt-internal computations. A same-channel token-DAG path is a
routing surrogate, not the real cross-layer WV/WO causal path.

Outputs: `samples/<relative cache path>.npz`, `settings.json`, `summary.json`,
`complete.json`. Each sample stores `[channel, query]` measurements, layer/head
IDs, actual query positions and their next-token prediction positions.
Canonical rows start at q=P, so the first response prediction is unavailable;
the final q=N-1 row predicts outside the saved response. Evaluation reports this
coverage instead of padding either endpoint with a fabricated score.

## What is measured

`prompt_reach`, `self_terminal_mass`, `unknown_mass` sum to one in the declared
path model. Unknown is not interpreted as hallucination. `prompt_path_length`
shows whether prompt access is direct or multi-hop.

`terminal_entropy_bits = carrier_conditional_entropy_bits + carrier_information_bits`
is the entropy chain rule for root Z and last carrier C. It does not measure
factual knowledge. `source_mismatch_bits` is the weighted JS between direct
prompt roots and prompt roots reached through response carriers. Missing either
branch makes it NaN: absence is not agreement. `history_permuted` preserves
per-row direct/self/missing mass and the history-weight multiset in lag bands,
not exact distances or global degrees. It tests dependence on relay identity.

Reanchor candidates use positive prompt/old-remote gain on fixed absolute
coordinates and a threshold computed from past strengths only. A link aging
across the local window does not trigger an event. FAI is available only with
`--offline-influence` and is saved as `offline_fai` with its observation count;
it is excluded from the online evaluation entry.

## RAGTruth evaluation needs identity and offsets

The six canonical attention fields suffice for graph analysis, not for joining
annotations. Supply a label-free JSONL metadata file using ORIGINAL tokenization
coordinates (not guessed retokenization). Example record:

```json
{"cache":"test/attention_10005.npz","id":"10005","source_id":"SOURCE_ID","split":"test","task":"QA","generator":"llama-2-7b-chat","response_sha256":"EXACT_RESPONSE_HASH","offsets":[[0,3],[3,8]]}
```

`cache` is relative to `--cache`; legacy `trace` is also accepted. Offsets are
response-relative character spans, one per response token. Hashes, source IDs
and official split are checked against the annotation file only after scoring.

```bash
python -m experiments.unsupervised_token_graph.run \
  --cache /path/to/canonical_npz --pattern '**/*.npz' \
  --metadata /path/to/label_free_metadata.jsonl \
  --output outputs/source_carrier_information_with_identity
python -m experiments.unsupervised_token_graph.run evaluate \
  --predictions outputs/source_carrier_information_with_identity \
  --annotations /path/to/RAGTruth/response.jsonl \
  --output outputs/source_carrier_information_with_identity/evaluation.json
```

The evaluator reports seven token scopes, pooled/source-weighted AUROC/AP,
coverage, and source bootstrap. Real/permuted mismatch scores are additionally
compared on common coverage. The 0.9 channel quantile is fixed and is applied
to scalar measurements AFTER head-specific analysis, not to attention tensors.
Do not select score direction, root bins, heads, or thresholds on test labels.
Prompt deficit is conditional on non-unknown mass; it is not a calibrated
hallucination probability. Topic changes and corrections may also yield source
mismatch, and high-confidence errors with no direct prompt branch may be missed.

## Historical autoencoder

```bash
python -m experiments.unsupervised_token_graph.run autoencoder \
  --cache /path/to/cache --output outputs/legacy_gae --phase fit
```

`autoencoder_run.py` retains the previous entry; `model.py`, `graph.py`, and
`score.py` remain baseline implementations, not the new source-flow algorithm.
The shared evaluator now handles tied scores correctly. Historical reports and
stored old outputs are not rewritten; old lookback-named commands/NPZ schemas
are not silently treated as the new measurements.
