# Teacher-forced attention mechanism audit

This experiment tests three mechanisms on real RAGTruth QA responses:

1. **routing drift**: evidence-conditioned messages are replaced by messages
   from the model's own response;
2. **routing dispersion**: source routes spread, heads disagree about source
   roles, or large messages cancel in the residual stream;
3. **message-independent capture**: the observed token remains preferred when
   evidence and response attention messages are removed, while evidence does
   not help it. This is the operational candidate for parametric bias.

It is a frozen-model mechanism audit. It is not an autoencoder, a graph
encoder, a trained hallucination detector, or the earlier controlled-pair
workflow.

## Dynamic DAG

For response token `y_t = token_ids[P+t]`, the predictor is the previous
position `q_t = P+t-1`. At layer `l`, head `h`, and source position `s`, the
saved directed message is

```text
m[l,h,t,s] = A[l,h,q_t,s] W_O[l,h] V[l,g(h),s]
```

where `g(h)` is the Llama GQA query-head to KV-head map. This gives the actual
edge-conditioned transformer message: attention chooses the edge at this
sample, `V` supplies its dynamic content, and the head block of `W_O` decides
how that content writes into the residual stream. A transformer layer updates
the token node by

```text
r' = r + sum(h,s) m[l,h,t,s]
r_next = r' + MLP(RMSNorm(r'))
```

The raw artifact saves `A`, native-GQA `V`, the actual `o_proj` input and
output, layer residual input, MLP update, and final hidden state. Any edge
message can therefore be reconstructed with the referenced checkpoint. It
does not materialize the prohibitive `[L,H,R,S,4096]` edge tensor.

The existing `operator_geometry.pt` is an `[L,H,H]` Gram summary of frozen
`W_O W_V` head operators. It is useful for comparing static head codes, but it
does not contain the `[H,d,d]` `W_O` block geometry needed to measure a
sample-specific captured `V`. This audit therefore reads `W_O` from the model
already loaded for replay and computes that small block geometry once.

The principal route magnitude is

```text
e[l,h,t,s] = A[l,h,q_t,s] ||W_O[l,h] V[l,g(h),s]||_2
```

This measures what actually enters the residual stream. The code also saves
role edge magnitudes, source entropy, head-role routes, cancellation, and the
largest source positions while retaining the layer and head axes. Net role
vectors remain reconstructible from raw `A/V/W_O`.

These route quantities are observational. Only the full frozen-model replay
differences below are called end-to-end functional effects.

## Same-sample causal branches

At every layer and every processed query, the audit subtracts a selected
post-softmax residual write without renormalizing the remaining attention:

```text
removed(G) = W_O sum(h, s in G) A[h,q,s] V[g(h),s]
attention_output' = attention_output - removed(G)
```

The modified hidden states continue through the real MLP and all later layers.
Three branches are run together:

- `evidence_removed`: remove evidence-source messages;
- `response_removed`: remove attention to all response tokens, including the response
  predictor's attention diagonal; its residual token embedding is retained;
- `evidence_response_removed`: remove both message groups.

For target log probability `L`, the registered effects are

```text
C_evidence  = L_full - L_evidence_removed
C_response  = L_full - L_response_removed
interaction = L_full - L_evidence_removed - L_response_removed
              + L_evidence_response_removed
```

The combined branch is deliberately named evidence-and-response-message
removed. It is not
a question-only run: teacher forcing still supplies the predictor token and
its residual embedding. The message-independent capture signature uses the
natural zero conditions

```text
evidence_response_removed_margin > 0
full_margin > 0
C_evidence <= 0
```

so it tests whether the frozen completion dynamics prefer the observed token
despite absent grounding messages. For later response tokens this can still
include lexical continuation from the predictor residual, so it is evidence
for parametric bias rather than a pure parameter-only measurement. An MLP norm
alone is never labeled "parameter knowledge."

## Efficient extraction on a 24 GB RTX 4090

The model is loaded once in BF16 and remains frozen. Each branch uses KV-cache
teacher forcing in 128-token chunks, so attention memory scales as
`branch_batch * heads * chunk * seen_tokens`, rather than `heads * N^2`.
Per-layer value buffers retain the exact past `V` needed by message deletion.
The three intervention branches share one batch, reducing each sample to two
streaming replays: one raw capture and one three-branch intervention replay.

Each sample is transferred to CPU and saved immediately under `traces/samples`;
a single writer overlaps the previous save with the next capture. Peak reserved
CUDA memory is recorded in every sample row and the manifest.

`TOKEN_CHUNK=128 INTERVENTION_BATCH=3` is the 4090 default. For an unusually
long sample, `TOKEN_CHUNK=64 INTERVENTION_BATCH=1` performs exactly the same
audit with lower peak memory and more runtime.

## Label separation and statistical test

Capture opens the formal attention archive with embedded labels sealed. It
uses cached `token_ids` and `response_idx` as the sequence authority and reads
`source_info.jsonl` only to mark evidence, question, constraint, and other
prompt tokens. Labels are opened only after all traces exist.

Evaluation binds each saved target sequence back to the formal cache. It then
reports hallucinated-minus-correct differences within the same source,
absolute-position bin, and relative-position decile before weighting sources
equally. The onset analysis is a source-matched difference-in-differences
against correct pseudo-onsets, rather than an unmatched raw curve.

When the RAGTruth generator differs from the Llama observer, these are the
observer's teacher-forced processing and maintenance dynamics. A claim about
the exact formation process requires matching generator and observer
checkpoints.

## Run

From the repository root, the default script already points to the QA formal
cache, RAGTruth source metadata, and the local Llama-3.1-8B checkpoint:

```bash
LIMIT=1 bash experiments/attention_mechanism_audit/run_qa.sh
```

After the smoke sample succeeds, run the complete QA split:

```bash
bash experiments/attention_mechanism_audit/run_qa.sh
```

The script intentionally does not use `set -euo pipefail`. A Python traceback
remains visible, and evaluation is not started after capture fails.

Outputs:

- `traces/samples/<sample_id>.pt`: raw A/V trajectory, model states, route
  summaries, and four branch scores;
- `traces/index.jsonl`: sample paths, sizes, and peak CUDA memory;
- `traces/manifest.json`: extraction configuration and checkpoint;
- `token_metrics.npz`: aligned token-level mechanism measurements;
- `report.json`: position-matched summaries and onset tests.

The default evaluation output prints only the key routing-imbalance,
dispersion, observed-token evidence-effect, capture-candidate, and onset
results, including matched source/cell counts and the exact claim boundary.
The complete diagnostic list is available with `--all-metrics`.
An existing report can be summarized without replaying or reevaluating:

```bash
python -m experiments.attention_mechanism_audit.run summarize \
  --report experiments/attention_mechanism_audit/outputs/qa/REPORT/report.json
```

This is a post-hoc mechanism audit. It does not train or evaluate an
unsupervised hallucination detector: there is no label-free anomaly score,
calibrated decision threshold, AUROC, or AUPRC in this experiment.

Focused tests:

```bash
pytest -q experiments/attention_mechanism_audit/tests
```
