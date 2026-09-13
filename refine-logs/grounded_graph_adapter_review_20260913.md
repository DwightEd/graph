# Grounded graph adapter: skeptical design review after SourceRel transfer failure

## Verdict

**A2 is worth a bounded implementation, but only as a grounded conditional
language-model and pointer-proposal experiment.** It is materially stronger than
the failed retrieval-only route because it trains a graph-conditioned token
state to reconstruct a source-derived continuation through the frozen observer
LM head, while requiring that the same state attend to an exact source
occurrence. That is a real additional learning signal, not another cosine or
prompt heuristic.

It still does not solve natural ownership by construction. Qwen restatements
are weak, source-derived supervision; source-field pointer NCE is not natural
response ownership ground truth; likelihood improvement is not factual support
or causal adoption. The only defensible first claim is whether a graph adapter
improves frozen, source-heldout **synthetic grounded continuation/pointer
objectives** and produces auditable natural candidate distributions. The new
Reasoned Graph bridge should remain a frozen flat-vs-graph reasoning baseline,
not the target generator or a source of training labels.

The previous result makes the distinction necessary: SourceRel-Mini reached
0.954 source-heldout pointer top-1 on Data2txt reconstruction, while its
natural transfer left 997/1,619 slots source-domain-unavailable and only 622
candidate-supply outputs. Better field lookup alone is therefore not the
missing semantic step.

## Smallest trainable architecture

Use **one trainable component**: `GroundedGraphAdapter`. The observer Llama
backbone, token embedding, unembedding/LM head, and source-token states are
frozen. There is no second classifier, calibrator, standalone graph module, or
relation head.

For a teacher-forced sequence `x = P || r_<t`, run the frozen model through a
chosen final hidden layer:

```text
H(x) ∈ R[T,d]                 frozen token states
N = {n_1, ..., n_K}           observed source nodes/components/bundle members
z_n = pool(H(P)[source_span(n)])
E                              only observed containment/record/component/keyset edges
```

`pool` is a frozen masked mean over exact original source token spans. A long
parent without a smaller exact component is itself a node. Unknown-valued
nodes remain nodes; missing/unmapped text has an unavailable receipt rather
than an invented vector.

The adapter contains all trainable projections and one relation-aware graph
message/cross-attention block:

```text
Z = GraphMessage({z_n}, E)                         # one observed-edge block
l_tn = <Wq H[t-1], Wk Z[n]> / sqrt(d_a)            # all K nodes eligible
A_t = softmax_n(l_tn)
g_t = Wo sum_n A_tn Wv Z[n]
hat_t = H[t-1] + gate(H[t-1], g_t) * g_t
p_A(y_t | P, y_<t, G) = softmax(W_LM hat_t)
```

`GraphMessage`, `Wq/Wk/Wv/Wo`, relation-type embeddings for *observed* edge
labels, and the bounded gate are a single adapter parameter set `θ_A`.
`W_LM` and every Llama weight stay frozen. This makes the occurrence logits
`l_tn` the pointer logits; adding a separate pointer head is unnecessary and
would be a second independently trainable predictor.

The graph block is permutation-equivariant: node IDs, record numbers, source
positions, and Qwen occurrence IDs never enter as learned input embeddings.
They remain labels/provenance only. Edges are limited to literal
field/component containment, record membership, exact source-span containment,
and explicitly represented bundle membership. There are no guessed
same-event/relation/condition edges.

For each target token, `H[t-1]` is a genuine pre-token state. At test, use the
original replay prefix only; never expose future response tokens or use a
hidden state computed after the scored target. This matters especially when an
error already appears in the prefix.

## Weak source-only bridge and exact training tensors

Split source IDs before any Qwen call. For each training source occurrence or
predeclared bundle member `o`, Qwen sees **only** the frozen source document,
its exact source spans, and an opaque occurrence ID. It returns one natural
restatement `r`, a character target span `M` in `r`, and cited occurrence IDs
`O`. The raw output, parser result, model/code/prompt hash, source hash,
occurrence spans, and tokenizer alignment are immutable artifacts.

A usable synthetic example requires only mechanical checks:

1. `O` is a nonempty subset of the supplied occurrence IDs;
2. `M` is an exact nonempty span in the raw returned text and maps to tokens;
3. all IDs/spans remain in the same source split;
4. the restatement is generated before any adapter update and is never repaired
   by a response, RAGTruth label, or human answer.

These checks prove provenance of the weak label, not semantic correctness. A
bad Qwen paraphrase, omitted condition, alias, or fabricated relation remains a
weak-label failure and must be counted. Do not retain only “plausible” outputs
by a second Qwen judgement.

With teacher forcing on the source prompt plus `r`, define token positions `R`
and target-span token positions `M_tok`. Let `O` permit multiple cited nodes.
The complete loss is:

```text
L_gen = - mean_{t in R} log p_A(r_t | P, r_<t, G)
L_ptr = - mean_{t in M_tok} log sum_{n in O} A_tn
L = L_gen + λ L_ptr
```

`λ` is a single protocol constant frozen before source-heldout evaluation; it
is not selected on natural response results. If a restatement has no valid
`M_tok`/`O`, it contributes no pointer loss and is reported separately rather
than receiving a guessed owner.

The generated span itself is never inserted into its own teacher-forced prefix:
for prediction of `r_t`, the input ends at `r_<t`. Standard teacher forcing can
still expose earlier subword pieces of the same textual value; that is inherent
and must be measured with a first-subtoken-only pointer-loss ablation, not
ignored. Source payload copying is also a legitimate shortcut. Report a
payload-masked source-view control and an edge-permuted graph control on the
same frozen examples; do not call raw reconstruction “reasoning” if either
control performs the same.

## Test-time interface

For an original observer replay `P || y_<t`, the adapter returns:

```json
{
  "response_id": "...",
  "pretoken_index": t - 1,
  "observed_next_token": "y_t coordinate only",
  "node_distribution": {"node_id": "probability"},
  "graph_conditioned_logp": "log p_A(y_t | P, y_<t, G)",
  "flat_logp": "frozen Llama log p_0(y_t | P, y_<t)",
  "delta_logp": "graph_conditioned - flat",
  "candidate_status": "proposal_only|unavailable|unknown_payload",
  "provenance": {"graph_digest": "...", "tokenizer": "..."},
  "semantic_truth": null,
  "native_certificate_count": 0
}
```

The distribution is an **applicability/owner proposal distribution**, not a
posterior over true owners. `delta_logp` is a conditional prediction comparison
on the observer's actually generated next token. It is not a correction score:
when that token is wrong, increasing its likelihood can be worse; when it is
correct, an increase can still be lexical copying. Any B/A contrast and native
route test must be constructed and frozen separately from this output.

## What the graph changes, and what it cannot change

A graph adapter can distinguish candidates whose token content is similar but
whose observed topology differs: a business `overall_rating` node and a
review-tuple rating node can receive distinct node states; city components can
remain linked to an address and record/name bundle; seven day endpoint nodes
can remain linked to a weekly bundle. This is the intended advantage over one
pooled retrieval vector.

It cannot infer that an answer's “overall rating” means a particular review
rating, that a city identifies a name, or that a response quantifies all seven
days. A weekly bundle is available to attention, not forced into an all-day
query. Only a separately applicable typed verifier may turn that source-side
structure into a condition requirement.

Address text that is not exactly mappable to a component must remain an
unavailable parent/source region. Unknown WiFi or other unknown payloads can be
attended to, but cannot yield value-level S/C/N or an exact B/A solely from the
adapter.

## Fundamental risks that remain

1. **Weak-label semantics.** Source-only Qwen restatements prevent direct
   RAGTruth leakage, but Qwen can translate an occurrence into an incorrect or
   overgeneral natural claim. The pointer label proves only the chosen source
   occurrence was supplied to Qwen.
2. **Clean-to-hallucinated shift.** Training continuations are source-derived;
   test prefixes are natural responses and can contain hallucinations, quotes,
   corrections, aliases, and prior errors. Better weak heldout loss does not
   show natural response applicability.
3. **Error-prefix contamination.** Teacher-forced original replay conditions
   later tokens on already erroneous text. A node distribution or likelihood
   delta after a first error can describe continuation of the error, not source
   adoption or correction.
4. **Alias/copy shortcut.** The base prompt already contains source text.
   Generation can win by value copying, record-name overlap, or position cues
   rather than graph structure. Payload-masked, edge-permuted, and node-order
   permutation controls are necessary diagnostic comparisons.
5. **Observed topology is not truth.** Record/bundle components are useful
   constraints, but no topology edge supplies an unobserved relation,
   condition, owner, or same-event judgment.
6. **No original-generator statement.** This is a new conditional predictor on
   a frozen observer. Its logits and attention do not explain the system that
   generated the RAGTruth response.

## Comparison with the Qwen reasoning bridge

Keep Qwen as a frozen **flat-vs-graph semantic baseline**: it reads the same
full source and response coordinates under equal budgets, returns its own
fact/evidence/S-C-N-U proposals, and is evaluated only after the outputs are
frozen. It is useful because it can reason over natural language without
pretending retrieval is entailment.

Do not use Qwen's response-side semantic output as A2 supervision, filtering,
loss weighting, threshold calibration, or a native certificate. Qwen-only and
A2 answer different questions: Qwen supplies a zero-training semantic baseline;
A2 tests whether graph-conditioned frozen-LM prediction/pointer proposals add
something beyond flat text. Neither is a verified semantic oracle.

## Minimal targeted natural evidence

No natural labels are required to run A2, publish its full candidate/loss
coverage, or decide whether its graph controls show any signal. The minimum
source-disjoint training corpus for A2 is the Qwen source-only weak-restatement
set above; report it as weak synthetic supervision.

A claim that A2 identifies natural owners or improves factual verification does
require a separate, source-disjoint natural evaluation set with at least:
`(response target span, exact source occurrence set or none, applicability,
and explicit unknown/ambiguous state)`. It need not be a complete RAGTruth
hallucination-label corpus and it must not block proposal generation. A small,
frozen audit sample covering city/name, overall/review, address, aliases, and
all-days can decide whether the weak bridge transfers. Without it, the only
valid natural report is coverage, score/likelihood distributions, and finite
verifier yield.

## Run order and stopping interpretation

1. Freeze source split, source-only Qwen restatements, raw/failed synthesis
   denominators, graph/query tokenization, and λ.
2. Train the single adapter on source-train; choose epoch only by
   source-heldout weak `L` and report `L_gen`, `L_ptr`, and every control.
3. Run the frozen adapter on the existing natural 36 and official heldout
   inputs without using labels for selection. Report all token/node proposals,
   unavailable/unknown counts, likelihood deltas, and graph-control deltas.
4. Apply only existing finite verifiers and construct/freeze B/A where they
   actually apply. Native work remains a separate conditional localization
   study.
5. If graph controls do not beat their flat/payload-masked counterparts on
   source-heldout weak data, or natural finite-verifier yield remains zero,
   preserve the negative result. Do not add another semantic head or convert
   Qwen outputs into ground truth.

This route has a credible new supervised signal but not a credible semantic
truth signal. That distinction is the condition for making progress without
repeating the retrieval-only dead end.

## Engineering review of `next_iteration/grounded_graph_adapter.py`

### Result

**Critical: 0. Required: 3.** The module has the intended single trainable
component: its pointer logits and graph residual share projections, graph IDs
mask nodes across samples, cross-sample edges are rejected, and the initial
zero output projection makes the adapter an exact identity on the supplied
pre-token states. `pointer_loss` is correctly a multi-positive OR objective;
a jointly necessary evidence bundle must therefore arrive as one compiled
inventory node, as its docstring requires, rather than as several positives.

The new CPU review tests cover node permutation equivariance, graph membership
masking and rejected cross-sample edges, a nonzero residual that changes under
observed edges, exact tiny-Llama first-step logits and CE, frozen LM-head
gradients, finite adapter gradients, and multi-positive pointer semantics.
They pass on CPU.

### Required before an actual A2 train/evaluation run

1. **Make the native-feature dtype/device contract explicit and executable.**
   The real observer paths load Llama in `bfloat16`, while a newly constructed
   adapter has `float32` parameters. Passing `bfloat16` source/query states to
   the current `nn.Linear` raises `RuntimeError: expected m1 and m2 to have the
   same dtype`. Topology tensors can likewise be on a different device and
   fail only inside an operation. The runner must either construct the adapter
   in the selected native dtype or, preferably, make the adapter's float32
   feature path explicit, convert its residual back at the defined injection
   boundary, and reject device/dtype mismatches before a forward. Add a
   bfloat16 tiny-Llama capture regression. This is an execution blocker, not a
   claim about the mechanism.

2. **Seal the exact frozen representation and coordinates at the adapter
   boundary.** `forward` correctly receives only matrices, but cannot know
   whether `query_states[t]` is the post-final-norm state that the frozen Llama
   LM head actually consumes, whether it is before its scored token, or whether
   `source_states` were pooled from the same full prompt. A runner receipt must
   bind model/tokenizer/code identity, input IDs, full prompt/source spans,
   source-node spans and graph digest, final-norm/layer representation,
   flattened query-to-target token coordinates, and source split. It must
   reject a raw block state, a future-token state, or a different source graph.
   The zero-residual identity test is meaningful only under that receipt.

3. **Replace the apparent chunking with a memory-bounded training strategy
   before full-context training.** `token_loss` calculates one full-vocabulary
   CE per chunk but retains every scalar loss and its autograd graph in
   `losses` until the final sum/backward. Thus it avoids a concatenated logits
   tensor but still retains chunk logits/backward state proportional to all
   scored tokens times vocabulary; `chunk_size` is not a bounded-memory
   guarantee. Use checkpointing or a documented streamed backward/gradient
   accumulation scheme, run a real maximum-length dry-run, and record peak
   memory and failure denominators. Do not silently shorten source or
   restatement sequences to make it fit.

### Detector evaluation contract

The proposed predeclared score
`s_t = log p_base(y_t) - log p_adapter(y_t)` can be evaluated against held-out
RAGTruth error labels over every word, including post-first partitions, without
natural owner ground truth or a typed-verifier pass. That is a legitimate
**error-prediction** evaluation if the source split, checkpoint epoch,
direction, token-to-word aggregation, missing/unavailable value, and all-word
denominator are frozen before labels are read. It is not a factuality,
ownership, correction, or causal-effect evaluation. The future runner must
compute both log probabilities from the same sealed pre-token state and frozen
LM head, keep the direction above even if AUROC is below .5, and publish all
word/post-first results plus availability counts. Weak heldout reconstruction
or pointer success cannot substitute for that natural evaluation.

### R1/R3 closure recheck

The revised kernel closes Required 1 and 3. It now rejects mixed devices and
non-floating feature matrices before a forward, explicitly promotes captured
`bfloat16` source/query states to the adapter projection dtype, and checks the
availability mask device separately. A CPU tiny-Llama in `bfloat16` now
produces exact initial frozen LM-head logits after this feature conversion;
the adapter output is intentionally float32 for trainable small-module
arithmetic and is cast only at the frozen-head boundary.

`token_loss` now uses non-reentrant checkpointing around both the frozen
projection and CE for every gradient-bearing chunk. This drops each chunk's
vocabulary logits before backward while retaining only scalar losses and the
checkpoint inputs. A direct CPU comparison verifies equal loss and
gradient-with-respect-to-hidden against uncheckpointed full-vocabulary CE.
It is still necessary to record an actual maximum-length training dry-run,
but the former guaranteed retained `T × vocabulary` intermediate is gone.

The review file now has five CPU tests; all pass, and Ruff is clean. Required
2 remains an integration requirement for the feature collector/runner rather
than a defect in this tensor kernel.

## Feature-capture review: `grounded_graph_features.py`

The actual capture kernel has the right causal geometry. It re-tokenizes the
entire original prompt without implicit special tokens, requires the recorded
explicit BOS-plus-prompt IDs to match exactly, and runs Llama on
`prompt || response[:-1]`. Thus `query_positions = target_positions - 1`: the
first response target is read at the final prompt state, and no query state
contains the scored token. Source vectors are pooled only from prompt indices;
the response suffix cannot alter them under the causal model. The final RMSNorm
hook and returned `last_hidden_state` are checked to be the same tensor, then
captured as finite float32 values. Packet rebuilding binds raw prompt/source
substring, inventory graph, response token offsets, weak spans, node order,
and every target coordinate.

Three new CPU checks use a local character-offset tokenizer and tiny frozen
Llama. They confirm final-norm query equality, first-target pretoken position,
prompt-only source pooling, source-vector invariance when the response suffix
changes, hook cleanup, tamper rejection, BOS mismatch rejection, and required
identity fields. Together with the adapter checks, eight CPU tests pass and
Ruff is clean.

### Remaining Required items before collection

1. **Close the 9-kind adapter boundary.** The feature compiler declares nine
   `NODE_KINDS`, including `text_context_provenance_bundle`, while the current
   adapter constructor defaults to eight. Passing a complete compiled graph to
   that default can index the ninth type out of range. The collector/trainer
   must derive `node_types=len(NODE_KINDS)` from the sealed packet vocabulary
   and reject an adapter checkpoint whose vocabulary/order differs. This was
   identified while the corresponding metadata fix is being made.

2. **Verify, rather than merely carry, the capture code identity.** `capture`
   currently requires a `capture_code_sha256` key in `model_identity`, but it
   does not compare the supplied value to the local frozen
   `grounded_graph_features.py` byte hash. A caller can provide an arbitrary
   string and obtain a superficially complete receipt. Compute and compare the
   local/snapshotted hash before capture, then seal the exact value in the
   receipt. Model and tokenizer file maps must be reverified by the collector,
   not trusted because their dictionary keys are present.

3. **Make array-byte verification symmetric at load time.** The receipt has
   packet/input IDs and source/query-array SHA-256 values, which is the correct
   publication material, but the future collector/trainer must store the exact
   arrays, byte-check them against these values, and re-run `validate_packet`
   against the bound inventory before concatenating samples. It must also bind
   packet/node/edge order to the array row order and preserve unavailable nodes
   rather than dropping them. Until that loader exists, this remains the R2
   integration boundary, not a semantic limitation of capture.

### Feature-kernel R2 closure recheck

The kernel-side items above are now closed. The adapter default has nine node
embeddings, exactly matching the sealed feature vocabulary; the new regression
locks this equality. `capture` computes the hash of its own executed source
file and rejects a supplied `capture_code_sha256` that differs, while the
receipt records actual observer dtype and attention implementation in addition
to the model/tokenizer identity maps. The test fixture now supplies the real
local code hash and verifies an arbitrary 64-character replacement is rejected.

Feature capture therefore has **0 Critical / 0 kernel Required**. The remaining
array persistence, byte verification, packet/inventory/order binding, and
prepared-run provenance are correctly deferred to the new feature runner; they
must be reviewed as that runner's integration contract before any training or
natural scoring run.

## Feature-runner integration review

`grounded_graph_feature_runner.py` correctly separates source-only weak
training examples from annotation-free natural response replay. It binds each
packet to a source inventory, rebuilds the packet with the current tokenizer
before reading an array, checks receipt self-digest, packet digest, model and
tokenizer identity, representation, raw float32 array bytes and shapes, and
the node/query row counts. Natural packets also require their reconstructed
IDs and response offsets to reproduce the old frozen row exactly. The runner
keeps an explicit unavailable entry rather than inventing a source vector.

Three new temporary-artifact CPU tests cover a valid persisted array load plus
rejection of changed array bytes, changed packet IDs, changed entry packet
digest, and a receipt whose self-hash is recomputed after model identity
tampering. Adapter, feature, and runner review tests total twelve passing CPU
checks; Ruff is clean.

### Required before `prepare`

1. **Publish a complete manifest only for the exact entry roster.**
   `verify(complete=True)` currently accepts whatever artifact subset the
   manifest names. It must require exactly `summary.json` plus `.npz` and
   `.json` for every `entries.status == "available"`, and no feature arrays
   for unavailable entries; the summary must freeze available/unavailable
   examples and token counts. Otherwise an interrupted or buggy capture can
   omit an available row, publish a smaller self-consistent manifest, and
   silently shrink a later all-word denominator.

2. **Freeze executable code and model/tokenizer bytes before packet
   construction.** The current `prepare` loads the tokenizer and builds packet
   JSON before calling `freeze_code` and recording `model_manifest`. A change
   during that interval can leave packets produced by an already loaded old
   tokenizer/code version while settings snapshot newer files. Snapshot and
   validate the exact code/model/tokenizer identities before any tokenization
   or `prepare_example` call, then verify they remain identical at publication.
   This is a provenance race, not an adversarial-caller scenario.

### Runner closure and deterministic reconstruction check

The feature runner now closes both runner findings: it snapshots all listed
code and the observer/tokenizer file manifest before loading the tokenizer or
building a packet, rechecks model bytes before publication, and `verify`
checks live and snapshotted code. Its complete manifest now requires exactly
the summary plus both persisted files for every available entry. The summary
is checked against the entire sealed entry roster for status counts, token
count, and actual forwards. A temporary-artifact regression proves a manifest
which omits an available entry is rejected while an unavailable entry remains
in the census without an invented feature file.

`grounded_graph_reconstruction.py` is mechanically honest about the new data:
the response is verbatim source text, source SHA/split and original prompt are
retained, every selected target is an exact raw lexical span with an existing
field/context owner, Data2txt selects the first field word, QA/Summary select
context words after their fixed prefix/stride, and the global warmup applies
before selection. Unknown field values and duplicate coordinate proposals are
not turned into pointer labels. Two CPU tests reconstruct Data2txt and QA
examples and verify these properties. This is source-copy-coordinate
pretraining only, never natural semantic ownership; the design's owner-payload
erasure control remains required before interpreting a reconstruction gain as
grounding rather than copying.

One **Required** remains in the reconstruction artifact verifier: it
reconstructs every example it finds but does not prove that the complete
selected source roster was retained. It must recompute or otherwise bind the
exact selected source-ID set and task/split census from the frozen input, then
require `examples`, `source_artifacts`, and summary source/pointer counts to
match it. Without that check, an omitted selected source can produce a
self-consistent smaller reconstruction manifest and silently alter the
pretraining denominator.

### Reconstruction roster closure

The reconstruction verifier now closes that final data-integrity requirement.
It derives the expected source IDs from sealed source-artifact names, rejects
duplicates or omissions, compares the exact task/split selected census to the
frozen selection census, derives summary source/pointer/no-pointer counts from
the complete examples, and rechecks each source's actual raw text SHA,
deterministic internal split, and task before accepting its rebuilt coordinate
record. A targeted temporary-artifact regression proves an otherwise
self-consistent manifest that omits one selected source is rejected.

The bounded adapter, capture, feature runner, and reconstruction review set is
now **0 Critical / 0 Required** for the documented CPU preparation and feature
capture interfaces. This closes engineering provenance only. The source-copy
objective remains pretraining, and its owner-payload erasure control plus the
frozen full-word natural evaluation remain necessary before interpreting any
model result as a detection or grounding gain.
