# Automatic query scoring after SourceRel natural-transfer failure

## Decision

Build one frozen, **candidate-supply-only** scorer before any new native run:
a masked response query is ranked against a complete source graph with
provenance-preserving source views. The first implementation is a frozen
late-interaction baseline over local Llama-3.1 or Qwen3-base hidden tokens. It
has no query-side relation, owner, condition, or quantifier classifier.

This is the smallest automatic interface that can rank natural candidates
without silently recreating an SRL/reader oracle. It can improve the current
question from “is the necessary source occurrence in the graph and retrievable
from the exact masked context?” to a measurable answer. It cannot certify that
a top-ranked source is the natural owner or that a source relation is true.

The SourceRel-Mini validation result (0.954 source-heldout top-1) remains a
source-field reconstruction result. Its supervision is generated from Data2txt
source fields and does not make a response-side natural ownership label.

## Fixed input contract

For every response target, freeze this object before encoding or scoring:

```json
{
  "response_id": "...",
  "source_id": "...",
  "target_id": "stable span + graph digest",
  "target_span_chars": [a, b],
  "parent_span_chars": [p, q],
  "prior_context_span_chars": [0, p],
  "raw_parent_text": "unaltered parent text",
  "raw_prior_context": "all prior response text",
  "masked_parent_text": "only [a,b) replaced by <VALUE>",
  "masked_full_query_text": "prior context + masked parent",
  "target_surface_kind": "observed scalar/content run/quote/unknown",
  "mask_mapping": {"char-to-token": "exact frozen mapping"},
  "query_code_and_tokenizer_digest": "..."
}
```

The model receives **only** hidden states recomputed from
`masked_full_query_text`; it never receives an unmasked target-token hidden
state. The exact target coordinate, not every detected slot or an entire
sentence, is replaced. Parent and prior-context text remain visible. The scorer
uses all non-padding query token states except the sentinel's own tokens. Each
state carries deterministic, nonsemantic features: signed bucketed distance to
the mask, `in_parent`, `in_prior_context`, token offset, and whether its raw
span intersects a preserved source/response graph span. These are coordinates,
not role labels.

A candidate is an observed source node or a graph-proven bundle. It stores a
stable ID, original source spans, record/component membership, field/key text
when literally present, text/unknown status, and key-set containment. Its
**ranking view** masks its own payload span with `<SOURCE_VALUE>` while retaining
record/parent context and literal field/key strings. The original payload is
kept in a separate evidence view for a later finite verifier; it is excluded
from owner ranking. This prevents an apparent owner score from being a target
value copy.

The source inventory must retain, rather than discard:

- scalar fields and long text parents;
- `surface_component` children only where exact source-span provenance survives;
- field-record, review-tuple, address/location, attribute-key, sentence/clause,
  and weekly-schedule bundles;
- unknown-valued source fields with `payload_status=unknown`;
- omitted or unparsable regions with a reason and denominator membership.

A source node belongs to a candidate pool only through observed containment and
record/component membership. Co-membership is not a same-event assertion.

## Smallest scoring implementation

Encode each frozen query and ranking view independently with an already-local
base model. Let `Q` be all eligible masked-query token vectors and `D(c)` the
source-view token vectors. L2-normalize both. The fixed baseline is
ColBERT-style late interaction:

```text
LI(q, c) = sum over query tokens i of max over source-view tokens j
           dot(Q[i], D(c)[j])

score(q, c) = LI(q, c)
```

Type is an eligibility rule rather than a learned or weighted score: reject only
an observed finite literal-syntax conflict (for example, a date-like target
against a finite number-only candidate); retain `text` and `unknown` against
every candidate. It cannot infer a semantic relation. A fixed score receipt
must retain the MaxSim source token index for each query token, score, model and
tokenizer hashes, target-mask mapping, candidate graph digest, and all rejected
candidate reasons. Ties use stable candidate IDs. A candidate is
`ambiguous_identity_match` precisely when every retained MaxSim witness is in
an observed `record_identity` source region and none is in the candidate's own
field/component region; otherwise it is `candidate_supply_only`. This is token
provenance, not an owner label or a score threshold.

This follows the independently encoded token interaction in ColBERT, whose
MaxSim operation is a retrieval score, not role binding or entailment. The
existing method note also supports retaining lexical and distributed signals as
separate diagnostics (Duet); it supplies no result that either signal verifies a
relation.

### Architecture ranking

1. **Frozen local late interaction — implement first.** It needs only token
   states, exact masking, and source provenance already required by the graph.
   It gives inspectable token-to-token receipts and does not need natural owner
   labels. It is a baseline, not a semantic verifier.
2. **Fixed lexical channel beside late interaction — implement only as an
   ablation.** Literal keys, `city`, `rating`, and `review` can be useful, but
   it must be fitted on no natural target labels. Use a deterministic tokenizer
   overlap/BM25-style channel over the same masked views, not the Data2txt
   TF-IDF vocabulary learned from source reconstruction.
3. **Local Llama/Qwen cross-encoding baseline — lower priority.** It is
   feasible only as a frozen likelihood/representation baseline over the exact
   same rendered pair. A prompt asking the model to declare an owner would be a
   new pseudo-semantic oracle and is rejected. Its extra context cost and less
   inspectable scalar make it secondary to late interaction.
4. **Learned all-token ranker — diagnostic only until new supervision exists.**
   Its only current training origin is the 720/163 source-disjoint Data2txt
   pointer corpus: query is a masked source field; positives are exact source
   occurrences (all identical masked-context positives); negatives are
   same-source, same-observed-type candidates. It may report held-out pointer
   reconstruction and test its transfer as candidate supply, but cannot call
   natural response ranking accuracy, calibrate a natural threshold, or provide
   B/A certification. No RAGTruth labels, Qwen judgements, or pseudo-GT enter
   this training set.

No model download is needed. A missing local encoder or a source view above its
frozen context limit is recorded as unavailable; it is never truncated into a
new candidate value.

## Explicit source-JSON to natural-source shift

The scorer has two rendering modes with the same candidate schema:

| Property | Source JSON mode | Natural-source mode |
| --- | --- | --- |
| Field label | literal AST key/path | literal heading/key when observed, otherwise absent |
| Record | AST dict/list containment | observed source record/paragraph/component containment |
| Component | typed scalar span | provenance-preserving surface component only |
| Unknown | `None`/unknown literal field | unknown source payload or unresolved component |
| Training status | exact pointer supervision available | no owner ground truth is assumed |

Natural mode must not invent a JSON key, a field path, or a record boundary to
make a score look structured. When a label is absent, the rendered label is an
explicit `<NO_OBSERVED_KEY>` token and `key_status=unobserved`; when a component
cannot be mapped exactly, it is unavailable rather than guessed. Consequently,
Data2txt pointer training is neither a proxy natural-owner label nor a way to
hide source-domain shift.

## Actual failure-family coverage

The inventory/scorer contract addresses each observed family without claiming a
semantic solution:

| Failure family | Required observed candidates | What ranking can establish | What remains unverified |
| --- | --- | --- | --- |
| city → name | city component, address bundle, containing business/name record | city query context retrieves a provenance-bound address/record candidate | that city identifies that business in the response relation |
| overall rating → review | business `overall_rating` and each review tuple/rating kept separate | raw `overall`/`review` context ranks different record-bearing nodes | which rating proposition the response asserts |
| address missing | full address parent plus exact components when mappable; explicit unresolved entry otherwise | prior failure becomes available/unavailable with a reason | any omitted component's factual value |
| seven-day bundle | individual day endpoints plus weekly bundle and membership list | schedule candidates are supplied together without arbitrary query filtering | that the response quantifies all days, unless an existing typed verifier fires |

For a weekly bundle, all seven members are a source-side observed structure. A
query is restricted to all seven only after an existing typed response verifier
proves the quantifier. The ranker never assumes this from a sentence or from
bundle membership.

## Decisive next CPU plan

1. Freeze inventory and query-view manifests for the actual 36 natural
   responses and a source-heldout source subset. Report all source nodes,
   components, unknowns, bundles, omissions, and target availability by failure
   family before a score is inspected.
2. Encode/rank the fixed late-interaction and lexical baseline on the same
   candidate IDs. Publish every target's top-k, score receipts, availability,
   ties, ambiguity state, and denominator. No arbitrary “10 B/A” threshold.
3. Run only already-existing finite verifiers: exact span/value equality and
   typed Data2txt hours where their declared syntax applies. These verifiers may
   label `applicable`, `supported`, `conflict`, or `unknown`; all other natural
   candidates stay `unverified_candidate_only`.
4. Compare candidate availability and finite-verifier yield against the prior
   graph, separately for city/name, rating/review, address, and weekly bundles.
   This is a label-independent resource decision for whether a later native
   measurement has eligible inputs. It is not an owner-accuracy evaluation.
5. Freeze any resulting B/A events before native work. If no finite verifier
   applies, retain the candidate result and stop; do not backfill with a reader
   label, a same-sentence assumption, or a generic condition formula.

The decisive output is thus an auditable natural candidate-and-verifier
coverage table. It can improve the real bottleneck—missing or unrankable source
occurrences—without pretending that candidate rank is semantic truth.
