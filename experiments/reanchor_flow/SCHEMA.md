# ETCC data contract and saved arrays

## Dimensions

| Symbol | Meaning |
|---|---|
| `N` | full teacher-forced token count |
| `S=q+1` | causal source positions used by one target |
| `L` | decoder layers |
| `H` | query heads |
| `D_h` | head dimension |
| `D` | hidden dimension |
| `P` | destination positions selected by `carrier_scope` |
| `E` | coverage-retained exact attention edges |
| `U` | source units |
| `R` | candidate source roots |
| `K` | causally tested carriers |

## Input pair (`pair_schema=1`)

| Field | Shape/type | Meaning |
|---|---|---|
| `sample_id` | string | stable example identity |
| `tokenizer_id` | string | tokenizer/model vocabulary used for every ID |
| `corruption` | string | predeclared clean/corrupt construction |
| `clean_token_ids` | `[N] int64` | clean prompt plus forced response |
| `corrupt_token_ids` | `[N] int64` | aligned corrupt prompt plus identical response |
| `response_start` | scalar int | first response token position |
| `token_unit_id` | `[N-1] int64` | source-unit assignment for every causal source token |
| `unit_name`, `unit_kind` | `[U] string` | semantic source-unit table |
| `candidate_unit_id` | `[R] int32` | units allowed to differ and requiring root screening |
| `query_position` | `[T] int32` | predictor `q`; prediction token is `q+1` |
| `positive_token_id` | `[T] int32` | candidate `a` in `z(a)-z(b)` |
| `negative_token_id` | `[T] int32` | candidate `b` |
| `contrast_origin` | `[T] string` | how `a,b` were fixed; exposes label use |

Load-time validation rejects unequal lengths, response differences, unnamed prompt changes, unchanged
candidate units, invalid predictor rows and equal target candidates.

## Output provenance (`etcc_schema=1`)

Each contrast produces one NPZ named `<sample>_q<q>_a<a>_b<b>_<signal>.npz`.

| Field | Meaning |
|---|---|
| `model_id`, `tokenizer_id`, `model_dtype` | model/token coordinate provenance |
| `layer_count`, `head_count`, `head_dim`, `hidden_size` | saved model axes |
| `flow_signal` | exactly `attention` or `message` |
| `edge_score_semantics` | explicit interpretation of `edge_score` |
| `edge_payload_semantics` | exact pre-`W_O` message payload definition |
| `edge_coverage`, `gradient_steps`, `carrier_scope`, `query_chunk` | capture settings |
| `root_screen_limit` | candidates receiving exact bidirectional patches; `0=all` |
| `carrier_limit`, `message_vector_materialized` | intervention/export settings |
| `query_position`, `prediction_position` | causal coordinate pair `q,p=q+1` |
| `causal_source_count` | `q+1`; proves no future token was computed |
| `positive_token_id`, `negative_token_id`, `contrast_origin` | fixed target function |

The output repeats the pair tokens and source-unit table so every edge can be interpreted without
joining an external file. `screen_corrupt_token_ids` is the original multi-candidate corruption;
`corrupt_token_ids` is the automatically isolated selected-root world actually used by the saved
flow. `screen_pair_effect` and `pair_effect` keep the corresponding margins separate.

## Sparse edge table

All fields below have first dimension `E` and the same row order.

| Field | Shape | Meaning |
|---|---:|---|
| `edge_layer`, `edge_head` | `[E]` | exact decoder layer and query head |
| `edge_source`, `edge_target` | `[E]` | exact absolute token positions `s→q` |
| `edge_source_unit` | `[E]` | semantic source unit of `s` |
| `edge_attention_clean/corrupt` | `[E]` | native softmax gates `A+`, `A-` |
| `edge_score` | `[E]` | raw `A+` in attention mode; signed target message score in message mode |
| `edge_clean_target_score`, `edge_corrupt_target_score` | `[E]` | path-gradient action of each native message; NaN in attention mode |
| `edge_selector_score` | `[E]` | symmetric contribution from changing `A`; NaN for attention mode |
| `edge_content_score` | `[E]` | symmetric contribution from changing `V`; NaN for attention mode |
| `edge_clean_code/corrupt_code` | `[E,D_h] float32` | pre-`W_O` `AV` used for reconstruction and patching |
| `edge_clean_message_norm`, `edge_corrupt_message_norm`, `edge_delta_message_norm` | `[E]` | norms after the matching head `W_O` block |
| `edge_clean_message_vector`, `edge_corrupt_message_vector`, `edge_delta_message_vector` | `[E,D]` or `[E,0]` | optional post-`W_O` vectors |
| `edge_transition_probability` | `[E]` | residual-aware candidate route probability |
| `edge_root_throughput` | `[E]` | `T(e|selected root,target)` |

Even in attention mode the pre-`W_O` codes are retained because causal confirmation must patch real
messages. They are intervention payload, not the edge-ranking signal. This separation is recorded by
`flow_signal` and `edge_score_semantics`.

## Coverage and throughput

| Field | Shape | Meaning |
|---|---:|---|
| `row_position` | `[P]` | represented destination positions |
| `row_total`, `row_retained` | `[L,H,P]` | full backend magnitude and retained magnitude |
| `row_message_budget` | `[L,P]` | sum of retained `||m+ - m-||` before aggregation |
| `row_net_message_norm` | `[L,P]` | norm after summing retained source/head messages |
| `row_message_coherence` | `[L,P]` | net norm divided by message budget; low means cancellation |
| `row_signed_target_score`, `row_positive_target_score`, `row_negative_target_score` | `[L,P]` | retained target-aligned decomposition; NaN in attention mode |
| `row_selector_score`, `row_content_score` | `[L,P]` | retained `A`/`V` score decomposition; NaN in attention mode |
| `source_unit_route_mass` | `[U]` | `C(u→t)` for every unit |
| `residual_transition_probability` | `[L,S]` | vertical residual probability |
| `reverse_node_visit` | `[L+1,S]` | target-originating retained path mass |
| `root_conditioned_node_throughput` | `[L+1,S]` | `T(v|u,t)` |
| `selected_root_route_mass` | scalar | retained path mass ending in selected root |

At full coverage, source-unit route mass sums to one. At lower coverage that unconditioned mass may
sum below one because pruned routes terminate in the sink. Whenever selected-root mass is nonzero,
the root-conditioned node throughput still sums to one at every depth.
`reverse_node_visit[0,s]` is the token-level root mass; `source_unit_route_mass` is its exact unit sum.

## Integration ledger

Message mode saves `[L,A]` matrices over `stage_position[A]`:

- `state_delta_norm`, `state_target_score`;
- `attention_write_delta_norm`, `attention_write_target_score`;
- `mlp_write_delta_norm`, `mlp_write_target_score`.

Attention mode saves correctly shaped empty arrays rather than populating them with attention proxies.

## Exact causal results

Root table `[R]`:

- `root_unit_id`, `root_route_mass`, `root_gradient_score`;
- `root_necessity`, `root_sufficiency`, `root_causal_score`;
- `root_evaluated`, plus `selected_root_unit_id` and `selected_root_confirmed`.
- `selected_root_necessity`, `selected_root_sufficiency` and
  `selected_root_causal_score` are recomputed in the isolated world.

Corridor scalars:

- `clean_margin`, `corrupt_margin`, `pair_effect`;
- `corridor_edge_count`, `corridor_necessity`, `corridor_sufficiency`;
- `corridor_blocked_sufficiency`, `corridor_mediated_sufficiency`;
- `corridor_clean_restoration_error`, `corridor_corrupt_restoration_error`;
- `corridor_restoration_error`, `corridor_restoration_tolerance`,
  `corridor_restoration_valid`, `corridor_confirmed`.

Carrier table `[K]`:

- `carrier_layer`, `carrier_position`, `carrier_source_unit`;
- `carrier_route_throughput`, `carrier_state_delta_norm`, `carrier_target_score`;
- `carrier_necessity`, `carrier_rescue`, `carrier_block_effect`;
- `carrier_blocked_rescue`, `carrier_mediated_rescue`, `carrier_block_tolerance`;
- `carrier_confirmed` additionally requires the absolute blocked rescue to stay within
  tolerance after the downstream Value/residual block.

No hallucination/correctness label is part of ETCC schema. Such labels may only be joined after a
complete audit for external evaluation.

## Native subset schemas

真实 RAGTruth pilot 不写入 `PAIR_SCHEMA` 或 `ETCC_SCHEMA`，避免把 source-message cut 误解为
clean/corrupt factual pair。它使用两个独立版本号：

- `native_world_schema=1`：小型、模型无关、可恢复的 target contract；
- `subset_audit_schema=1`：完成精确 rerun 后的紧凑机制结果。

### Native world

| Field | Shape | Meaning |
|---|---:|---|
| `sample_id`, `tokenizer_id` | scalar | safe artifact identity and tokenizer |
| `token_ids` | `[N]` | native prompt plus teacher-forced response |
| `response_start` | scalar | first response token position |
| `token_unit_id` | `[N-1]` | source position to semantic unit |
| `unit_name`, `unit_kind` | `[U]` | passage/sentence/field/response table |
| `evidence_unit_id` | `[R]` | represented passage/sentence/field roots |
| `query_position` | `[K]` | frozen predictor positions |
| `positive_token_id` | `[K]` | observed token at `q+1` |
| `negative_token_id` | `[K]` | frozen native runner excluding observed |
| `contrast_origin` | `[K]` | label-free selection and contrast semantics |

### Compact native audit

Provenance and claim boundary:

- `world_kind=native_source_value_message_cut`;
- `claim_scope=observed-target dependence under a source Value-message cut`;
- `factual_correctness_identified=0`, `labels_used_for_capture=0`;
- `transport_score_semantics`, `functional_score_semantics`,
  `root_cut_functional_score_semantics`,
  `residual_transition_semantics`, `source_cut_semantics`.

Native/root-cut quantities:

- `native_margin`, `root_cut_margin`, `root_value_effect`;
- `all_evidence_cut_margin`;
- root table `root_route_mass`, `root_functional_score`, `root_value_necessity`,
  `root_conditional_sufficiency`, `root_causal_score`, `root_evaluated`;
- selected-root scalars, `causal_effect_tolerance`, and `selected_root_confirmed`.

Transport and aggregation:

- `row_total_transport`, `row_retained_transport` with shape `[L,H,P]`;
- `row_residual_weight`, message budget/net norm/coherence with shape `[L,P]`;
- positive, negative and signed functional score with shape `[L,P]`;
- `transport_source_unit_route_mass` before functional filtering;
- `source_unit_route_mass`, reverse-node and root-conditioned throughput after
  `functional_score<=0` edges are sent to the sink.
- `transport_token_root_mass` and `support_token_root_mass` retain layer-0 token
  contributions before and after functionality filtering.

Only the highest-throughput support edges are persisted. The full in-memory corridor is connected
only in the augmented layer-unrolled graph that includes implicit residual edges; this truncated
sparse-message export does not itself claim connectivity:

- `edge_candidate_count`, `corridor_edge_count`, `edge_saved_count`;
- `edge_saved_corridor_mass`, `edge_total_corridor_mass` and their fraction;
- layer/head/source/target/unit coordinates;
- native functional score and root-cut code projected onto the frozen native gradient
  (`edge_root_cut_native_gradient_projection`), plus native/root-cut attention and message norm;
- transition probability and root-conditioned throughput.

`edge_payload_saved=0` means `[E,D_h]` codes were used for reruns and discarded before saving.
Corridor fields use explicit native names: `corridor_necessity`,
`corridor_conditional_rescue`, `corridor_blocked_rescue`, `corridor_mediated_rescue`, plus
both-world restoration errors and validity. Carrier and stage tables retain the same coordinates as
controlled ETCC but compare native against selected-root Value-cut states.
`carrier_any_confirmed` is the local single-carrier diagnostic. `carrier_value_mediated` and
`full_chain_confirmed` are identical evaluation outcomes and require both `corridor_confirmed` and at
least one confirmed carrier; a carrier alone is not reported as a complete mediated chain.

### Manifest and evaluation

`run_manifest.json` records resolved data/model paths, dataset/source hashes, frozen selection,
target policy, backend and all audit limits. Every completed audit points to one validated NPZ;
re-running an identical configuration resumes it. A changed configuration is rejected and requires a
new output directory.

`mechanism_evaluation.json` is created separately. It records
`labels_accessed_after_capture=true` and joins labels by
`prediction_position-response_start`; labels are never copied back into native world or audit NPZs.
