# Surface owner v1 reader failure review — 2026-09-13

Scope: read-only code-review-and-quality fault check of `outputs/surface_owner_v1_20260913`, its `reader_cache`, and the frozen reader path in `next_iteration/surface_verifier.py`, `next_iteration/surface_runner.py`, `next_iteration/reader_receipt.py`, and `route_graph/frozen_reader.py`. I did not load GPU, did not modify frozen code/results, and did not read gold labels. This review checks whether the observed C36 failure is caused by implementation/cache/label bugs or by the finite first-token reader semantics.

## Verdict

I found no Required implementation bug in request construction, cache identity, label order, model identity, or strict-original cache reuse. The observed failures are real outputs of the frozen finite-label reader, and they show that the first-token SCNU/SCIU/PFU/VKU verifier is not semantically reliable enough to serve as a hard evidence/contrast supply gate.

This is not a positive result: C36 produced no usable contrast handoff. It is an interface/model limitation, not a hidden code-path success.

## Evidence checked

Run status and scope:

- `outputs/surface_owner_v1_20260913/progress.json` reports `status=complete`, `stage=all`, `native_forward_calls=0`, `labels_evaluated=false`.
- `B/` has 36 files, `C/` has 36 files, and `reader_cache/` has 2199 files.
- The frozen settings and executed-code snapshot match the live files for the inspected path: `surface_verifier.py`, `surface_runner.py`, `reader_receipt.py`, `frozen_reader.py`, and `soft_graph_phases.py`.

C-stage aggregate:

- 1619 target slots were assessed.
- 44 target slots passed `C+N >= .8` and therefore opened strict candidate checks.
- 145 candidates were strictly checked.
- Final statuses among attempts include 124 `edited_full_base_not_supported_or_multislot_error`, 16 `non_target_meaning_or_grammar_not_preserved`, 3 `value_only_edit_not_validated`, and 2 `selected_owner_or_constraint_not_validated`; none reached a native contrast certificate.
- All 145 strict candidates had `edit_kind.V < .8` in the stored checks, even though one had V as the argmax below threshold.

Reader/cache accounting:

- Unique cache records by label set: 1764 `SCNU`, 145 `SCIU`, 145 `PFU`, 145 `VKU`.
- The 1764 `SCNU` records equal 1619 target assessments plus 145 edited-base assessments. The strict `target_original` checks did not create new cache files because they reused the prior target requests.
- C artifacts report 2344 returned finite requests and 145 cache hits. This exactly matches 1619 target calls + 145 candidates × 5 checks, with one cached strict-original per candidate.
- Sum of actual `FrozenReader.calls` across C artifacts is 2199, equal to the unique cache files. No reader cache record had digest, identity, label-order, max-new-token, or reader-error problems.
- Full check over the 145 strict candidates found zero mismatches between the initial target assessment and `verification.checks.target_original` in `request_sha256`, `reader_record`, or `scores`.

Label tokenization/request reproduction:

- The frozen Qwen tokenizer maps the unspaced labels to single tokens: `S=[50]`, `C=[34]`, `N=[45]`, `U=[52]`, `I=[40]`, `P=[47]`, `F=[37]`, `V=[53]`, `K=[42]`.
- Leading-space variants are different tokens, but the rendered chat prompt ends after the assistant thinking block and blank line; the frozen request explicitly scores the unspaced label tokens and records `max_new_tokens=1`.
- For an inspected cache request, recomputing the rendered prompt hash from the stored instruction and payload matched `rendered_prompt_sha256`, so the cache is tied to the actual prompt text.

## Concrete failing requests

1. Supported-looking target assessed as N.

   Cache file: `reader_cache/d3abad69c4c496a6ad4acaf0197f7bb3326d26c74d6e67aab11a46834f3a251a.json`.

   Request summary:

   - Label set: `SCNU`.
   - Original base: `Additionally, it is used as a flavoring agent in candies, root beers, chewing gums, mouth fresheners, mouthwashes, toothpastes, and other oral care products.`
   - Slot: quote `candies`, surface type `content_run`, span `[549, 556]`.
   - The supplied source text contains `this oil is used as a flavoring agent in candies, root beer, chewing gums, ...`.

   Stored logits/probabilities:

   - logits: `S=33.0`, `C=30.25`, `N=39.75`, `U=37.5`.
   - probabilities: `S=0.0011`, `C=0.00007`, `N=0.9036`, `U=0.0952`.

   This is not a label-order inversion or missing-source bug; the source substring is present in the request, and the model still assigns high N.

2. Absurd selected occurrence assessed as S.

   Cache file: `reader_cache/cce8d3964cb3e4040d576cab285a132b3d9aaa3f7c63c26a383b928395b00765.json`.

   Request summary:

   - Label set: `SCIU`.
   - Original base: `Additionally, it serves bagels, made-to-order sandwiches, and a ham and cheese croissant.`
   - Edited base: `Additionally, it serves 2021-10-27 01:50:28, made-to-order sandwiches, and a ham and cheese croissant.`
   - Slot: `bagels`, surface type `content_run`.
   - Occurrence: display surface `2021-10-27 01:50:28`, field path `['review_info', 0, 'review_date']`, value type `str`, surface type `content_run`.

   Stored probabilities:

   - `selected_occurrence`: `S=0.9992`, `C=0.00026`, `I=0.00055`, `U=0.00003`.
   - The same candidate was rejected elsewhere: edited-base `S≈0`, preservation `F=0.9946`, edit-kind `K=0.9859`.

   This shows the local selected-occurrence label can mean “the selected literal exists in source” rather than “this source occurrence is the correct owner/role support for the edited assertion.” The final finite stack did not pass the candidate, but the SCIU subcheck itself is not trustworthy as an owner oracle.

## Implementation findings

No Required implementation bug found:

- Cache filenames equal `digest(request) + '.json'`.
- Cached request model identity equals the frozen `reader_files` + frozen `code_sha256` identity.
- Prediction label order equals request label order for all cache files.
- `FrozenReader.calls` counts actual model forwards/generations; cache-hit accounting is separate and consistent.
- `ReceiptReader` verifies the exact cache bytes before returning a finite prediction, and `verify_receipt` binds path, hash, instruction, payload, labels, and prediction.
- `surface_runner` uses `validate_target_assessment` immediately after `assess_target`, and strict candidate verification reuses the same target-original cache record.
- The run did not execute native interventions and did not read evaluation labels.

Protocol/interface limitations exposed by this run:

- `TARGET_PROMPT` + one-token SCNU scoring can mark a directly source-mentioned slot as N. That breaks the target CN gate independently of any native mechanism or owner matcher.
- `NODE_PROMPT` + one-token SCIU scoring can assign near-certain S to a selected occurrence whose value is syntactically and semantically wrong for the edited assertion. In the bagels/date case, the occurrence also reached the reader because a string date scalar is typed as generic `content_run`; however, the payload still included `field_path=review_date`, and the finite reader ignored the owner/role mismatch.
- The strict stack is fail-closed in these examples, but it fails by rejecting all usable contrasts, not by producing a reliable filtered set. Therefore the C36 result should be reported as reader-interface failure / zero contrast supply, not as evidence that no natural native risk exists.

## Required status

Required code bugs in the inspected frozen implementation: **0**.

Required methodological/protocol conclusion before any next run: do not treat this finite first-token reader as a hard ground-truth-like verifier or as a reliable owner gate. Any next interface must remove or replace this open semantic burden, or explicitly report that contrast supply is blocked by finite-reader invalidity. Prompt wording changes alone would be speculative unless validated against independent labels; the evidence here is that the current frozen finite-label interface is semantically unstable despite correct implementation.
