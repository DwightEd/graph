# grounded graph data/synthesis bounded engineering review — 2026-09-13

Scope: `docs/GROUNDED_GRAPH_MODEL_20260913.md`, `next_iteration/grounded_graph_data.py`, and `next_iteration/grounded_graph_synthesize.py`. I added CPU-only review tests in `tests/test_grounded_graph_data_review.py`. I did not edit implementation files, run prepare, load a model, read labels, or start GPU work.

Verdict: **Critical 0, Required 2** before fresh synthesis execution.

## Required

1. **Identical seven-day bundles can be silently dropped by the eight-target cap.**

   `target_candidates()` constructs `identical_weekday_bundle` candidates, but then hashes all candidate families and takes the first eight. In a simple Data2txt source with `name`, seven identical weekday hour fields, and one unknown WiFi field, the eligible set is nine targets: one name, seven day scalars, and the all-weekday bundle. The selected eight in the current run were the name plus all seven day scalars; the bundle was dropped. That means the source-only training data can omit the explicit multi-member owner exactly where the design needs it to learn bundle ownership rather than seven independent scalar owners.

   This is not a semantic-label problem; it is a deterministic sampling/protocol issue at `grounded_graph_data.py` lines 112-128. Minimal fix: prioritize explicit bundle owner nodes before their member scalar families when the cap binds, or collapse the seven identical day scalars behind the bundle and fill remaining slots afterward. Add census fields for `bundle_eligible` and `bundle_selected` so the prepared data cannot claim bundle support when none is actually emitted.

2. **Prepared requests do not bind the loaded inventory task/upstream settings strongly enough.**

   `prepare()` validates the parent inventory manifest and selected source artifact hashes, then checks only `inventory["text"]` and `inventory["source_id"]` against the selected roster source. It does not check `inventory["task"] == source["task"]`, nor does it load the inventory `settings.json` to verify that the inventory was compiled from the same input SHA/protocol expected by this run. This is exactly the `source/task/settings` binding boundary the protocol depends on: stale or mismatched inventory metadata can pass if source id and raw source text match.

   Minimal fix: after reading the inventory settings, require its input SHA/protocol to match the current run's intended inventory dependency, and for every selected source require `inventory["task"] == source["task"]` in addition to source id/text. This should be mirrored in `grounded_graph_synthesize.verify()` so execution catches an inventory settings change after preparation, not only source artifact changes.

## Verified by targeted tests

The passing review tests cover these boundaries:

- `choose_sources()` uses only official train rows, deduplicates by raw source SHA, and keeps source SHA splits disjoint under a reduced-count fixture.
- `compile_templates()` invalidates every duplicate target ID rather than first-wins, while preserving other valid examples.
- Malformed JSON keeps the selected target denominator and records missing targets instead of pretending usable weak examples exist.
- `decode_result()` handles batch padding/EOS and only treats output after the Qwen thinking close token as final JSON text.

The failing review test is intentional and captures Required finding 1:

- `test_target_candidates_keep_unknowns_out_of_payload_targets_and_add_explicit_weekday_bundle`

## Test command

Run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_grounded_graph_data_review.py -q
```

Result: **4 passed, 1 failed**. The failure is the seven-day bundle selection gap above.

## Non-blocking notes

- The modules consistently mark generated templates and compiled examples as weak supervision, not semantic ground truth, and do not use natural responses for synthesis payloads.
- The source-only prompt intentionally gives Qwen the target source value and asks for a `<VALUE>` template; this is acceptable for source-derived weak restatement, but normalized paraphrases of the target value outside `<VALUE>` remain a semantic weakness rather than a mechanical validation bug.
- `batch_actual_forwards` is recorded per result as a batch-level count, so downstream aggregation should use summary-level `actual_batched_forwards` rather than summing per-record batch counts.

---

## R1 closure after fixes

Root fixed both Required findings before synthesis prepare.

1. The weekday bundle selection blocker is closed. Hour fields now share the `field:weekday` family, and `identical_weekday_bundle` is sorted first among family-first candidates. The prior failing fixture now selects the explicit seven-member bundle within the eight-target cap instead of consuming all slots with redundant weekday scalar targets.

2. The inventory binding blocker is closed. `inventory_parent()` now binds the complete inventory manifest/settings to the current input SHA, inventory schema, and no-labels protocol. Prepared settings store the inventory settings SHA, digest, and protocol. Per-source prepare validates wrapper settings digest plus source id/text/task. Execution `verify()` rereads the parent/settings and checks manifest, settings SHA, digest, protocol, selected artifacts, model identity, and live/snapshot code.

I added one regression for the new `inventory_parent()` boundary and reran the bounded test file.

Final bounded status: **Critical 0, Required 0** for `grounded_graph_data.py` and `grounded_graph_synthesize.py` under the stated weak source-only reconstruction scope. This remains engineering readiness only; it does not claim semantic correctness, natural transfer, owner ground truth, or internal mechanism success.

Command run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_grounded_graph_data_review.py -q
```

Result: **6 passed in 12.04s**.

Optional note: `target_census` still reports only aggregate eligible/selected/unknown counts. It would be useful to add bundle-eligible and bundle-selected counts for later audit readability, but the deterministic selection bug itself is fixed and this is not a prepare blocker.
