# typed_hours / typed_hours_prepare engineering review — 2026-09-13

Scope: bounded `code-review-and-quality` review of `next_iteration/typed_hours.py`, `tests/test_typed_hours.py`, the added review regression `tests/test_typed_hours_review.py`, and `next_iteration/typed_hours_prepare.py`. I did not read gold labels, did not start GPU work, and did not change implementation files.

## Verdict

Critical: 0.

Required: 1, in the prepare CLI provenance fields. Core `typed_hours.py` has no Required blocker for the stated narrow adapter scope.

## Required

1. `typed_hours_prepare.py` uses the same field name, `settings_sha256`, for two different hashes.

   Evidence from a label-blind dry run to `/tmp/typed_hours_prepare_review_20260913`:

   - `summary.json.settings_sha256` and `rows/9273.json.settings_sha256` equal `digest(settings)`: `a59a9ad92b25417521c52f0dfc1a4120c90e988532d995efcb4521747e8a7ae9`.
   - `manifest.json.settings_sha256` equals `file_sha256(settings.json)`: `87222aa094966f0ac247d44a68a5e6a9ca3cb0f77a1ab71fabc152bab75a1745`.

   This is a freeze/provenance correctness issue because downstream audit code cannot know whether a `settings_sha256` value should match the JSON object digest or the written artifact hash. Fix before launching the full prepare: either make every `settings_sha256` field the settings file sha, or split the fields into explicit names such as `settings_file_sha256` and `settings_object_digest`. The file-sha option is cleaner for artifact manifests; if the object digest is useful, store it under a separate name.

## Checked and accepted

- The core adapter is explicitly narrow: Data2txt root dict only, all seven day `hours` fields required, 24h source intervals must have `end > start`, response endpoints must have explicit AM/PM, and only a single endpoint change can produce a contrast. This is the right boundary for the current typed-provider claim.
- `The business` is treated as the Data2txt root business owner only under the single-record typed schema. This must not be generalized to arbitrary text, but it is declared in `PROTOCOL` and enforced by the grammar.
- Source proof is recomputed before native handoff: `native_contrast()` calls `prepare_row(row)` and requires exact equality with the prepared artifact before returning a contrast.
- The 9273 row reproduces the intended handoff without labels: one typed contrast, shared prefix length 711, continuation lengths 11/11, seven day endpoint spans mapping to 21 source token keys, and slot masks covering the edited endpoint in both branches.
- The non-target WiFi conjunct is checked independently at the typed layer. Native source keys intentionally cover the target hours endpoint only; full-sentence support remains auditable through the prepared schedule artifact. If future downstream reports need a standalone proof from the contrast artifact alone, include the WiFi occurrence id/span there, but that is not a launch blocker for this typed endpoint handoff.
- The full-roster dry run over the 36-row frozen input was label-blind and matched the expected denominator: 24 task-outside rows, 6 source-schedule-unresolved rows due to overnight/ambiguous source intervals, 6 typed-assessment rows, 47 base units, 45 grammar-outside facts, 1 typed-supported fact, and 1 native input compiled.

## Tests run

From `graph/` with the requested conda Python and CPU thread limits:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_typed_hours.py tests/test_typed_hours_review.py -q
```

Result: `19 passed`.

I added `tests/test_typed_hours_review.py` to lock the non-initial-base native handoff boundary: the B/A event is the exact response prefix through the hours sentence, excludes the later sentence, preserves source endpoint token mapping, and produces aligned slot masks.

## Targeted R1 closure

The sole Required finding is closed.

Targeted check after the fix:

- `typed_hours_prepare.py` now writes `settings_object_digest` to rows and summary, and `settings_file_sha256` to the manifest. The old ambiguous `settings_sha256` field is no longer used by this prepare path.
- The settings object digest is computed once as `settings_digest = digest(settings)` immediately after `settings.json` is written, then reused for every row and for `summary.json`.
- `typed_native_runner.py` consumes the new fields: it compares `manifest["settings_file_sha256"]` to `file_sha256(settings.json)` and `summary["settings_object_digest"]` to `digest(settings)`.
- `typed_hours_prepare.py` now reuses `route_graph.audit_runner.rows(args)`, which checks the annotation-free 14-field schema, duplicate response IDs, digit response IDs, and response text hash before writing artifacts.
- The observer tokenizer provenance now uses an explicit file-name allowlist including `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`, with optional tokenizer assets when present.

I did not find any remaining Required blocker in the bounded typed-hours core/prepare scope.
