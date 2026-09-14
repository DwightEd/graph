# New fixed source confirmation, before P6 development results

Use preexisting population **inputs**, not annotation or evaluation files.
Official-test only; fixed Llama2-7b/13b responses. Exclude all R04 source IDs and
exact source-content hashes. Choose32 sources (12QA/12Summary/8Data2txt), sorted
within task by SHA256(`P6-confirm-v1|20260914|source_id|source_sha256`), exact-content
deduplicated. Retain missing generator combinations in denominator; do not replace.
Do not rank by errors, scores, quality, length or compute budget. No GPU forwards.

All64 responses form ONE primary confirmation endpoint; do not peek at partial
cohorts or choose a winning task. Source overlap withR04 must be zero. These are
official-test sources, but prior population measurements touched them, so do not
claim globally pristine unseen data. No labels will be joined until a selected
method is frozen and all64 new predictions are complete. P5 failed R04 validation
remains failed. Future methods may not repeatedly optimize this confirmation set.
