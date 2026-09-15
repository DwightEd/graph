# Evaluate existing results with missing split metadata

Six-field canonical attention files carry token IDs and attention, not necessarily
split/source/offset fields. An older run can therefore have a correct input cache
path ending in `train` but empty `record_json.split`. This is an evaluation join
problem, not evidence that train and test attentions were mixed.

The evaluator now binds metadata in memory, leaving scoring results and resume
settings untouched:

1. Snapshot completed sample files before opening annotation files.
2. Read the original input path from saved `settings.json.cache` and each record's
   relative cache path. An exact input directory named train/test supplies a split
   hint. The output directory name and requested SPLIT never assign truth labels.
3. Use saved response IDs, normalizing the old `attention_<id>` filename fallback,
   to join the existing `response.jsonl`. The official split must agree with the
   input partition and any existing saved split. Existing hash/source conflicts
   remain errors; missing values are filled only for the evaluation report.
4. Use saved offsets when an existing response hash binds them. Otherwise verify
   the cached token sequence with the original observer tokenizer. Offsets are
   accepted only when re-encoded token IDs match exactly, not just in length.
   No model weights or attention are recomputed. The tokenizer is loaded locally
   and only when needed; a missing/wrong tokenizer is an explicit error.
5. Read task_type from the existing source_info.json/jsonl beside the annotations
   when task metadata is missing. This file is not used as hallucination labels.

```bash
git pull --ff-only origin main
SPLIT=train BOOTSTRAP=0 \
  bash experiments/unsupervised_token_graph/evaluate.sh --completed-only
```

If old results lack offsets or a response hash, tokenizer discovery uses declared
paths in existing cache manifest/settings files. It does not guess from the
answer-generator name. Alternatively pass `TOKENIZER=/existing/observer/model`
(or `--tokenizer`). This must be the tokenizer used to capture these token IDs.
The repository's documented Llama-3.1 observer path is
`/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`;
its presence on a particular server has not been verified here.

The report includes input_cache, annotation/source paths, split_origin and
alignment_origin. Full/partial metric equations and q -> q+1 alignment are
unchanged. evaluation_partial.json never replaces complete.json, summary.json,
settings.json, any sample NPZ, or the formal evaluation.json. A preview is still
an execution-order subset, not a full benchmark.

Regression tests: `python -m pytest tests/test_legacy_split_binding.py -q`.
Tokenizer tests use a deterministic test tokenizer; no real Llama model or
server-side RAGTruth data was loaded in this patch's validation.
