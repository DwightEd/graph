"""Official character annotations, lexical units and cached model tokens are distinct."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import write_json
from .positions import merge_spans
from .sentences import sentence_intervals


WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def response_counts(record, task):
    text = record["response"]
    labels = record["labels"]
    covered = np.zeros(len(text), bool)
    for label in labels:
        start, end = label["start"], label["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid official annotation offsets: " + record["id"])
        covered[start:end] = True
    lexical = np.array([character.isalnum() for character in text])
    sentences = sentence_intervals(text)
    positive, fully_covered = 0, 0
    for start, end in sentences:
        content = lexical[start:end]
        hits = covered[start:end][content]
        positive += int(hits.any())
        fully_covered += int(len(hits) > 0 and hits.all())
    words = list(WORD.finditer(text))
    word_gold = [bool(covered[word.start():word.end()].any()) for word in words]
    later = merge_spans((label["start"], label["end"]) for label in labels)[1:]
    later_mask = np.zeros(len(text), bool)
    for start, end in later:
        later_mask[start:end] = True
    return dict(id=str(record["id"]), source_id=str(record["source_id"]), task=task,
        model=record["model"], split=record["split"], responses=1, error_responses=int(bool(labels)),
        raw_spans=len(labels), sentences=len(sentences), error_sentences=positive,
        fully_covered_sentences=fully_covered, lexical_words=len(words), error_words=sum(word_gold),
        later_span_words=sum(later_mask[word.start():word.end()].any() for word in words),
        span_text_mismatches=sum(text[item["start"]:item["end"]] != item["text"] for item in labels),
        multisentence_spans=sum(sum(a < label["end"] and b > label["start"]
                                    for a, b in sentences) > 1 for label in labels))


def summarize_responses(frame, keys):
    counts = [name for name in frame.columns if name not in ("id", "source_id", "task", "model", "split")]
    if keys:
        table = frame.groupby(keys)[counts].sum().reset_index()
        table["sources"] = frame.groupby(keys).source_id.nunique().to_numpy()
    else:
        table = pd.DataFrame([dict(frame[counts].sum(), sources=frame.source_id.nunique())])
    table["partial_error_sentence_fraction"] = 1 - table.fully_covered_sentences / table.error_sentences
    table["later_span_error_word_fraction"] = table.later_span_words / table.error_words
    return table


def token_population(table):
    """Denominators use actual saved model-token labels, not lexical words."""
    error = table[table.gold.eq(1)]
    spans = error[["id", "span_start", "span_end"]].drop_duplicates()
    lengths = spans.span_end - spans.span_start
    first_tokens = int((error.span_index.eq(0) & error.offset.eq(0)).sum())
    after_onset = int(error.offset.gt(0).sum())
    later_spans = int(error.span_index.gt(0).sum())
    back = int(((error.offset + .5) / error.span_length >= .5).sum())
    runs = 0
    for _, group in table.sort_values(["id", "token"]).groupby("id"):
        labels = group.gold.astype(bool).to_numpy()
        runs += int((labels & ~np.r_[False, labels[:-1]]).sum())
    count = len(error)
    return dict(answers=int(table.id.nunique()), sources=int(table.source_id.nunique()),
        tokens=len(table), error_tokens=count, annotated_token_spans=len(spans), positive_runs=runs,
        first_error_tokens=first_tokens, within_span_continuation=after_onset,
        within_span_continuation_fraction=after_onset / count if count else None,
        tokens_after_answer_first_error=count - first_tokens,
        after_answer_first_error_fraction=(count - first_tokens) / count if count else None,
        later_span_error_tokens=later_spans, later_span_fraction=later_spans / count if count else None,
        back_half_error_tokens=back, back_half_fraction=back / count if count else None,
        median_span_tokens=float(lengths.median()) if len(lengths) else None,
        mean_span_tokens=float(lengths.mean()) if len(lengths) else None,
        max_span_tokens=int(lengths.max()) if len(lengths) else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks = {row["source_id"]: row["task_type"] for row in read_jsonl(args.sources)}
    rows = [response_counts(row, tasks[row["source_id"]]) for row in read_jsonl(args.responses)]
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "response_counts.csv", index=False)
    for name, keys in (("overall", []), ("by_task", ["task"]),
                       ("by_task_model_split", ["task", "model", "split"])):
        summarize_responses(frame, keys).to_csv(args.output / (name + ".csv"), index=False)
    write_json(args.output / "protocol.json", dict(
        source="https://github.com/ParticleMedia/RAGTruth", model_token_statistics=False,
        lexical_unit=WORD.pattern, sentence_unit="punctuation_v1; not manually annotated sentences",
        full_sentence_rule="every alphanumeric character covered by the annotation union",
        later_span_rule="character spans merged only when overlapping; second and subsequent spans"))
    print(summarize_responses(frame, []).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
