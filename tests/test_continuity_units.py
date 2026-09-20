"""Guard annotation units, held-out thresholds and delay denominators."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.charm_structure_audit.annotation_population import response_counts, token_population
from experiments.charm_structure_audit.continuity_units import (
    pair_delays, run, sentence_metrics, sentence_scores, summarize_delays,
)
from experiments.charm_structure_audit.unit_data import read_tokens


def test_official_spans_need_not_cover_a_sentence():
    record = dict(id="a", source_id="s", model="m", split="test", response="Red cats run. Dogs sleep.",
                  labels=[dict(start=4, end=8, text="cats")])
    counts = response_counts(record, "QA")
    assert counts["sentences"] == 2 and counts["error_sentences"] == 1
    assert counts["fully_covered_sentences"] == 0 and counts["error_words"] == 1


def mapped_table(tmp_path):
    words = ["A", " B", " C.", " D", " E."]
    table = pd.DataFrame(dict(id="a", source_id="s", token=np.arange(5), text=words,
                              gold=[1, 1, 1, 0, 1], score=[.9, .8, .7, .1, .6]))
    record = dict(id="a", source_id="s", response="".join(words),
                  labels=[dict(start=0, end=3), dict(start=3, end=6), dict(start=8, end=11)])
    path = tmp_path / "tokens.csv"
    table.to_csv(path, index=False)
    return read_tokens(path, ["score"], {"a": record}), record, path


def test_adjacent_annotations_and_later_spans_are_not_continuation(tmp_path):
    table, record, path = mapped_table(tmp_path)
    summary = token_population(table)
    assert summary["annotated_token_spans"] == 3 and summary["positive_runs"] == 2
    assert summary["within_span_continuation"] == 1
    assert summary["later_span_error_tokens"] == 2
    assert summary["tokens_after_answer_first_error"] == 3
    record["response"] += "x"
    with pytest.raises(ValueError, match="concatenate"):
        read_tokens(path, ["score"], {"a": record})


def test_sentence_pooling_does_not_use_labels_and_threshold_uses_calibration_only(tmp_path):
    table, _, _ = mapped_table(tmp_path)
    scored = sentence_scores(table, ["score"])
    changed = sentence_scores(table.assign(gold=0), ["score"])
    np.testing.assert_array_equal(scored.score, changed.score)
    calibration = changed.copy()
    calibration["score"] = .5
    result = sentence_metrics(scored, calibration, .05)
    assert result.threshold.eq(.5).all()
    missing = sentence_metrics(scored, None, .05)
    assert missing.threshold.isna().all() and missing.recall.isna().all()


def test_delayed_pairs_keep_short_span_attrition_explicit():
    table = pd.DataFrame(dict(id="a", source_id="s", token=np.arange(12),
        gold=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], score=np.arange(12, dtype=float)))
    pairs = [dict(tier="cluster", id="a", source_id="s", error_start=0, normal_start=4, length=4),
             dict(tier="cluster", id="a", source_id="s", error_start=8, normal_start=10, length=2)]
    rows = pair_delays(table, pairs, ["score"], [1, 2, 4])
    summary = summarize_delays(rows, [1, 2, 4], 10)
    fixed = summary[summary.cohort.eq("same_pairs_all_k")]
    assert fixed.pairs.eq(1).all()
    early = summary[summary.cohort.eq("eligible_at_k") & summary.observed_tokens.eq(1)]
    assert early.pairs.eq(2).all()
    # Altering an unseen later score cannot change the k=1 measurements.
    table.loc[3, "score"] = 1000
    changed = pair_delays(table, pairs, ["score"], [1, 2, 4])
    np.testing.assert_array_equal(rows[rows.observed_tokens.eq(1)].gap, changed[changed.observed_tokens.eq(1)].gap)


def test_calibration_cannot_reuse_test_sources(tmp_path):
    _, record, path = mapped_table(tmp_path)
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(record) + "\n", encoding="utf-8")
    args = SimpleNamespace(responses=responses, tokens=path, methods=["score"], calibration=path)
    with pytest.raises(ValueError, match="sources overlap"):
        run(args)
