import numpy as np
import pandas as pd

from experiments.ragtruth_flow.grounding_report import (
    grouped_metrics,
    macro_within_answer,
)


def sample_table():
    return pd.DataFrame(dict(
        id=["a", "a", "a", "b", "b", "b"],
        task=["QA"] * 3 + ["Summary"] * 3,
        generator=["g"] * 6,
        gold=[0, 1, 1, 0, 0, 1],
        previous_gold=[0, 0, 1, 0, 0, 0],
        sentence_start=[False, True, False, False, True, True],
        raw_surprise=[0., 2., 1., 0., 1., 2.],
        head_contrast_surprise=[0., 2., 1., 0., 1., 2.],
        source_gain=[0., 1., 1., 0., 0., 1.],
        history_gain=[0., 0., 1., 0., 1., 0.],
        grounding_balance=[0., 1., 0., 0., -1., 1.],
        self_jump=[0., 2., 1., 0., 2., 2.],
    ))


def test_grouped_metrics_preserves_task_scope():
    result = grouped_metrics(sample_table(), high_jump=1.5, group="task")
    qa = result[
        (result.task == "QA")
        & (result.scope == "previous_gold_0")
        & (result.score == "raw_surprise")
    ].iloc[0]
    assert qa.tokens == 2
    assert qa.positives == 1
    assert np.isclose(qa.auroc, 1.0)


def test_macro_within_answer_reports_mixed_answer_count():
    result = macro_within_answer(sample_table(), high_jump=1.5)
    row = result[
        (result.scope == "all")
        & (result.score == "raw_surprise")
    ].iloc[0]
    assert row.mixed_answers == 2
    assert np.isfinite(row.macro_auroc)
