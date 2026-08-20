import math

import numpy as np

from experiments.attention_phenomenology.evaluation import (
    _paired_bootstrap_delta,
    _sample_bootstrap,
)


def test_bootstrap_reports_nan_when_a_smoke_subset_has_one_label_class():
    result = _sample_bootstrap(
        [np.zeros(4, dtype=np.int8)],
        [np.arange(4, dtype=np.float32)],
        replicates=5,
        seed=1,
    )

    assert all(math.isnan(value) for value in result.values())


def test_paired_bootstrap_reports_nan_when_no_valid_replicate_exists():
    labels = [np.zeros(4, dtype=np.int8)]
    score = [np.arange(4, dtype=np.float32)]

    result = _paired_bootstrap_delta(
        labels,
        score,
        score,
        replicates=5,
        seed=1,
    )

    assert all(math.isnan(value) for value in result.values())
