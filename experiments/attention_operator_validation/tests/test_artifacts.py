from dataclasses import replace

import numpy as np
import pytest

from experiments.attention_operator_validation.artifacts import (
    load_feature_table,
    save_feature_table,
)

from .helpers import make_table


def test_feature_artifact_is_byte_deterministic(tmp_path):
    table = make_table()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    save_feature_table(first, table)
    save_feature_table(second, table)

    assert first.read_bytes() == second.read_bytes()
    loaded = load_feature_table(first)
    assert loaded.feature_names == table.feature_names
    assert np.array_equal(loaded.feature, table.feature)


def test_feature_artifact_rejects_wrong_implementation():
    table = make_table()
    metadata = {**table.metadata, "implementation_sha256": "0" * 64}

    with pytest.raises(ValueError, match="running code"):
        replace(table, metadata=metadata).validate()


def test_feature_artifact_rejects_duplicate_names_and_infinity():
    table = make_table()
    with pytest.raises(ValueError, match="unique"):
        replace(
            table,
            feature_names=(table.feature_names[0],) * len(table.feature_names),
        ).validate()

    invalid = table.feature.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="never infinite"):
        replace(table, feature=invalid).validate()
