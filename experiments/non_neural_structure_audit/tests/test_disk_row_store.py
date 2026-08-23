import shutil

import numpy as np
import pytest

from experiments.disk_row_store import DiskRowStore, FieldSpec


def test_row_store_appends_one_sample_without_concatenating_prior_rows(tmp_path):
    store = DiskRowStore(
        tmp_path / "rows",
        capacity=5,
        fields={
            "label": FieldSpec(np.dtype("int8")),
            "map": FieldSpec(np.dtype("float32"), (2, 3)),
        },
    )
    first = store.append(
        {
            "label": np.asarray([0, 1], dtype=np.int8),
            "map": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        }
    )
    second = store.append(
        {
            "label": np.asarray([1], dtype=np.int8),
            "map": np.ones((1, 2, 3), dtype=np.float32),
        }
    )

    assert first == slice(0, 2)
    assert second == slice(2, 3)
    assert isinstance(store.view("map", first), np.memmap)
    np.testing.assert_array_equal(store.view("label"), [0, 1, 1])
    assert store.view("map").shape == (3, 2, 3)
    store.close()


def test_row_store_rejects_a_bad_sample_before_advancing(tmp_path):
    store = DiskRowStore(
        tmp_path / "rows",
        capacity=2,
        fields={"value": FieldSpec(np.dtype("float32"), (2,))},
    )

    with pytest.raises(TypeError, match="dtype"):
        store.append({"value": np.ones((1, 2), dtype=np.float64)})

    assert store.rows == 0
    store.close()


def test_row_store_context_closes_windows_mappings_on_failure(tmp_path):
    root = tmp_path / "rows"
    with (
        pytest.raises(RuntimeError, match="audit failed"),
        DiskRowStore(
            root,
            capacity=1,
            fields={"value": FieldSpec(np.dtype("float32"))},
        ) as store,
    ):
        store.append({"value": np.asarray([1.0], dtype=np.float32)})
        raise RuntimeError("audit failed")

    shutil.rmtree(root)
    assert not root.exists()
