import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from experiments.non_neural_structure_audit.bounded_ensemble import EnsembleAUPRC
from experiments.non_neural_structure_audit.bounded_samples import DiskBackedSamples
from experiments.non_neural_structure_audit.evaluation_data import EvaluationBundle


def test_compact_sample_matrices_are_sliced_per_sample_and_storage_is_released(
    tmp_path,
):
    root = tmp_path / "samples"
    store = DiskBackedSamples(root, capacity=5, relations=2)
    matrix = np.arange(6, dtype=np.float32).reshape(3, 2)

    arrays = store.add(
        labels=np.asarray([0, 1, 0], dtype=np.int8),
        eligible=np.asarray([True, True, False]),
        relation=matrix,
        layer_order_real=matrix + 1,
        endpoint_null=matrix + 2,
        layer_shuffle=matrix + 3,
    )

    np.testing.assert_array_equal(arrays[0], [0, 1, 0])
    np.testing.assert_array_equal(arrays[2], matrix)
    store.close()
    shutil.rmtree(root)
    assert not root.exists()


def test_evaluation_bundle_cleans_memmaps_on_an_audit_failure(tmp_path):
    temporary = TemporaryDirectory(dir=tmp_path)
    temporary_path = temporary.name
    store = DiskBackedSamples(temporary_path + "/samples", capacity=1, relations=1)
    metrics = EnsembleAUPRC(real=np.zeros(1), null=np.zeros((1, 1)))
    bundle = EvaluationBundle([], metrics, metrics, store, temporary)

    with pytest.raises(RuntimeError, match="audit failed"), bundle:
        raise RuntimeError("audit failed")

    assert not Path(temporary_path).exists()


def test_evaluation_bundle_cleans_scratch_when_store_close_reports_an_error(
    tmp_path, monkeypatch
):
    temporary = TemporaryDirectory(dir=tmp_path)
    temporary_path = temporary.name
    store = DiskBackedSamples(temporary_path + "/samples", capacity=1, relations=1)
    metrics = EnsembleAUPRC(real=np.zeros(1), null=np.zeros((1, 1)))
    bundle = EvaluationBundle([], metrics, metrics, store, temporary)
    close = store.close

    def close_then_report_error():
        close()
        raise OSError("flush reported an error")

    monkeypatch.setattr(store, "close", close_then_report_error)

    with pytest.raises(OSError, match="flush reported an error"):
        bundle.close()

    assert not Path(temporary_path).exists()
