import numpy as np

from experiments.non_neural_structure_audit.bounded_ensemble import (
    DiskBackedAUPRC,
)
from experiments.non_neural_structure_audit.statistics import binary_metrics


def test_disk_backed_ensemble_matches_pooled_auprc_definition(tmp_path):
    labels = [np.asarray([0, 1, 0]), np.asarray([1, 0])]
    real = [
        np.asarray([[0.1, 0.8], [0.9, 0.2], [0.3, 0.6]]),
        np.asarray([[0.7, 0.4], [0.2, 0.9]]),
    ]
    null = [
        np.stack((real[0][::-1], real[0] * 0.5, real[0])),
        np.stack((real[1][::-1], real[1] * 0.5, real[1])),
    ]

    accumulator = DiskBackedAUPRC(
        tmp_path / "ensemble.dat",
        capacity=5,
        replicates=3,
        relations=2,
    )
    for current_labels, current_real, current_null in zip(
        labels, real, null, strict=True
    ):
        accumulator.add(current_labels, current_real, current_null)
    result = accumulator.finish()
    accumulator.close()

    pooled_labels = np.concatenate(labels)
    pooled_real = np.concatenate(real)
    pooled_null = np.concatenate(null, axis=1)
    expected_real = np.asarray(
        [
            binary_metrics(pooled_labels, pooled_real[:, index])["auprc"]
            for index in range(2)
        ]
    )
    expected_null = np.asarray(
        [
            [
                binary_metrics(pooled_labels, pooled_null[replicate, :, index])["auprc"]
                for index in range(2)
            ]
            for replicate in range(3)
        ]
    )

    np.testing.assert_allclose(result.real, expected_real)
    np.testing.assert_allclose(result.null, expected_null)


def test_masked_add_matches_explicitly_selected_rows(tmp_path):
    labels = np.asarray([0, 1, 0, 1])
    real = np.arange(8, dtype=np.float32).reshape(4, 2)
    null = np.stack((real[::-1], real * 0.5))
    mask = np.asarray([True, False, True, False])
    accumulator = DiskBackedAUPRC(
        tmp_path / "masked.dat", capacity=2, replicates=2, relations=2
    )

    accumulator.add_masked(labels, real, null, mask)

    np.testing.assert_array_equal(accumulator.labels[:2], labels[mask])
    np.testing.assert_array_equal(accumulator.real[:2], real[mask])
    np.testing.assert_array_equal(accumulator.null[:, :2], null[:, mask])
    accumulator.close()
