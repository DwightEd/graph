import unittest
import warnings

import numpy as np

from attention_graph.one_class import (
    CalibratedMaxFusion,
    OneClassConfig,
    OneClassReference,
)


class OneClassReferenceTests(unittest.TestCase):
    def test_constant_reference_is_scored_without_pca_warning(self):
        zeros = np.zeros((6, 4), dtype=np.float32)
        bins = np.zeros(6, dtype=np.int16)

        with warnings.catch_warnings(record=True) as caught:
            reference = OneClassReference(
                OneClassConfig(position_bins=1, subspace_components=2)
            ).fit(zeros, bins, zeros, bins)
            result = reference.transform(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), [.5]
            )

        self.assertEqual(caught, [])
        self.assertGreater(float(result.subspace_residual[0]), 0.0)

    def test_calibration_outlier_does_not_change_fit_subspace(self):
        fit_values = np.asarray([
            [-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0],
            [1.0, 1.0], [2.0, 2.0],
        ], dtype=np.float32)
        fit_bins = np.zeros(len(fit_values), dtype=np.int16)
        ordinary_calibration = np.asarray([
            [-1.5, -1.5], [-.5, -.5], [.5, .5], [1.5, 1.5],
        ], dtype=np.float32)
        contaminated_calibration = ordinary_calibration.copy()
        contaminated_calibration[-1] = [1_000.0, -1_000.0]
        config = OneClassConfig(position_bins=1, subspace_components=1, seed=7)

        ordinary = OneClassReference(config).fit(
            fit_values, fit_bins, ordinary_calibration, fit_bins[:4]
        )
        contaminated = OneClassReference(config).fit(
            fit_values, fit_bins, contaminated_calibration, fit_bins[:4]
        )

        ordinary_state = ordinary.state()
        contaminated_state = contaminated.state()
        np.testing.assert_allclose(
            ordinary_state["pca_components"], contaminated_state["pca_components"]
        )
        np.testing.assert_allclose(
            ordinary_state["pca_mean"], contaminated_state["pca_mean"]
        )

    def test_independent_block_references_do_not_change_an_atomic_score(self):
        fit_values = np.asarray([
            [-1.0, 0.0], [-.5, .2], [0.0, .4], [.5, .6], [1.0, .8],
        ], dtype=np.float32)
        cal_values = np.asarray([
            [-.75, .1], [-.25, .3], [.25, .5], [.75, .7],
        ], dtype=np.float32)
        bins = np.zeros(len(fit_values), dtype=np.int16)
        cal_bins = np.zeros(len(cal_values), dtype=np.int16)
        config = OneClassConfig(position_bins=1, subspace_components=1, seed=11)
        atomic = OneClassReference(config).fit(
            fit_values, bins, cal_values, cal_bins
        )
        before = atomic.transform(np.asarray([[2.0, -1.0]], dtype=np.float32), [.5])

        unrelated = OneClassReference(config).fit(
            np.arange(15, dtype=np.float32).reshape(5, 3), bins,
            np.arange(12, dtype=np.float32).reshape(4, 3), cal_bins,
        )
        unrelated.transform(np.asarray([[100.0, -100.0, 4.0]], dtype=np.float32), [.5])
        after = atomic.transform(np.asarray([[2.0, -1.0]], dtype=np.float32), [.5])

        np.testing.assert_allclose(before.score, after.score)
        np.testing.assert_allclose(before.tail, after.tail)
        np.testing.assert_allclose(before.subspace_residual, after.subspace_residual)


class CalibratedMaxFusionTests(unittest.TestCase):
    def test_fusion_is_deterministic_and_bounded(self):
        calibration = {
            "mass": np.asarray([.1, .2, .4, .8], dtype=np.float32),
            "topology": np.asarray([.3, .1, .6, .7], dtype=np.float32),
        }
        values = {
            "mass": np.asarray([-.5, .2, .9, 4.0], dtype=np.float32),
            "topology": np.asarray([.4, .1, .5, 3.0], dtype=np.float32),
        }

        fusion = CalibratedMaxFusion().fit(calibration)
        first = fusion.transform(values)
        second = fusion.transform(values)

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first <= 1.0))
        self.assertGreater(float(first[-1]), float(first[0]))


if __name__ == "__main__":
    unittest.main()
