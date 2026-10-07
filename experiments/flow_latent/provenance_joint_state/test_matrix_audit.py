"""Scientific checks for matrix projection, discarded directions, and causality."""
import unittest

import numpy as np

from .matrix_audit import local_projection


class MatrixProjectionTests(unittest.TestCase):
    def test_dense_svd_agrees_and_full_reconstructs(self):
        generator = np.random.default_rng(42)
        values = generator.normal(size=(11, 17))
        direction = generator.normal(size=17)
        measured = local_projection(values, direction)
        for target in range(len(values)):
            window = values[max(0, target - 7):target + 1]
            temporal, singular, heads = np.linalg.svd(window, full_matrices=False)
            for rank in (1, 2, 4, 8):
                projected = (temporal[:, :rank] * singular[:rank]) @ heads[:rank]
                self.assertAlmostEqual(measured[rank]['score'][target],
                                       float(projected[-1] @ direction), places=10)
        np.testing.assert_allclose(measured[8]['score'], values @ direction, atol=1e-10)

    def test_high_variance_can_discard_entire_readout(self):
        values = np.array([[100., 0.], [0., 1.]])
        measured = local_projection(values, np.array([0., 1.]), ranks=(1, 2), window=2)
        self.assertGreater(measured[1]['variance'][-1], .999)
        self.assertAlmostEqual(measured[1]['readout'][-1], 0.)
        self.assertAlmostEqual(measured[2]['score'][-1], 1.)

    def test_no_future_and_each_row_keeps_own_readout(self):
        generator = np.random.default_rng(123)
        values = generator.normal(size=(10, 12))
        direction = generator.normal(size=12)
        original = local_projection(values, direction)
        changed = values.copy()
        changed[8:] += 500
        alternative = local_projection(changed, direction)
        for rank in (1, 2, 4, 8):
            np.testing.assert_allclose(original[rank]['score'][:8], alternative[rank]['score'][:8])
        self.assertGreater(np.std(original[8]['score']), 0.)


if __name__ == '__main__':
    unittest.main()
