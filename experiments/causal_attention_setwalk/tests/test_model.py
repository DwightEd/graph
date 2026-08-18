import unittest

import numpy as np

from experiments.causal_attention_setwalk.model import (
    ReferenceConfig,
    anomaly_score,
    fit_reference_model,
)


class SetWalkReferenceTests(unittest.TestCase):
    def test_reference_is_label_free_and_scores_far_points_higher(self):
        rng = np.random.default_rng(7)
        train = rng.normal(0.0, 0.2, size=(80, 4)).astype(np.float32)
        position = np.tile(np.arange(8), 10).astype(np.int16)
        task = np.asarray(["QA"] * len(train))
        model = fit_reference_model(
            train,
            position,
            task,
            ReferenceConfig(position_bins=8, min_task_bin_rows=4),
        )
        test = np.asarray([[0.0] * 4, [8.0] * 4], dtype=np.float32)
        score = anomaly_score(
            test,
            np.asarray([0, 0]),
            np.asarray(["QA", "QA"]),
            model,
        )
        self.assertGreater(float(score[1]), float(score[0]))


if __name__ == "__main__":
    unittest.main()

