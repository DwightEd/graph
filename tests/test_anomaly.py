import inspect
import unittest

import torch

from anomaly import ConditionalStudentMixture, EmpiricalTailCalibrator


def sample_embeddings():
    return [
        torch.tensor(
            [[0.0, 0.1], [0.2, -0.1], [0.4, 0.0], [0.3, 0.2]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[-0.2, 0.0], [0.0, 0.2], [0.1, 0.3], [0.3, 0.1], [0.5, 0.2]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.1, -0.2], [0.2, 0.0], [0.0, 0.1]],
            dtype=torch.float32,
        ),
    ]


class ConditionalStudentMixtureTests(unittest.TestCase):
    def test_scores_variable_length_samples_without_looking_into_the_future(self):
        model = ConditionalStudentMixture(
            num_components=2,
            contamination=0.05,
            variance_floor=1e-4,
        ).fit(sample_embeddings())
        probe = sample_embeddings()[1]

        original = model.score([probe])[0]
        changed = probe.clone()
        changed[3:] = torch.tensor([[30.0, -20.0], [-40.0, 50.0]])
        rescored = model.score([changed])[0]

        self.assertEqual(original.shape, (probe.shape[0],))
        self.assertEqual(rescored.shape, (probe.shape[0],))
        torch.testing.assert_close(original[:3], rescored[:3], atol=1e-6, rtol=1e-6)

    def test_multiple_components_contamination_and_variance_floor_stay_finite(self):
        samples = [
            torch.zeros((4, 3), dtype=torch.float32),
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor(
                [[0.0, 0.0, 0.0], [1e4, -1e4, 5e3]], dtype=torch.float32
            ),
        ]
        model = ConditionalStudentMixture(
            num_components=3,
            contamination=0.10,
            variance_floor=1e-3,
        ).fit(samples)

        scores = model.score(samples)

        self.assertEqual([score.shape for score in scores], [(4,), (2,), (2,)])
        self.assertTrue(all(torch.isfinite(score).all() for score in scores))

    def test_fit_and_score_have_no_label_input(self):
        model = ConditionalStudentMixture(
            num_components=2,
            contamination=0.05,
            variance_floor=1e-4,
        )

        for method in (model.fit, model.score):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("labels", parameters)
            self.assertFalse(
                any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )


class EmpiricalTailCalibratorTests(unittest.TestCase):
    def test_tail_probabilities_use_only_the_fit_scores(self):
        calibrator = EmpiricalTailCalibrator().fit(
            torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
        )

        alone = calibrator.transform(torch.tensor([3.0]))
        with_unrelated_queries = calibrator.transform(
            torch.tensor([3.0, 100.0, -100.0])
        )

        torch.testing.assert_close(alone, torch.tensor([0.5]))
        torch.testing.assert_close(
            with_unrelated_queries, torch.tensor([0.5, 0.25, 1.0])
        )


if __name__ == "__main__":
    unittest.main()
