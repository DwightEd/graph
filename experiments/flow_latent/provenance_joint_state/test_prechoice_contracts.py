"""Checks that forecasting features precede the predicted error and its history."""
import unittest

import numpy as np

from .prechoice_marker_audit import first_error_cohort, horizon_report
from .prechoice_states_pilot import prefix_before_candidate


class PrechoiceContractTests(unittest.TestCase):
    def test_target_and_future_words_cannot_change_prediction_prefix(self):
        case = dict(prompt_length=3, target=2, token_ids=[1, 2, 3, 4, 5, 6, 7])
        before = prefix_before_candidate(case)
        changed = dict(case, token_ids=[1, 2, 3, 4, 5, 98, 99])
        self.assertEqual(before, [1, 2, 3, 4, 5])
        self.assertEqual(before, prefix_before_candidate(changed))
        first_word = dict(case, target=0)
        self.assertEqual(prefix_before_candidate(first_word), [1, 2, 3])

    def test_clean_cohort_stops_at_first_error_even_when_error_repeats(self):
        row = dict(labels=np.array([0, 0, 0, 1, 1, 0, 1]))
        positions, truth = first_error_cohort(row, horizon=1)
        np.testing.assert_array_equal(positions, [1, 2, 3])
        np.testing.assert_array_equal(truth, [0, 0, 1])
        missing = dict(labels=np.array([1, 0, 1]))
        positions, _ = first_error_cohort(missing, horizon=1)
        self.assertEqual(len(positions), 0)

    def test_shift_does_not_use_current_or_future_marker(self):
        rows = [dict(id='wrong', labels=np.array([0, 0, 0, 1, 1])),
                dict(id='normal', labels=np.zeros(5, dtype=int))]
        features = {row['id']: {name: np.array([0., 0., 1., 0., 0.])
                    if row['id'] == 'wrong' else np.zeros(5)
                    for name in ('teacher', 'signed')} for row in rows}
        result = horizon_report(rows, features, horizon=1)
        for name in ('teacher', 'signed'):
            self.assertEqual(result['metrics'][name]['clean_prefix_first_error']['auroc'], 1.)
            # A post-error marker cannot affect the clean first-error forecast.
            features['wrong'][name][3:] = 1000.
        changed = horizon_report(rows, features, horizon=1)
        for name in ('teacher', 'signed'):
            self.assertEqual(changed['metrics'][name]['clean_prefix_first_error'],
                             result['metrics'][name]['clean_prefix_first_error'])
        self.assertEqual(result['first_errors'], 1)
        self.assertEqual(result['first_error_clean_tokens'], 7)


if __name__ == '__main__':
    unittest.main()
