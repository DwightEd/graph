"""Check oracle budget ties, healthy-prefix scope and changed-alarm accounting."""
import unittest

import numpy as np

from .source_route_refine_eval import (alarm_counts, answer_masks,
                                       threshold_for_budget, transitions)


class SourceRouteRefineEvaluationTests(unittest.TestCase):
    def test_strict_budget_thresholds_never_exceed_budget_with_ties(self):
        normal = np.array([.9, .9, .8, .1])
        for budget in range(len(normal) + 1):
            threshold = threshold_for_budget(normal, budget)
            self.assertLessEqual(int((normal > threshold).sum()), budget)
        self.assertEqual(threshold_for_budget(normal, 1), .9)
        self.assertEqual(int((normal > threshold_for_budget(normal, 1)).sum()), 0)
        self.assertEqual(int((normal > threshold_for_budget(normal, 2)).sum()), 2)
        self.assertIsNone(threshold_for_budget([], 0))

    def test_strict_first_error_excludes_post_error_normal_tokens(self):
        labels = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0])
        firsts = np.zeros(len(labels), dtype=bool)
        firsts[[3, 7]] = True
        onsets = np.zeros_like(firsts)
        onsets[[3, 5, 7]] = True
        pack = dict(labels=labels, firsts=firsts, onsets=onsets)
        records = [dict(id='normal', packed_start=0, packed_stop=2),
                   dict(id='error_a', packed_start=2, packed_stop=6),
                   dict(id='error_b', packed_start=6, packed_stop=9)]
        clean, first, answers = answer_masks(pack, records)
        np.testing.assert_array_equal(np.flatnonzero(clean), [0, 1, 2, 6])
        np.testing.assert_array_equal(np.flatnonzero(clean | first), [0, 1, 2, 3, 6, 7])
        scores = np.array([.1, .2, .8, .9, .99, .9, .1, .8, .99])
        counts = alarm_counts(pack, scores, .5, clean, first, answers)
        self.assertEqual(counts['first_errors_detected'], 2)
        self.assertEqual(counts['first_errors_without_prior_alarm'], 1)
        self.assertEqual(counts['clean_normal_false_alarms'], 1)
        self.assertEqual(counts['normal_answer_alarms'], 0)

    def test_transition_counts_and_strict_threshold_identity(self):
        labels = np.array([1, 1, 0, 0, 1, 0])
        baseline = np.array([.8, .2, .7, .3, .5, .5])
        candidate = np.array([.2, .8, .3, .7, .5, .5])
        counts = transitions(labels, baseline, candidate, .5, .5)
        self.assertEqual(counts['new_tp'], 1)
        self.assertEqual(counts['new_fp'], 1)
        self.assertEqual(counts['lost_tp'], 1)
        self.assertEqual(counts['removed_fp'], 1)
        unchanged = transitions(labels, baseline, baseline, .5, .5)
        self.assertEqual(sum(unchanged[name] for name in
                             ('new_tp', 'new_fp', 'lost_tp', 'removed_fp')), 0)


if __name__ == '__main__':
    unittest.main()
