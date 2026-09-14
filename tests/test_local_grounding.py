import unittest
from next_iteration.local_grounding import units, local_sentence, messages, common_prefix, align_scores


class LocalGroundingTests(unittest.TestCase):
    def test_units_and_offsets(self):
        self.assertEqual(units('  Red box.\nBlue'), [(2, 5), (6, 10), (11, 15)])

    def test_sentence_marks_exact_target(self):
        text = 'Red box. Blue box is empty. End.'
        self.assertIn('[[TARGET]]Blue[[/TARGET]]', local_sentence(text, 9, 13))
        self.assertNotIn('End.', local_sentence(text, 9, 13))

    def test_full_response_context(self):
        result = messages('source', 'task', 'first. second.', (0, 6))
        self.assertIn('first. second.', result[1]['content'])
        self.assertIn('Only SOURCE is evidence', result[0]['content'])

    def test_prefix_never_consumes_whole_query(self):
        self.assertEqual(common_prefix([[1, 2, 3], [1, 2, 4]]), 2)
        self.assertEqual(common_prefix([[1, 2, 3]]), 2)

    def test_every_token_gets_score(self):
        spans = [(1, 4), (5, 8)]
        offsets = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 9)]
        self.assertEqual(align_scores(offsets, spans, [-2, 3]), [-2, -2, -2, 3, 3, 3])

    def test_overlap_max_and_whitespace(self):
        self.assertEqual(align_scores([(0, 9)], [(0, 2), (4, 8)], [4, -5]), [4])


if __name__ == '__main__':
    unittest.main()
