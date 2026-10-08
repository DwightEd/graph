"""Scientific checks for source-generated candidate-aware compatibility targets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import source_programs
from .source_programs import (balanced_source_settings, build_source_programs, common_prefix_length,
    decode_source_spans, duplicate_owner_controls, literal_quote_candidates,
    program_coverage, program_token_targets, source_controls)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(value) for value in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '<user>' + messages[0]['content'] + '</user><assistant>'

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i + 1) for i in range(len(text))])


class SourceProgramTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.quotes = [dict(text='same blue lake', ids=self.tokenizer.encode('same blue lake')),
                       dict(text='same green hill', ids=self.tokenizer.encode('same green hill'))]

    def test_disconnected_source_spans_do_not_create_fake_quotes(self):
        ids = self.tokenizer.encode('red HIDDEN blue')
        mask = [i < 3 or i >= 11 for i in range(len(ids))]
        spans = decode_source_spans(self.tokenizer, ids, mask)
        self.assertEqual([row['text'] for row in spans], ['red', 'blue'])
        self.assertEqual(literal_quote_candidates(self.tokenizer, spans, 20), [])

    def test_quotes_are_literal_substrings_and_have_whole_word_boundaries(self):
        text = 'A blue lake is near a green hill.'
        spans = [dict(start=5, stop=5 + len(text), text=text)]
        quotes = literal_quote_candidates(self.tokenizer, spans, 18)
        self.assertGreater(len(quotes), 2)
        for quote in quotes:
            self.assertEqual(text[quote['char_start']:quote['char_stop']], quote['text'])
            self.assertEqual(self.tokenizer.decode(quote['ids']), quote['text'])
            self.assertLessEqual(len(quote['ids']), 18)

    def test_same_candidate_support_reverses_only_with_source_owner(self):
        records = source_controls(self.tokenizer, '17', 'fit', self.quotes)
        self.assertEqual(len(records), 4)
        original, swapped = records[0], records[2]
        self.assertEqual(original['answer_ids'], swapped['answer_ids'])
        self.assertEqual(original['query_owner'], swapped['query_owner'])
        self.assertNotEqual(original['source_text'], swapped['source_text'])
        self.assertTrue(all(value == 0 for value in original['labels']))
        self.assertTrue(all(label == 1 for label, valid in zip(swapped['labels'], swapped['valid']) if valid))
        for record in records:
            facts = json.loads(record['source_text'])
            self.assertEqual(set(row['quote'] for row in facts.values()), {quote['text'] for quote in self.quotes})

    def test_shared_prefix_not_conflicting_label_and_candidate_needed_at_divergence(self):
        correct, wrong = source_controls(self.tokenizer, '17', 'fit', self.quotes)[:2]
        divergence = correct['first_divergence']
        self.assertEqual(divergence, len('same '))
        self.assertEqual(correct['answer_ids'][:divergence], wrong['answer_ids'][:divergence])
        self.assertTrue(all(not valid for valid in wrong['valid'][:divergence]))
        self.assertEqual(correct['labels'][divergence], 0)
        self.assertEqual(wrong['labels'][divergence], 1)
        self.assertNotEqual(correct['answer_ids'][divergence], wrong['answer_ids'][divergence])
        self.assertTrue(correct['candidate_input_required'])

    def test_each_token_compatibility_checks_prefix_not_mean_label(self):
        targets = program_token_targets([1, 2, 7, 4], [1, 2, 3, 4], 2)
        self.assertEqual(targets['compatibility'], [1, 1, 0, 0])
        self.assertEqual(targets['valid'], [False, False, True, True])
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 7]), 2)

    def test_duplicate_owner_same_value_remains_compatible(self):
        records = duplicate_owner_controls(self.tokenizer, '17', 'fit', self.quotes[0])
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record['facts']['owner_a'], record['facts']['owner_b'])
            self.assertTrue(all(value == 1 for value in record['compatibility']))
            self.assertTrue(all(record['valid']))

    def test_source_mask_covers_only_json_source(self):
        record = source_controls(self.tokenizer, '17', 'fit', self.quotes)[0]
        masked_ids = [token for token, included in zip(record['prompt_with_source'], record['source_mask']) if included]
        self.assertEqual(self.tokenizer.decode(masked_ids), record['source_text'])
        self.assertFalse(record['source_mask'][0])
        self.assertFalse(record['source_mask'][-1])

    def test_provider_reads_sources_without_natural_response_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            source_ids = self.tokenizer.encode('A blue lake is near a green hill and a red valley.')
            payload = dict(prompt_with_source=source_ids, source_mask=[True] * len(source_ids))
            (cache / 'source.json').write_text(json.dumps(payload))
            sources = [dict(source_id='17', partition='fit', source_file='source.json')]
            with patch.object(source_programs, 'select_train_sources',
                       return_value=({}, sources)):
                records, coverage = build_source_programs(cache, self.tokenizer, value_tokens=18)
            self.assertEqual(len(records), 4)
            self.assertFalse(coverage['natural_answers_read'])
            self.assertFalse(coverage['natural_annotations_read'])
            self.assertEqual(coverage['partitions']['fit']['sources'], 1)

    def test_owner_order_counterbalanced_and_first_field_not_label(self):
        records = [record for source in range(32)
                   for record in source_controls(self.tokenizer, str(source), 'fit', self.quotes)]
        settings = {(row['query_owner'], tuple(row['source_key_order'])) for row in records}
        self.assertEqual(len(settings), 4)
        first_labels = {row['proposal_index'] != row['correct_index']
                        for row in records if row['candidate_is_first']}
        self.assertEqual(first_labels, {False, True})
        for row in records:
            self.assertEqual(row['correct_answer_text'], row['facts'][row['query_owner']]['quote'])
        for source in range(32):
            group = [row for row in records if row['source_id'] == str(source)]
            self.assertEqual(len({row['query_owner'] for row in group}), 1)
            self.assertEqual(len({row['template_id'] for row in group}), 1)
            for proposal in (0, 1):
                pair = [row for row in group if row['proposal_index'] == proposal]
                self.assertEqual(pair[0]['answer_ids'], pair[1]['answer_ids'])
                self.assertNotEqual(pair[0]['labels'][pair[0]['first_divergence']],
                                    pair[1]['labels'][pair[1]['first_divergence']])

    def test_template_holdout_and_shortcut_coverage_computed_from_records(self):
        sources = [dict(source_id=str(source), partition='fit' if source < 32 else 'dev')
                   for source in range(64)]
        records = [record for source in sources for record in source_controls(
            self.tokenizer, source['source_id'], source['partition'], self.quotes)]
        coverage = program_coverage(records, sources)
        self.assertTrue(coverage['template_holdout'])
        self.assertEqual(coverage['unseen_dev_templates'], ['exact_field'])
        self.assertTrue(coverage['source_disjoint'])
        for partition in ('fit', 'dev'):
            diagnostics = coverage['partitions'][partition]['shortcut_diagnostics']
            self.assertEqual(diagnostics['candidate_first_fraction'], .5)
            self.assertEqual(diagnostics['incompatible_record_fraction'], .5)
            self.assertGreater(diagnostics['query_first_fraction'], 0)
            self.assertLess(diagnostics['query_first_fraction'], 1)
        fit_templates = coverage['partitions']['fit']['template_record_counts']
        self.assertNotIn('exact_field', fit_templates)

    def test_roster_balancing_removes_first_position_shortcut_exactly(self):
        sources = [dict(source_id=str(source), partition='fit' if source < 96 else 'dev')
                   for source in range(128)]
        settings = balanced_source_settings(sources)
        self.assertEqual(settings, balanced_source_settings(sources[::-1]))
        records = [record for source in sources for record in source_controls(self.tokenizer,
            source['source_id'], source['partition'], self.quotes, settings[source['source_id']])]
        coverage = program_coverage(records, sources)
        for partition in ('fit', 'dev'):
            diagnostics = coverage['partitions'][partition]['shortcut_diagnostics']
            self.assertEqual(diagnostics['query_first_fraction'], .5)
            self.assertEqual(diagnostics['incompatible_given_candidate_first'], .5)
            self.assertEqual(diagnostics['incompatible_given_candidate_later'], .5)
            self.assertEqual(len(set(diagnostics['query_owner_counts'].values())), 1)
            self.assertEqual(len(set(diagnostics['key_order_counts'].values())), 1)


if __name__ == '__main__':
    unittest.main()
