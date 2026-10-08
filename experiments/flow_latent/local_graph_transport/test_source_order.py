"""CPU mathematical, document-identity and causal-prefix contracts."""
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .source_order import (CONDITIONS, ORDERS, compile_orders, complete_readout, document_blocks,
                           logit_readout, order_fields, token_cut, verify_readout)


class SourceOrderTests(unittest.TestCase):
    def test_full_bijection_preserves_token_chunks_and_original_document_identity(self):
        class TailTokenizer:
            """Minimal BPE surrogate with punctuation+separator merged tokens."""
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return 'USER:\n' + messages[0]['content'] + '\nASSISTANT:'

            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                ids, offsets, index = [], [], 0
                while index < len(text):
                    width = 3 if text.startswith('.\n\n', index) else 1
                    ids.append(10000 if width == 3 else ord(text[index]))
                    offsets.append((index, index + width))
                    index += width
                return dict(input_ids=ids, offset_mapping=offsets)
        tokenizer = TailTokenizer()
        prompt = 'Question\npassages:\npassage 1: alpha.\n\npassage 2: beta.\n\npassage 3: gamma.\n\nIn case the passages lack answers\noutput:'
        text = tokenizer.apply_chat_template([dict(role='user', content=prompt)])
        encoded = tokenizer(text)
        start = text.index(prompt) + prompt.index('passage 1:')
        stop = text.index(prompt) + prompt.index('\nIn case the passages')
        mask = [(a < stop and b > start) for a, b in encoded['offset_mapping']]
        compiled = compile_orders(tokenizer, prompt, encoded['input_ids'], mask)
        original = np.asarray(encoded['input_ids'])
        native = compiled['variants']['native']
        for name in CONDITIONS:
            variant = compiled['variants'][name]
            n2o, o2n = np.asarray(variant['new_to_old']), np.asarray(variant['old_to_new'])
            np.testing.assert_array_equal(np.asarray(variant['prompt_ids']), original[n2o])
            np.testing.assert_array_equal(n2o[o2n], np.arange(len(original)))
            np.testing.assert_array_equal(np.asarray(variant['original_doc_id']), np.asarray(native['original_doc_id'])[n2o])
            for doc in (1, 2, 3):
                original_chunk = original[np.asarray(native['original_doc_id']) == doc]
                moved_chunk = np.asarray(variant['prompt_ids'])[np.asarray(variant['original_doc_id']) == doc]
                np.testing.assert_array_equal(moved_chunk, original_chunk)
                np.testing.assert_array_equal(np.asarray(variant['original_doc_token_offset'])[np.asarray(variant['original_doc_id']) == doc], np.arange(len(original_chunk)))
        wrong = encoded['input_ids'].copy()
        wrong[0] += 1
        with self.assertRaises(ValueError):
            compile_orders(tokenizer, prompt, wrong, mask)

    def test_document_headers_and_tail_separators_travel_with_original_id(self):
        prompt = 'Question\npassages:\npassage 1: alpha.\n\npassage 2: beta.\n\npassage 3: gamma.\n\nIn case the passages lack answers\noutput:'
        parts = document_blocks(prompt)
        texts = [parts['prefix'] + ''.join(parts['blocks'][j] for j in order) + parts['suffix'] for order in ORDERS]
        self.assertEqual(texts[0], prompt)
        self.assertLess(texts[1].index('passage 2:'), texts[1].index('passage 3:'))
        self.assertLess(texts[1].index('passage 3:'), texts[1].index('passage 1:'))
        self.assertEqual(parts['blocks'][0], 'passage 1: alpha.\n\n')
        for text in texts:
            self.assertTrue(text.startswith(parts['prefix']))
            self.assertTrue(text.endswith(parts['suffix']))

    def test_duplicate_headers_or_normalized_separator_fail(self):
        prompt = 'passage 1: a\n\npassage 2: b\n\npassage 3: c\n\nIn case the passages ...'
        with self.assertRaises(ValueError):
            document_blocks(prompt.replace('passage 3:', 'passage 2:'))
        with self.assertRaises(ValueError):
            document_blocks(prompt.replace('a\n\n', 'a\n'))

    def test_cut_rejects_splitting_body_and_separator_of_one_token(self):
        offsets = [(0, 3), (3, 8), (8, 10)]
        self.assertEqual(token_cut(offsets, 8), 2)
        with self.assertRaises(ValueError):
            token_cut(offsets, 6)

    def test_frozen_rival_is_distinct_and_does_not_reselect_after_permutation(self):
        native = torch.tensor([[5., 4., 1.], [0., 2., 3.]])
        original = logit_readout(native, [0, 1])
        np.testing.assert_array_equal(original['rival_id'], [1, 2])
        changed = logit_readout(torch.tensor([[4., 0., 8.], [9., 3., 0.]]), [0, 1], original['rival_id'])
        np.testing.assert_allclose(changed['margin'], [4., 3.])
        np.testing.assert_array_equal(changed['rival_id'], [1, 2])
        with self.assertRaises(ValueError):
            logit_readout(native, [0, 1], [0, 2])

    def test_actual_logp_full_normalization_and_logit_translation_invariance(self):
        logits = torch.tensor([[2., 1., 0.]], dtype=torch.float64)
        original = logit_readout(logits, [1])
        expected = 1. - np.log(np.exp(2.) + np.exp(1.) + 1.)
        self.assertAlmostEqual(float(original['actual_logp'][0]), expected)
        translated = logit_readout(logits + 100., [1], original['rival_id'])
        np.testing.assert_allclose(original['actual_logp'], translated['actual_logp'], atol=1e-12)
        np.testing.assert_array_equal(original['margin'], translated['margin'])

    def test_highest_nonactual_exact_tie_uses_first_vocabulary_id(self):
        readout = logit_readout(torch.tensor([[2., 2., 1., 2.]]), [1])
        np.testing.assert_array_equal(readout['rival_id'], [0])

    def test_primary_range_uses_three_conditions_without_token_averaging(self):
        values = np.array([[1., 7., 9.], [4., 2., 9.], [-3., 3., 9.]])
        fields = order_fields(values, values * 2)
        np.testing.assert_array_equal(fields['margin_range'], [14., 10., 0.])
        np.testing.assert_array_equal(fields['actual_logp_range'], [7., 5., 0.])
        np.testing.assert_array_equal(fields['delta_margin'], (values[1:] - values[0]) * 2)
        with self.assertRaises(ValueError):
            order_fields(values[:2], values[:2])

    def test_identity_gate_does_not_tolerate_changed_ids(self):
        arrays = dict(actual_id=np.array([3]), rival_id=np.array([2]), query_position=np.array([5]),
            actual_logp=np.array([-1.]), margin=np.array([2.]))
        self.assertTrue(verify_readout(arrays, arrays, 1e-6)['passed'])
        changed = dict(arrays, rival_id=np.array([1]))
        with self.assertRaises(ValueError):
            verify_readout(changed, arrays, 100.)
        with self.assertRaises(ValueError):
            verify_readout(dict(arrays, margin=np.array([2.002])), arrays, .001)

    def test_complete_prefill_prediction_rows_and_no_future_tokens(self):
        torch.manual_seed(719)
        config = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            attention_dropout=0., max_position_embeddings=128)
        model = LlamaForCausalLM(config).float().eval().requires_grad_(False)
        prompt, answer = [1, 2, 3, 4], [5, 6, 7]
        original, audit = complete_readout(model, prompt, answer)
        np.testing.assert_array_equal(original['query_position'], [3, 4, 5])
        np.testing.assert_array_equal(original['actual_id'], answer)
        self.assertEqual(audit['layer_visits'], [1, 1])
        self.assertFalse(audit['cached_KV_reused'])
        self.assertEqual(audit['full_model_forwards'], 1)
        self.assertEqual(audit['lm_head_projection_calls'], 1)
        for t, actual in enumerate(answer):
            with torch.no_grad():
                inputs = torch.tensor([prompt + answer[:t]])
                logits = model.lm_head(model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, -1:])
                expected = logit_readout(logits, [actual], original['rival_id'][t:t + 1])
            np.testing.assert_allclose(original['actual_logp'][t:t + 1], expected['actual_logp'], atol=1e-6)
            np.testing.assert_allclose(original['margin'][t:t + 1], expected['margin'], atol=1e-6)
        replay, _ = complete_readout(model, prompt, answer)
        self.assertTrue(verify_readout(replay, original, 1e-6)['passed'])
        changed, _ = complete_readout(model, prompt, [5, 6, 8], original['rival_id'])
        np.testing.assert_allclose(changed['actual_logp'][:2], original['actual_logp'][:2], atol=1e-6)
        self.assertEqual(CONDITIONS, ('native', 'cycle_fwd', 'cycle_rev'))


if __name__ == '__main__':
    unittest.main()
