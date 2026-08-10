import json
import tempfile
import unittest
from pathlib import Path

from ragtruth import load_ragtruth_samples, tokenize_ragtruth_sample


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return "system:user:" + messages[1]["content"] + "|"

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class RagTruthTests(unittest.TestCase):
    def test_reader_filters_without_reading_hallucination_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_info.jsonl").write_text(json.dumps({
                "source_id": "s1", "prompt": "question", "task_type": "qa",
            }) + "\n")
            (root / "response.jsonl").write_text("\n".join(json.dumps(row) for row in [
                {"id": "r1", "source_id": "s1", "response": "answer", "split": "test", "model": "llama-2-7b-chat", "quality": "good", "labels": ["do not read"]},
                {"id": "r2", "source_id": "s1", "response": "skip", "split": "test", "model": "other", "quality": "good"},
            ]) + "\n")
            samples = load_ragtruth_samples(root, split="test", generator_model="llama-2-7b-chat", task_type="qa")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].response_id, "r1")
        self.assertFalse(hasattr(samples[0], "labels"))

    def test_tokenizer_uses_one_full_text_call_and_exact_response_boundary(self) -> None:
        tokenizer = FakeTokenizer()
        token_ids, response_idx = tokenize_ragtruth_sample(
            tokenizer, prompt="question", response="answer"
        )

        self.assertEqual(tokenizer.messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(response_idx, len("system:user:question|"))
        self.assertEqual(token_ids.tolist()[response_idx:], list(range(response_idx, len(token_ids))))

