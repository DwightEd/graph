"""Native Qwen3 reader; the teaching intervention adapter excludes Qwen3."""

import torch
from torch.nn import functional as functional

from transformers import AutoModelForCausalLM, AutoTokenizer


class Reader:
    def __init__(self, model_path, batch_size):
        torch.manual_seed(42)
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda:0").eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.letter_ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
        if any(len(ids) != 1 for ids in self.letter_ids):
            raise ValueError("A/B scoring requires single-token response labels")

    def encode(self, messages):
        texts = [self.tokenizer.apply_chat_template(message, tokenize=False,
                 add_generation_prompt=True, enable_thinking=False) for message in messages]
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        if inputs.input_ids.shape[1] > self.model.config.max_position_embeddings - 3072:
            raise ValueError("Input exceeds reserved model context; no silent truncation")
        return inputs.to(self.model.device)

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=3072):
        results = []
        for start in range(0, len(messages), self.batch_size):
            inputs = self.encode(messages[start:start + self.batch_size])
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)
            outputs = output[:, inputs.input_ids.shape[1]:]
            for ids in outputs:
                results.append(dict(text=self.tokenizer.decode(ids, skip_special_tokens=True),
                    generated_tokens=int((ids != self.tokenizer.pad_token_id).sum()),
                    reached_limit=len(ids) == max_new_tokens and int(ids[-1]) != self.tokenizer.eos_token_id))
        return results

    @torch.inference_mode()
    def margins(self, messages):
        """Full transformer, FP32 two-row output head: stable z(B)-z(A)."""
        results = []
        ids = torch.tensor([item[0] for item in self.letter_ids], device=self.model.device)
        weight = self.model.lm_head.weight[ids].float()
        for start in range(0, len(messages), self.batch_size):
            inputs = self.encode(messages[start:start + self.batch_size])
            positions = inputs.attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(inputs.attention_mask == 0, 0)
            hidden = self.model.model(**inputs, position_ids=positions, use_cache=False).last_hidden_state[:, -1]
            logits = functional.linear(hidden.float(), weight)
            results.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
        return results
