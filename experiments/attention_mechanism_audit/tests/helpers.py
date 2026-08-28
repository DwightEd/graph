"""Small label-free fixtures for prompt roles and counterfactual masks."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.attention_mechanism_audit.roles import (
    PromptRoleMap,
    prompt_token_sha256,
)


class CharacterChatTokenizer:
    """A deterministic offset-preserving tokenizer used only by unit tests."""

    prefix = "<system>You are a helpful assistant.</system><user>"
    suffix = "</user><assistant>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages[1]["content"]},
        ]
        return self.prefix + messages[1]["content"] + self.suffix

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) + 1 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def source_digest(source_id: str) -> str:
    return hashlib.sha256(
        json.dumps({"source_id": source_id}, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_role_map(
    prompt_token_ids,
    role_ids,
    *,
    source_id="source-a",
    task_type="QA",
) -> PromptRoleMap:
    prompt = np.asarray(prompt_token_ids, dtype=np.int64)
    return PromptRoleMap(
        source_id=source_id,
        task_type=task_type,
        response_idx=len(prompt),
        role_ids=np.asarray(role_ids, dtype=np.int8),
        prompt_token_sha256=prompt_token_sha256(prompt),
        source_info_sha256=source_digest(source_id),
        prompt_token_ids=prompt,
    ).validate()


def qa_source():
    prompt = (
        "Briefly answer the following question:\n"
        "Which color?\n"
        "Bear in mind that your response should be strictly based on the "
        "following ten passages:\n"
        "passage 1: The flag is blue.\n"
        "In case the passages do not contain the necessary information to "
        "answer the question, please reply with: \"Unable to answer based on "
        "given passages.\"\n"
        "output:"
    )
    return {
        "source_id": "qa-source",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "Which color?",
            "passages": "passage 1: The flag is blue.",
        },
        "prompt": prompt,
    }


def summary_source():
    article = "The court met on Wednesday."
    return {
        "source_id": "summary-source",
        "task_type": "Summary",
        "source": "CNN/DM",
        "source_info": article,
        "prompt": f"Summarize the following news within 20 words:\n{article}\n\noutput:",
    }


def data2txt_source():
    data = {"name": "Cafe", "stars": 4.0}
    return {
        "source_id": "data-source",
        "task_type": "Data2txt",
        "source": "Yelp",
        "source_info": data,
        "prompt": (
            "Instruction:\nWrite an objective overview based only on the data.\n"
            f"Structured data:\n{data}\nOverview:"
        ),
    }
