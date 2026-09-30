"""Native prediction alignment, branch isolation, complete coverage, and label isolation."""


import numpy as np
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from experiments.native_support.evidence_contrast.views import prepare_views, query_positions, unit_intervals


def test_units_keep_decimals_list_markers_titles_and_all_tokens():
    pieces = ["Title", "\n", "1", ".", " Value", " 3", ".", "14", ".", "\n", "Tail"]
    response = dict(prompt_length=1, token_text=["prompt", *pieces])
    units = unit_intervals(response, 5)
    covered = [target for unit in units for target in range(unit["start"], unit["stop"])]
    assert covered == list(range(len(pieces)))
    assert all(unit["stop"]-unit["start"] <= 5 for unit in units)
    assert {(u["text_start"], u["text_stop"]) for u in units} == {(0, 2), (2, 9), (9, 11)}


def test_source_deletion_and_queries_preserve_original_answer_ids():
    response = dict(id="a", prompt_length=5, token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    token_text=["<", "evidence", "Q", "evidence2", ">", "A", ". ", "B"])
    sources = dict(blocks=[[1], [3]], group_ids=[4, 0, 3, 1, 4, 2, 2, 2])
    views = prepare_views(response, sources, 64)
    assert views["prompt_without_source"] == [1, 3, 5]
    assert views["answer_ids"] == [6, 7, 8]
    assert views["removed_prompt_positions"] == [1, 3]
    positions = query_positions(views)
    np.testing.assert_array_equal(positions["full_query_with_source"], [4, 5, 6])
    np.testing.assert_array_equal(positions["local_query_without_source"], [2, 3, 2])
