"""CPU contract checks for v2 all-source erased query capture."""

import copy

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from next_iteration.grounded_graph_restoration_features import (
    capture_empty,
    intervention,
)


class BoundaryTokenizer:
    """A deterministic tokenizer whose source token extends past the source span."""

    bos_token_id = 1

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        assert text == " "
        return [2]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        assert text == "PabcQ"
        result = {"input_ids": [10, 11]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(0, 1), (1, 5)]
        return result


def _packet():
    # The second prompt token spans ``abcQ`` while the source is only ``abc``.
    # It must be erased whole, but the response-history token must remain intact.
    return {
        "sha256": "packet-receipt",
        "input_ids": [1, 10, 11, 13],
        "prompt_record": {"prompt": "PabcQ", "prompt_length": 3, "source_span": [1, 4]},
        "source_offsets": [[0, 0], [0, 0], [0, 3]],
        "query_positions": [2, 3],
        "target_positions": [3, 4],
    }


def _model():
    torch.manual_seed(101)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval().requires_grad_(False)


def test_whole_boundary_token_erasure_preserves_history_and_captures_actual_pretoken_states():
    packet = _packet()
    original = copy.deepcopy(packet)
    tokenizer, model = BoundaryTokenizer(), _model()

    erased = intervention(packet, tokenizer)
    assert erased["erased_prompt_token_indices"] == [2]
    assert erased["boundary_crossing_tokens"] == [{"position": 2, "prompt_char_span": [1, 5]}]
    assert erased["executed_input_ids"] == [1, 10, 2, 13]
    assert erased["executed_input_ids"][packet["prompt_record"]["prompt_length"]:] == packet["input_ids"][packet["prompt_record"]["prompt_length"]:]
    assert packet == original

    query, receipt = capture_empty(model, packet, tokenizer)
    with torch.no_grad():
        expected = model.model(
            input_ids=torch.tensor([erased["executed_input_ids"]]), use_cache=False, return_dict=True
        ).last_hidden_state[0, packet["query_positions"]].float().numpy()
    np.testing.assert_array_equal(query, expected)
    assert receipt["query_positions"] == [2, 3]
    assert receipt["actual_observer_forwards"] == receipt["actual_final_norm_hook_count"] == 1
    assert receipt["observer_dtype"] == str(model.dtype)
    assert receipt["attention_implementation"] == model.config._attn_implementation
    assert not model.model.norm._forward_hooks
