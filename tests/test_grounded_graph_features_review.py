"""CPU provenance/alignment checks for grounded graph feature capture."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from next_iteration.constraint_inventory import compile_inventory
from next_iteration.grounded_graph_adapter import GroundedGraphAdapter
from next_iteration.grounded_graph_features import (
    NODE_KINDS,
    capture,
    prepare_example,
    validate_packet,
)
from route_graph.audit_artifacts import file_sha256


class CharacterTokenizer:
    """Deterministic local offsets; each non-special character occupies one token."""

    bos_token_id = 1

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [2 + (ord(character) % 89) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def _example(text="The value is Ada."):
    tokenizer = CharacterTokenizer()
    source = repr({"name": "Ada", "city": "Pisa", "hours": {"Monday": "8:0-16:0"}})
    inventory = compile_inventory(source, source_id="7", task="Data2txt")
    prompt = "Task:\n" + source + "\nAnswer:\n"
    source_start = prompt.index(source)
    prompt_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    record = {
        "source_id": "7",
        "prompt": prompt,
        "source_span": [source_start, source_start + len(source)],
        "prompt_token_ids": prompt_ids,
        "prompt_length": len(prompt_ids),
    }
    owner = next(field for field in inventory["fields"] if field["display_text"] == "Ada")
    packet = prepare_example(
        record,
        inventory,
        text,
        [{"target_span": [13, 16], "source_owner_id": owner["id"]}],
        tokenizer,
    )
    return tokenizer, inventory, packet


def _model():
    torch.manual_seed(73)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval().requires_grad_(False)


def _identity():
    return {
        "model_files": {"weights.bin": "sealed"},
        "tokenizer_files": {"tokenizer.json": "sealed"},
        "capture_code_sha256": file_sha256(
            Path(__file__).parents[1] / "next_iteration" / "grounded_graph_features.py"
        ),
    }


def test_capture_uses_final_norm_genuine_pretoken_states_and_prompt_only_source_pool():
    tokenizer, inventory, packet = _example()
    model = _model()

    arrays, receipt = capture(model, tokenizer, inventory, packet, _identity())
    ids = torch.tensor([packet["input_ids"]])
    with torch.no_grad():
        final_norm = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0].float()

    assert packet["query_positions"] == [target - 1 for target in packet["target_positions"]]
    assert packet["query_positions"][0] == packet["prompt_record"]["prompt_length"] - 1
    np.testing.assert_array_equal(arrays["query"], final_norm[packet["query_positions"]].numpy())
    assert all(
        index < packet["prompt_record"]["prompt_length"]
        for node in packet["nodes"]
        for index in node["prompt_token_indices"]
    )
    mapped = next(node for node in packet["nodes"] if node["available"])
    expected_source = final_norm[mapped["prompt_token_indices"]].mean(0).numpy()
    np.testing.assert_allclose(
        arrays["source"][packet["nodes"].index(mapped)], expected_source, rtol=1e-6, atol=1e-6
    )
    assert receipt["packet_sha256"] == packet["sha256"]
    assert receipt["final_norm_hook_executions"] == receipt["actual_observer_forwards"] == 1
    assert not model.model.norm._forward_hooks


def test_response_changes_do_not_change_prompt_source_features_and_packet_tampering_is_rejected():
    tokenizer, inventory, first = _example("The value is Ada.")
    _, _, second = _example("A wholly different response.")
    model = _model()

    first_arrays, _ = capture(model, tokenizer, inventory, first, _identity())
    second_arrays, _ = capture(model, tokenizer, inventory, second, _identity())
    np.testing.assert_array_equal(first_arrays["source"], second_arrays["source"])

    tampered = copy.deepcopy(first)
    tampered["input_ids"][0] += 1
    with pytest.raises(ValueError, match="exact reconstruction"):
        validate_packet(tampered, inventory, tokenizer)


def test_packet_rejects_missing_bos_alignment_and_capture_requires_complete_identity():
    tokenizer, inventory, packet = _example()
    malformed = copy.deepcopy(packet)
    malformed["prompt_record"]["prompt_token_ids"] = malformed["prompt_record"]["prompt_token_ids"][1:]
    with pytest.raises(ValueError, match="prompt/BOS tokenization differs"):
        validate_packet(malformed, inventory, tokenizer)

    with pytest.raises(ValueError, match="exact observer/tokenizer/capture identities"):
        capture(_model(), tokenizer, inventory, packet, {"model_files": {}, "tokenizer_files": {}})

    wrong_code = _identity()
    wrong_code["capture_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="capture code identity differs"):
        capture(_model(), tokenizer, inventory, packet, wrong_code)


def test_feature_node_vocabulary_matches_the_default_adapter_checkpoint_shape():
    adapter = GroundedGraphAdapter(input_dim=16, adapter_dim=4)
    assert adapter.node_type.num_embeddings == len(NODE_KINDS) == 9
