"""CPU coordinate/pooling checks, using randomly initialized tiny model only."""

import copy

import numpy as np
import pytest

from route_graph.event_matcher import match_catalog
from route_graph.source_event_graph import compile_pointer_events, raw_inventory
from route_graph.span_feature_capture import capture_post_block, span_mapping


def inputs():
    inv = raw_inventory("A crane lifted crates.", side="source", sample_id="s")
    catalog = match_catalog(inv)
    # Deliberately cross lexical boundaries to check exact overlap weighting.
    offsets = [[0, 4], [4, 9], [9, 13], [13, 17], [17, 22]]
    mapping = span_mapping(inv, catalog, [1, 2, 3, 4, 5], offsets,
                           {"prompt": inv["text"], "response": "", "source_span": [0, len(inv["text"])]})
    encoder = {k: "a" * 64 for k in ("model_sha256", "tokenizer_sha256", "response_input_sha256", "capture_code_sha256")}
    encoder.update(source_input_sha256=mapping["input_sha256"], layers=[0, 1], representation="post_block_residual_mean")
    return inv, catalog, mapping, encoder


@pytest.mark.parametrize("overlap", [False, True])
def test_pooling_matches_direct_weighted_block_outputs_and_removes_hooks(overlap):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    _, catalog, mapping, encoder = inputs()
    if overlap:
        offsets = copy.deepcopy(mapping["token_offsets"])
        offsets[1][0] -= 1
        offsets[2][0] -= 1
        mapping = span_mapping(mapping["inventory"], catalog, mapping["input_ids"], offsets, mapping["input_context"])
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
    states = []
    def direct(module, args, output):
        states.append((output[0] if isinstance(output, tuple) else output).detach().float().numpy())
    handles = [layer.register_forward_hook(direct) for layer in model.model.layers]
    vectors, record = capture_post_block(model, catalog, mapping, encoder)
    for handle in handles:
        handle.remove()
    for layer, state in enumerate(states):
        for row, membership in enumerate(mapping["feature_memberships"]):
            idx = [m["token_index"] for m in membership["members"]]
            weights = [m["weight"] for m in membership["members"]]
            expected = np.average(state[0, idx], weights=weights, axis=0)
            np.testing.assert_allclose(vectors[row, layer], expected, rtol=2e-5, atol=2e-7)
    assert record["mapping"]["input_sha256"] == encoder["source_input_sha256"]
    assert all(not layer._forward_hooks for layer in model.model.layers)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        capture_post_block(model, catalog, mapping, encoder)
    assert not torch.cuda.is_initialized()


def test_source_mapping_rejects_response_leak_and_incomplete_coordinates():
    inv, catalog, mapping, _ = inputs()
    context = copy.deepcopy(mapping["input_context"])
    context["response"] = "response information"
    with pytest.raises(ValueError, match="precede any response"):
        span_mapping(inv, catalog, mapping["input_ids"], mapping["token_offsets"], context)
    broken = copy.deepcopy(mapping["token_offsets"])
    broken[1] = None
    with pytest.raises(ValueError, match="does not cover"):
        span_mapping(inv, catalog, mapping["input_ids"], broken, mapping["input_context"])


def test_mapping_splits_character_mass_for_overlapping_offsets():
    inv, catalog, mapping, _ = inputs()
    overlapping = copy.deepcopy(mapping["token_offsets"])
    overlapping[1][0] = overlapping[0][1] - 1
    updated = span_mapping(inv, catalog, mapping["input_ids"], overlapping, mapping["input_context"])
    for member in updated["feature_memberships"]:
        assert sum(n["weight"] for n in member["members"]) == pytest.approx(member["span"][1] - member["span"][0])


def test_real_llama_byte_fallback_offset_pattern_preserves_characters():
    # Offsets observed from the installed Llama-3.1 tokenizer, not model results.
    text = "🧬 🦦 café"
    inv = raw_inventory(text, side="source", sample_id="unicode")
    catalog = match_catalog(inv)
    offsets = [[0, 1], [0, 1], [0, 1], [1, 3], [2, 3], [2, 3], [3, 8]]
    mapping = span_mapping(inv, catalog, [9468, 100, 105, 11410, 99, 99, 53050], offsets,
                           {"prompt": text, "response": "", "source_span": [0, len(text)]})
    for member in mapping["feature_memberships"]:
        assert sum(n["weight"] for n in member["members"]) == pytest.approx(member["span"][1] - member["span"][0])


def test_event_span_explicitly_pools_non_whitespace_mass():
    text = "A crane lifted crates."
    inv = raw_inventory(text, side="source", sample_id="whitespace")
    pointer = lambda a, b: {"unit_id": "u0", "start_token": a, "end_token": b}
    envelope = {
        "inventory_sha256": inv["sha256"],
        "side": "source",
        "sample_id": "whitespace",
        "prediction": {
            "events": [{
                "anchor": pointer(0, 5),
                "roles": [
                    {"role": "subject", "pointer": pointer(0, 2)},
                    {"role": "predicate", "pointer": pointer(2, 3)},
                ],
            }],
        },
    }
    catalog = match_catalog(inv, compile_pointer_events(inv, envelope))
    offsets = [[0, 1], [2, 7], [8, 14], [15, 21], [21, 22]]
    mapping = span_mapping(inv, catalog, [1, 2, 3, 4, 5], offsets,
                           {"prompt": text, "response": "", "source_span": [0, len(text)]})
    event = mapping["feature_memberships"][-1]
    assert event["raw_span_characters"] == 22
    assert event["non_whitespace_characters"] == 19
    assert event["pooled_character_mass"] == pytest.approx(19)
    assert "non_whitespace_character_mass" in mapping["pooling"]
