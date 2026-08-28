import numpy as np
import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from experiments.attention_mechanism_audit.replay import FrozenCausalReplay
from experiments.grounded_route.graph import TokenEdges, TokenGraph


def test_tiny_llama_full_causal_forward_paths_are_numerically_equivalent():
    """Independently bind the cache-style and legacy full-causal HF calls."""

    torch.manual_seed(13)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    input_ids = torch.tensor([[1, 5, 7, 9, 11, 13]], dtype=torch.long)
    sequence_length = int(input_ids.shape[1])
    extraction_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(sequence_length, dtype=torch.long)[None, :]
    allowed = torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool)
    )

    with torch.no_grad():
        extraction = model(
            input_ids=input_ids,
            attention_mask=extraction_mask,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        embeddings = model.get_input_embeddings()(input_ids)
        legacy_mask = torch.zeros(
            (sequence_length, sequence_length),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        legacy_mask.masked_fill_(~allowed, torch.finfo(embeddings.dtype).min)
        legacy = model(
            inputs_embeds=embeddings,
            attention_mask=legacy_mask[None, None, :, :],
            position_ids=position_ids,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    assert extraction.attentions is not None and legacy.attentions is not None
    assert extraction.hidden_states is not None and legacy.hidden_states is not None
    torch.testing.assert_close(
        extraction.hidden_states[-1],
        legacy.hidden_states[-1],
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        extraction.logits,
        legacy.logits,
        rtol=1e-6,
        atol=1e-7,
    )
    for extraction_layer, legacy_layer in zip(
        extraction.attentions,
        legacy.attentions,
        strict=True,
    ):
        torch.testing.assert_close(
            extraction_layer,
            legacy_layer,
            rtol=1e-6,
            atol=1e-7,
        )


def _exact_graph_from_attention(replay, tokens, prompt_length, allowed):
    token_tensor = torch.as_tensor(tokens, dtype=torch.long)
    causal = np.tri(len(tokens), dtype=np.bool_)
    if np.array_equal(allowed, causal):
        forward = {
            "input_ids": token_tensor[None],
            "attention_mask": torch.ones_like(token_tensor[None]),
        }
    else:
        embeddings = replay.model.get_input_embeddings()(token_tensor[None])
        forward = {
            "inputs_embeds": embeddings,
            "attention_mask": replay._additive_mask(
                allowed,
                dtype=embeddings.dtype,
                device=embeddings.device,
            ),
            "position_ids": torch.arange(len(tokens), dtype=torch.long)[None],
        }
    with torch.no_grad():
        output = replay._backbone()(
            **forward,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    assert output.attentions is not None
    response_count = len(tokens) - prompt_length
    source, target, layer, head, weight = [], [], [], [], []
    diagonal = torch.empty(response_count, len(output.attentions), 2)
    retained = torch.zeros_like(diagonal)
    for layer_index, attention in enumerate(output.attentions):
        for head_index in range(2):
            for response_index in range(response_count):
                query = prompt_length + response_index
                diagonal[response_index, layer_index, head_index] = attention[
                    0, head_index, query, query
                ]
                for key in range(query):
                    value = attention[0, head_index, query, key]
                    source.append(key)
                    target.append(query)
                    layer.append(layer_index)
                    head.append(head_index)
                    weight.append(value)
                    retained[response_index, layer_index, head_index] += value
    return TokenGraph(
        sample_id="tiny",
        source_id="tiny-source",
        task_type="QA",
        response_start=prompt_length,
        token_count=len(tokens),
        response_count=response_count,
        layer_count=2,
        head_count=2,
        attention_floor=0.0,
        edges=TokenEdges(
            source=torch.as_tensor(source, dtype=torch.long),
            target=torch.as_tensor(target, dtype=torch.long),
            layer=torch.as_tensor(layer, dtype=torch.long),
            head=torch.as_tensor(head, dtype=torch.long),
            weight=torch.stack(weight).float(),
        ),
        diagonal=diagonal.float(),
        unresolved=(1.0 - retained - diagonal).clamp_min(0).float(),
        token_ids=token_tensor,
    ).check().canonicalize()


@pytest.mark.parametrize("custom_mask", [False, True])
def test_tiny_llama_extraction_and_intervention_mask_projection_hooks(custom_mask):
    """Exercise extraction-equivalent 2D and intervened 4D Llama paths."""

    torch.manual_seed(17)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    replay = FrozenCausalReplay(model, checkpoint="tiny-local-llama")
    tokens = np.asarray([1, 5, 7, 9, 11, 13], dtype=np.int64)
    prompt_length = 3
    allowed = np.tri(len(tokens), dtype=np.bool_)
    if custom_mask:
        # Exercise a nontrivial 4D additive mask while keeping every row valid.
        allowed[4:, 1] = False
    graph = _exact_graph_from_attention(replay, tokens, prompt_length, allowed)

    capture = replay.capture_baseline(
        tokens,
        prompt_length,
        allowed_attention=allowed,
        vocab_chunk_size=7,
        gradient_probes=2,
        attribution_seed=23,
        expected_graph=graph,
    )

    assert capture.value_states.shape == (2, 6, 1, 8)
    assert capture.o_proj_input_gradient_probes.shape == (2, 2, 3, 2, 8)
    assert capture.o_proj_input_gradients.shape == (2, 3, 2, 8)
    assert capture.attention_cache_binding["verified"] is True
    assert torch.isfinite(capture.o_proj_input_gradient_probes).all()
    assert torch.isfinite(capture.chosen_logprob).all()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
