import pytest
import torch

from experiments.evidence_route_state.capture import RouteMessageReplay


def test_tiny_llama_capture_is_invariant_to_predictor_chunk_size():
    transformers = pytest.importorskip("transformers")
    LlamaConfig = transformers.LlamaConfig
    LlamaForCausalLM = transformers.LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(4)
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    replay = RouteMessageReplay(model)
    gram_pointers = tuple(gram.data_ptr() for gram in replay.output_grams)
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6])

    def run(chunk_size: int):
        rows = {}

        def consume(chunk):
            for local, query in enumerate(chunk.query_position.tolist()):
                causal_sources = query + 1
                rows[(chunk.layer, query)] = (
                    chunk.statistics.capacity[local, :, :causal_sources].detach().cpu(),
                    chunk.statistics.support[local, :, :causal_sources].detach().cpu(),
                    chunk.statistics.attention_write[local].detach().cpu(),
                )

        trace = replay.capture(
            token_ids,
            response_start=3,
            predictor_chunk=chunk_size,
            logit_chunk=2,
            consume_chunk=consume,
        )
        return trace, rows

    full_trace, full_rows = run(5)
    assert tuple(gram.data_ptr() for gram in replay.output_grams) == gram_pointers
    expected_rows = {(layer, query) for layer in range(2) for query in range(5)}
    assert set(full_rows) == expected_rows
    assert full_trace.reconstruction_relative_l2.max() < 2e-5

    for chunk_size in (1, 2):
        trace, rows = run(chunk_size)
        assert set(rows) == expected_rows
        for key in expected_rows:
            for actual, expected in zip(rows[key], full_rows[key], strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        for name in (
            "target_logprob",
            "target_confidence",
            "target_margin",
            "reconstruction_max_abs",
            "reconstruction_relative_l2",
            "mlp_write_norm",
            "mlp_relative_norm",
            "mlp_state_cosine",
        ):
            torch.testing.assert_close(
                getattr(trace, name),
                getattr(full_trace, name),
                rtol=2e-5,
                atol=2e-6,
            )
        for family in ("attention_prompt", "functional_prompt"):
            actual = getattr(trace, family)
            expected = getattr(full_trace, family)
            torch.testing.assert_close(
                actual.effective_sources,
                expected.effective_sources,
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                actual.effective_rank,
                expected.effective_rank,
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                actual.anchor_source,
                expected.anchor_source,
                rtol=0,
                atol=0,
            )
        assert tuple(gram.data_ptr() for gram in replay.output_grams) == gram_pointers


def test_bfloat16_capture_keeps_all_derived_route_geometry_in_float32():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_bias=False,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval().to(torch.bfloat16)
    model.set_attn_implementation("eager")
    observed = []

    def consume(chunk):
        observed.append(
            (
                chunk.statistics.capacity.dtype,
                chunk.statistics.support.dtype,
                chunk.statistics.attention_write.dtype,
                chunk.selected_messages(
                    torch.tensor([0]),
                    torch.tensor([0]),
                    torch.tensor([0]),
                ).dtype,
            )
        )

    replay = RouteMessageReplay(model)
    assert all(gram.dtype == torch.float32 for gram in replay.output_grams)
    trace = replay.capture(
        torch.tensor([1, 2, 3, 4, 5]),
        response_start=2,
        predictor_chunk=2,
        consume_chunk=consume,
    )

    assert observed
    assert all(dtype == torch.float32 for row in observed for dtype in row)
    assert torch.isfinite(trace.target_logprob).all()
    assert trace.reconstruction_relative_l2.max() < 0.02
