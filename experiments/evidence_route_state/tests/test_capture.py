import pytest
import torch

from experiments.evidence_route_state.capture import RegisterGraphReplay


def tiny_replay(dtype: torch.dtype = torch.float32) -> RegisterGraphReplay:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
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
    model = transformers.LlamaForCausalLM(config).eval().to(dtype)
    model.set_attn_implementation("eager")
    return RegisterGraphReplay(model)


def capture(replay: RegisterGraphReplay, chunk: int):
    return replay.capture(
        torch.tensor([1, 2, 3, 4, 5, 6]),
        response_start=3,
        evidence_mask=torch.tensor([False, True, False]),
        predictor_chunk=chunk,
    )


def test_capture_preserves_coordinates_and_every_graph_axis():
    trace = capture(tiny_replay(), 2)
    graph = trace.graph

    assert graph.query_position.tolist() == [2, 3, 4]
    assert graph.prediction_position.tolist() == [3, 4, 5]
    assert graph.node_embedding.shape == (3, 4, 8)
    assert graph.residual_gram.shape == (3, 3, 4, 4)
    assert graph.head_write_gram.shape == (3, 2, 2, 4, 4)
    assert graph.route_topology.shape == (3, 2, 2, 4, 7)
    assert graph.mlp_relation.shape == (3, 2, 5)
    assert graph.margin_contribution.shape == (3, 4)
    assert graph.valid.tolist() == [False, False, True]
    torch.testing.assert_close(
        graph.margin_contribution.sum(1),
        trace.target_margin,
        rtol=2e-5,
        atol=2e-6,
    )
    assert trace.attention_write_error.max() < 2e-5
    assert trace.register_closure_error.max() < 2e-5


def test_complete_graph_state_is_invariant_to_predictor_chunking():
    replay = tiny_replay()
    row = capture(replay, 1)
    block = capture(replay, 5)

    for name in (
        "node_embedding",
        "residual_gram",
        "head_write_gram",
        "route_topology",
        "mlp_relation",
        "margin_contribution",
    ):
        torch.testing.assert_close(
            getattr(row.graph, name),
            getattr(block.graph, name),
            rtol=3e-5,
            atol=3e-6,
        )


def test_bfloat16_replay_keeps_derived_graph_finite_and_closes_native_margin():
    trace = capture(tiny_replay(torch.bfloat16), 2)

    for name in (
        "node_embedding",
        "residual_gram",
        "head_write_gram",
        "route_topology",
        "mlp_relation",
        "margin_contribution",
    ):
        value = getattr(trace.graph, name)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    assert trace.attention_write_error.max() < 0.02
    torch.testing.assert_close(
        trace.graph.margin_contribution.sum(1),
        trace.target_margin,
        rtol=0,
        atol=2e-6,
    )
