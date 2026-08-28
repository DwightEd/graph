import pytest
import torch

from ..capture import ExactLlamaReplay
from .helpers import tiny_model_bundle


def test_exact_tiny_replay_captures_all_required_signals():
    model, _basis = tiny_model_bundle()
    replay = ExactLlamaReplay(model, checkpoint="tiny-checkpoint")
    capture = replay.capture(
        torch.tensor([1, 2, 3, 4, 5]),
        response_start=2,
        conservation_atol=2e-5,
        conservation_rtol=2e-5,
    )
    assert capture.layer_count == 2
    assert capture.response_count == 3
    assert capture.layers[0].attention.shape == (2, 3, 5)
    assert capture.layers[0].value_states.shape == (5, 1, 2)
    assert capture.layers[0].o_proj_input.shape == (5, 2, 2)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_tiny_replay_runs_through_exact_graph_construction():
    from ..graph import build_graph_tensors

    model, basis = tiny_model_bundle()
    replay = ExactLlamaReplay(model, checkpoint="tiny-checkpoint")
    capture = replay.capture(
        torch.tensor([1, 2, 3, 4, 5]),
        response_start=2,
        conservation_atol=2e-5,
        conservation_rtol=2e-5,
    )
    graph = build_graph_tensors(capture, basis)
    assert graph.edge_index.shape[1] == graph.audit["causal_token_pairs_considered"]
    assert graph.audit["max_attention_reconstruction_abs_error"] < 2e-5
    assert graph.audit["max_o_proj_input_reconstruction_abs_error"] < 2e-5


def test_default_numeric_tolerance_accepts_exact_bfloat16_replay():
    model, _ = tiny_model_bundle()
    model = model.to(dtype=torch.bfloat16)
    from ..basis import extract_operator_basis
    from ..graph import build_graph_tensors

    basis = extract_operator_basis(
        model,
        checkpoint="tiny-bfloat16",
        compute_device="cpu",
        compute_dtype=torch.float32,
    )
    replay = ExactLlamaReplay(model, checkpoint="tiny-bfloat16")
    capture = replay.capture(torch.tensor([1, 2, 3, 4, 5]), response_start=2)
    graph = build_graph_tensors(capture, basis)
    assert graph.audit["max_attention_reconstruction_abs_error"] < 5e-3

def test_replay_rejects_any_post_oproj_attention_mutation():
    """The recorded attention update must be the direct frozen o_proj output."""

    model, _basis = tiny_model_bundle()
    attention = model.model.layers[0].self_attn
    original_forward = attention.forward

    def altered_forward(hidden_states, **kwargs):
        output, weights = original_forward(hidden_states, **kwargs)
        return output + output.new_tensor(0.125), weights

    attention.forward = altered_forward
    replay = ExactLlamaReplay(model, checkpoint="tiny-mutated-attention")
    with pytest.raises(RuntimeError, match="differs from direct o_proj output"):
        replay.capture(
            torch.tensor([1, 2, 3, 4, 5]),
            response_start=2,
        )

