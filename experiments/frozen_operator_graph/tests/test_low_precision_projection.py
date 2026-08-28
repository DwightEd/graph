"""Finite-precision regression tests for frozen operator reconstruction."""

import torch

from ..basis import extract_operator_basis
from ..capture import ExactLlamaReplay
from ..graph import build_graph_tensors
from .helpers import TinyCausalLM


def test_large_bfloat16_late_layer_records_rounding_without_false_failure():
    """A one-ULP bfloat16 output difference is not an operator mismatch."""

    torch.manual_seed(23)
    model = TinyCausalLM(
        layers=32,
        hidden=64,
        heads=8,
        kv_heads=2,
    ).to(dtype=torch.bfloat16)
    # Increase only the final projection scale.  The former implementation
    # compared an unrounded float32 composition against this bfloat16 output and
    # raised with a max error above 0.03 despite an exact frozen projection.
    with torch.no_grad():
        final_projection = model.model.layers[-1].self_attn.o_proj
        final_projection.weight.mul_(16)
        final_projection.bias.mul_(16)

    basis = extract_operator_basis(
        model,
        checkpoint="tiny-bfloat16-large-final-layer",
        compute_device="cpu",
        compute_dtype=torch.float32,
    )
    replay = ExactLlamaReplay(
        model,
        checkpoint="tiny-bfloat16-large-final-layer",
    )
    capture = replay.capture(
        torch.arange(20, dtype=torch.long) % 32,
        response_start=5,
    )
    graph = build_graph_tensors(capture, basis)

    assert graph.audit["max_attention_reconstruction_abs_error"] > 3e-2
    assert graph.audit["max_native_projection_reconstruction_abs_error"] < 5e-3
    assert graph.audit["max_numerical_remainder_abs_error"] > 3e-2
    assert graph.audit["max_exact_attention_decomposition_abs_error"] < 1e-5
    assert graph.audit["projection_validation"] == (
        "captured_o_proj_input_float32_accumulation_quantized_to_model_dtype"
    )
