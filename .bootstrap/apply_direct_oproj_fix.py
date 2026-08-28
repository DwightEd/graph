from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


capture_path = "experiments/frozen_operator_graph/capture.py"
replace_once(
    capture_path,
    '''        def attention_hook(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                message, weights = _attention_output(output)
                if tuple(message.shape) != (1, tokens, self.hidden_size):
                    raise RuntimeError(f"layer {index} attention output has wrong shape")
                if tuple(weights.shape) != (
                    1,
                    self.head_count,
                    tokens,
                    tokens,
                ):
                    raise RuntimeError(
                        f"layer {index} attention weights have wrong shape; "
                        "eager full attention is required"
                    )
                store("attention_output", index, message[0])
                store("attention", index, weights[0, :, response_start:, :])

            return hook
''',
    '''        def attention_hook(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                message, weights = _attention_output(output)
                if tuple(message.shape) != (1, tokens, self.hidden_size):
                    raise RuntimeError(f"layer {index} attention output has wrong shape")
                if tuple(weights.shape) != (
                    1,
                    self.head_count,
                    tokens,
                    tokens,
                ):
                    raise RuntimeError(
                        f"layer {index} attention weights have wrong shape; "
                        "eager full attention is required"
                    )
                direct_projection = captured["attention_output"][index]
                if direct_projection is None:
                    raise RuntimeError(
                        f"layer {index} o_proj output hook did not fire before "
                        "the self-attention output hook"
                    )
                returned_message = message[0].detach().cpu()
                if not torch.equal(returned_message, direct_projection):
                    maximum = float(
                        (returned_message.float() - direct_projection.float())
                        .abs()
                        .max()
                        .item()
                    )
                    raise RuntimeError(
                        f"layer {index} self-attention output differs from direct "
                        f"o_proj output: max_abs_error={maximum:.6g}"
                    )
                store("attention", index, weights[0, :, response_start:, :])

            return hook
''',
)
replace_once(
    capture_path,
    '''            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    o_proj_input_hook(index)
                )
            )
            handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
''',
    '''            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    o_proj_input_hook(index)
                )
            )
            handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    simple_output_hook("attention_output", index)
                )
            )
            handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
''',
)

schema_path = "experiments/frozen_operator_graph/schema.py"
replace_once(schema_path, "GRAPH_VERSION = 2\n", "GRAPH_VERSION = 3\n")
replace_once(
    schema_path,
    '''    ``attention`` contains only response-query rows and has shape ``[H,R,N]``.
    The other tensors contain all token positions so source values and residual
    identities are never reconstructed from an approximation.
''',
    '''    ``attention`` contains only response-query rows and has shape ``[H,R,N]``.
    ``attention_output`` is captured directly from the frozen ``o_proj`` module
    and is bitwise-bound to the tensor returned by self-attention. The other
    tensors contain all token positions so source values and residual identities
    are never reconstructed from an approximation.
''',
)

graph_path = "experiments/frozen_operator_graph/graph.py"
replace_once(
    graph_path,
    "    max_native_projection_reconstruction = 0.0\n",
    "",
)
replace_once(
    graph_path,
    '''        # ``A @ V`` and ``o_proj`` are two distinct finite-precision operations.
        # Validate each against the tensor that actually entered that operation.
        # Comparing an unrounded float32 composition ``W_O(float32(A @ V))``
        # directly with a bfloat16/float16 module output creates a false failure
        # at large late-layer activations (typically one output ULP).
        output_factor_native = basis.output_factor[layer_index].detach().cpu()
        output_weight_native = _output_weight(output_factor_native)
        output_bias_native = basis.output_bias[layer_index].detach().cpu()
        output_weight = output_weight_native.float()
        output_bias = output_bias_native.float()
        captured_attention_native = layer_capture.attention_output[
            capture.response_start :
        ]
        captured_attention = captured_attention_native.float()

        captured_context_flat = captured_o_proj_input.reshape(response, hidden)
        native_projection = functional.linear(
            captured_context_flat.float(),
            output_weight,
            output_bias,
        ).to(dtype=captured_attention_native.dtype).float()
        native_projection_error = (
            native_projection - captured_attention
        ).abs().max()
        max_native_projection_reconstruction = max(
            max_native_projection_reconstruction,
            float(native_projection_error.item()),
        )
        if not torch.allclose(
            native_projection,
            captured_attention,
            atol=config.conservation_atol,
            rtol=config.conservation_rtol,
        ):
            raise ValueError(
                f"layer {layer_index} native-dtype W_O(o_proj_input) "
                "reconstruction failed: "
                f"max_abs_error={float(native_projection_error.item()):.6g}"
            )

''',
    '''        # The actual output is captured from the frozen ``o_proj`` module in
        # the same forward pass. ``capture.py`` also requires self-attention to
        # return that exact tensor, so no cross-device re-execution is used as a
        # proxy for CUDA bfloat16/float16 GEMM semantics. Edge/role messages are
        # decomposed in float32 and the unavoidable hardware numerical residual
        # is retained explicitly below.
        output_factor_native = basis.output_factor[layer_index].detach().cpu()
        output_weight_native = _output_weight(output_factor_native)
        output_bias_native = basis.output_bias[layer_index].detach().cpu()
        output_weight = output_weight_native.float()
        output_bias = output_bias_native.float()
        captured_attention = layer_capture.attention_output[
            capture.response_start :
        ].float()

''',
)
replace_once(
    graph_path,
    '''        "max_attention_reconstruction_abs_error": max_attention_reconstruction,
        "max_native_projection_reconstruction_abs_error": (
            max_native_projection_reconstruction
        ),
        "max_context_rounding_abs_error": max_context_rounding,
''',
    '''        "max_attention_reconstruction_abs_error": max_attention_reconstruction,
        "max_context_rounding_abs_error": max_context_rounding,
''',
)
replace_once(
    graph_path,
    '''        "projection_validation": (
            "captured_o_proj_input_float32_accumulation_quantized_to_model_dtype"
        ),
''',
    '''        "projection_output_binding": (
            "direct_o_proj_forward_hook_bitwise_equals_self_attention_output"
        ),
        "projection_validation": (
            "float32_edge_decomposition_plus_explicit_finite_precision_remainder"
        ),
''',
)

low_precision_path = Path(
    "experiments/frozen_operator_graph/tests/test_low_precision_projection.py"
)
low_precision_path.write_text(
    '''"""Finite-precision regression tests for frozen operator reconstruction."""

import torch

from ..basis import extract_operator_basis
from ..capture import ExactLlamaReplay
from ..graph import build_graph_tensors
from .helpers import TinyCausalLM


def test_large_bfloat16_late_layer_records_rounding_without_false_failure():
    """A one-ULP bfloat16 backend difference is retained, not misclassified."""

    torch.manual_seed(23)
    model = TinyCausalLM(
        layers=32,
        hidden=64,
        heads=8,
        kv_heads=2,
    ).to(dtype=torch.bfloat16)
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
    assert graph.audit["max_numerical_remainder_abs_error"] > 3e-2
    assert graph.audit["max_exact_attention_decomposition_abs_error"] < 1e-5
    assert "max_native_projection_reconstruction_abs_error" not in graph.audit
    assert graph.audit["projection_output_binding"] == (
        "direct_o_proj_forward_hook_bitwise_equals_self_attention_output"
    )
    assert graph.audit["projection_validation"] == (
        "float32_edge_decomposition_plus_explicit_finite_precision_remainder"
    )
''',
    encoding="utf-8",
)

capture_test_path = "experiments/frozen_operator_graph/tests/test_capture.py"
replace_once(capture_test_path, "import torch\n", "import pytest\nimport torch\n")
capture_tests = Path(capture_test_path)
text = capture_tests.read_text(encoding="utf-8")
addition = '''\n\ndef test_replay_rejects_any_post_oproj_attention_mutation():
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
'''
if "test_replay_rejects_any_post_oproj_attention_mutation" in text:
    raise RuntimeError("direct o_proj binding test already exists")
capture_tests.write_text(text.rstrip() + addition + "\n", encoding="utf-8")

readme_path = "experiments/frozen_operator_graph/README.md"
replace_once(
    readme_path,
    '''The package checks both

\\[
A^{(l)}V^{(l)}=\\text{o\\_proj input}
\\]

and

\\[
W_O^{(l)}(A^{(l)}V^{(l)})+b_O^{(l)}
=
\\text{captured attention output}
\\]

before accepting a graph.
''',
    '''The package checks the float32 edge sum against the actual captured
`o_proj` input, captures the actual `o_proj` output directly, and requires the
self-attention module to return that exact tensor. The graph then records the
finite-precision residual required by

\\[
\\text{captured }o\\_proj\\text{ output}
=
W_O^{(l)}(A^{(l)}V^{(l)})+b_O^{(l)}
+\\varepsilon_{\\mathrm{numeric}}^{(l)}.
\\]

This avoids pretending that a CPU/float32 re-execution can reproduce the exact
CUDA bfloat16/float16 GEMM rounding path.
''',
)
replace_once(
    readme_path,
    '''actual o_proj input
actual attention output
''',
    '''actual o_proj input
actual o_proj output, bitwise-bound to the self-attention return value
''',
)

design_path = "experiments/frozen_operator_graph/DESIGN.md"
replace_once(
    design_path,
    '''- `W_O(A V)+b` reconstructs the captured attention output within tolerance;
''',
    '''- direct `o_proj` output is bitwise-bound to the self-attention return value;
- float32 edge messages plus the explicit finite-precision remainder reconstruct
  the captured `o_proj` output within tolerance;
''',
)
