"""Local tangent transport through a bias-free Llama RMSNorm/SwiGLU block.

This measures infinitesimal responsiveness at a native activation, not a finite
deletion effect, a decomposition of the FFN write, or factual support.
"""

import torch


@torch.no_grad()
def ffn_transport(block, residual, directions):
    """One [D] site and [directions,D] tangents; include both SwiGLU branches.

    Reference inputs/weights use float32 or float64 on one device; this is not a
    bitwise reproduction of mixed-precision native kernels. The caller supplies
    the residual immediately AFTER attention. No attention/KV paths are included.
    """
    norm = block.post_attention_layernorm
    mlp = block.mlp
    scale = (residual.square().mean() + norm.variance_epsilon).sqrt()
    normalized = norm.weight * residual / scale
    radial = (directions * residual).mean(-1, keepdim=True)
    normalized_directions = norm.weight * (
        directions / scale - residual * radial / scale.pow(3)
    )

    gate = normalized @ mlp.gate_proj.weight.T
    up = normalized @ mlp.up_proj.weight.T
    gate_direction = normalized_directions @ mlp.gate_proj.weight.T
    up_direction = normalized_directions @ mlp.up_proj.weight.T
    sigmoid = gate.sigmoid()
    activation = gate * sigmoid
    derivative = sigmoid * (1 + gate * (1 - sigmoid))

    hidden_direction = derivative * gate_direction * up + activation * up_direction
    output = residual + (activation * up) @ mlp.down_proj.weight.T
    transported = directions + hidden_direction @ mlp.down_proj.weight.T
    return output, transported
