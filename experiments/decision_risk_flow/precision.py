"""FP32 execution of frozen BF16 weights without retaining FP32 weight copies."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        ctx.save_for_backward(weight)
        return F.linear(inputs.float(), weight.float())

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        return FrozenTranspose.apply(gradient, weight), None


class FrozenTranspose(torch.autograd.Function):
    """Keep BF16 weights across the double backward used for exact JVPs."""

    @staticmethod
    def forward(ctx, inputs, weight):
        ctx.save_for_backward(weight)
        return inputs.float() @ weight.float()

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        return FrozenLinear.apply(gradient, weight), None


def linear_forward(module, inputs):
    return FrozenLinear.apply(inputs, module.weight)


def enable_fp32_execution(model):
    """Model is frozen and bias-free; native RMS/attention/MLP remain FP32."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None or module.weight.requires_grad:
                raise ValueError('FP32 frozen-linear execution requires frozen bias-free weights')
            module.forward = MethodType(linear_forward, module)
    model.model.embed_tokens.register_forward_hook(lambda module, args, output: output.float())
    return model
