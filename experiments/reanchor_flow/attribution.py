"""Target-margin gradients used to screen ETCC messages and carrier states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from experiments.common.llama_message_intervention import ForwardCache, forward_layers

from .worlds import TargetContrast


@dataclass(frozen=True)
class TargetGradients:
    """Path-averaged gradients at explicit Transformer computation nodes."""

    position: Tensor
    head_output: Tensor
    layer_input: Tensor
    attention_write: Tensor
    mlp_write: Tensor


class GradientObserver:
    def __init__(self) -> None:
        self.head: dict[int, list[tuple[int, Tensor]]] = {}
        self.layer_input: dict[int, Tensor] = {}
        self.attention_write: dict[int, Tensor] = {}
        self.mlp_write: dict[int, Tensor] = {}

    @staticmethod
    def retain(value: Tensor) -> None:
        if value.requires_grad:
            value.retain_grad()

    def observe_head_output(self, layer: int, start: int, value: Tensor) -> None:
        self.retain(value)
        self.head.setdefault(layer, []).append((start, value))

    def observe_layer_input(self, layer: int, value: Tensor) -> None:
        self.retain(value)
        self.layer_input[layer] = value

    def observe_attention_write(self, layer: int, value: Tensor) -> None:
        self.retain(value)
        self.attention_write[layer] = value

    def observe_mlp_write(self, layer: int, value: Tensor) -> None:
        self.retain(value)
        self.mlp_write[layer] = value

    def layer_gradients(
        self,
        positions: Tensor,
        layer: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return selected gradients from one locally recomputed layer."""

        position = positions.long()
        chunks = sorted(self.head[layer], key=lambda item: item[0])
        chunk_grad = [
            torch.zeros_like(value) if value.grad is None else value.grad
            for _, value in chunks
        ]
        head_grad = torch.cat(chunk_grad, dim=2)[0]
        selected_head = (
            head_grad.index_select(1, position.to(head_grad.device)).float().cpu()
        )
        selected_nodes = []
        for source in (self.layer_input, self.attention_write, self.mlp_write):
            value = source[layer]
            gradient = (torch.zeros_like(value) if value.grad is None else value.grad)[
                0
            ]
            selected_nodes.append(
                gradient.index_select(0, position.to(gradient.device)).float().cpu()
            )
        return selected_head, *selected_nodes

    def gradients(self, positions: Tensor, layers: int) -> TargetGradients:
        position = positions.long()
        head = []
        layer_input = []
        attention = []
        mlp = []
        for layer in range(layers):
            current = self.layer_gradients(position, layer)
            head.append(current[0])
            layer_input.append(current[1])
            attention.append(current[2])
            mlp.append(current[3])
        return TargetGradients(
            position.cpu(),
            torch.stack(head),
            torch.stack(layer_input),
            torch.stack(attention),
            torch.stack(mlp),
        )


def contrast_direction(model, target: TargetContrast) -> tuple[Tensor, Tensor]:
    """Return the fixed positive-minus-negative unembedding direction and bias."""

    device = model.lm_head.weight.device
    index = torch.tensor(
        [target.positive_token_id, target.negative_token_id],
        device=device,
    )
    selected_weight = model.lm_head.weight.detach().index_select(0, index).float()
    direction = selected_weight[0] - selected_weight[1]
    bias = getattr(model.lm_head, "bias", None)
    if bias is None:
        return direction, direction.new_zeros(())
    selected = bias.detach().index_select(0, index).float()
    return direction, selected[0] - selected[1]


def integrated_target_gradients(
    model,
    clean_token_ids: Tensor,
    corrupt_token_ids: Tensor,
    target: TargetContrast,
    positions: Tensor,
    *,
    steps: int = 1,
    query_chunk: int | None = None,
) -> TargetGradients:
    """Average target gradients along the corrupt-to-clean embedding path.

    With ``steps=1`` this is a midpoint attribution-patching screen. More
    steps reduce single-point saturation. Exact corridor claims still come
    from the later cut/patch/block reruns.
    """

    if steps < 1:
        raise ValueError("gradient steps must be positive")
    device = model.get_input_embeddings().weight.device
    clean = clean_token_ids[:-1].to(device)[None]
    corrupt = corrupt_token_ids[:-1].to(device)[None]
    model.eval().requires_grad_(False)
    with torch.no_grad():
        clean_embedding = model.model.embed_tokens(clean)
        corrupt_embedding = model.model.embed_tokens(corrupt)
    direction, bias = contrast_direction(model, target)
    total: TargetGradients | None = None
    for step in range(steps):
        alpha = (step + 0.5) / steps
        hidden = (
            corrupt_embedding + alpha * (clean_embedding - corrupt_embedding)
        ).detach()
        hidden.requires_grad_(True)
        observer = GradientObserver()
        final = forward_layers(
            model,
            hidden,
            0,
            observer=observer,
            attention_query_chunk=query_chunk,
        )
        margin = torch.dot(final[0, target.query_position].float(), direction) + bias
        margin.backward()
        current = observer.gradients(positions, len(model.model.layers))
        if total is None:
            total = current
        else:
            total = TargetGradients(
                total.position,
                total.head_output + current.head_output,
                total.layer_input + current.layer_input,
                total.attention_write + current.attention_write,
                total.mlp_write + current.mlp_write,
            )
        del final, margin, hidden, observer
    assert total is not None
    scale = 1.0 / steps
    return TargetGradients(
        total.position,
        total.head_output * scale,
        total.layer_input * scale,
        total.attention_write * scale,
        total.mlp_write * scale,
    )


def native_target_gradients(
    model,
    cache: ForwardCache,
    target: TargetContrast,
    positions: Tensor,
    *,
    query_chunk: int | None = None,
) -> TargetGradients:
    """Differentiate one fixed target margin with layer-local reverse VJPs.

    These gradients provide target-specific functionality for native messages.
    They are deliberately separate from the attention/message quantity used to
    construct the transport graph.  Cached native layer inputs let the reverse
    sweep retain only one decoder layer's autograd graph at a time.  This
    removes full-depth graph retention; it is not a fixed-VRAM guarantee, and
    one eager-attention layer can still require O(sequence_length**2) memory.
    Exact claims still require later reruns.
    """

    device = model.get_input_embeddings().weight.device
    layers = len(model.model.layers)
    if cache.layer_count != layers or set(cache.layer_input) != set(range(layers)):
        raise ValueError("native gradients require every cached layer input")
    tokens = cache.layer_input[0].shape[0]
    if not 0 <= target.query_position < tokens:
        raise ValueError("target query is outside the causal source positions")
    position = torch.as_tensor(positions, dtype=torch.long).flatten().cpu()
    if not len(position):
        raise ValueError("native target gradients require represented positions")
    if int(position.min()) < 0 or int(position.max()) >= tokens:
        raise ValueError("gradient position is outside the causal prefix")

    target_slot = torch.nonzero(
        cache.query.long() == target.query_position,
        as_tuple=False,
    ).flatten()
    if len(target_slot) != 1:
        raise ValueError("native target query is absent from the baseline cache")
    slot = int(target_slot[0])
    if int(cache.target[slot]) != target.positive_token_id:
        raise ValueError("positive candidate is not the cached observed token")
    if int(cache.runner[slot]) != target.negative_token_id:
        raise ValueError("negative candidate is not the frozen native runner")

    model.eval().requires_grad_(False)
    direction, bias = contrast_direction(model, target)
    cotangent: Tensor | None = None
    head: list[Tensor] = []
    layer_input: list[Tensor] = []
    attention_write: list[Tensor] = []
    mlp_write: list[Tensor] = []
    for layer in range(layers - 1, -1, -1):
        # ``baseline_forward`` stores checkpoints under inference mode.  Clone
        # them into ordinary tensors before enabling autograd for the local VJP.
        hidden = cache.layer_input[layer].to(device).clone()[None].detach()
        hidden.requires_grad_(True)
        observer = GradientObserver()
        output = forward_layers(
            model,
            hidden,
            layer,
            observer=observer,
            attention_query_chunk=query_chunk,
            end_layer=layer + 1,
            apply_final_norm=False,
        )
        if cotangent is None:
            final = model.model.norm(output)
            objective = (
                torch.dot(final[0, target.query_position].float(), direction) + bias
            )
            objective.backward()
            del final, objective
        else:
            output.backward(cotangent.to(device=output.device, dtype=output.dtype))

        current = observer.layer_gradients(position, layer)
        head.append(current[0])
        layer_input.append(current[1])
        attention_write.append(current[2])
        mlp_write.append(current[3])
        if hidden.grad is None:
            raise RuntimeError("native layer VJP did not produce an input cotangent")
        cotangent = hidden.grad.detach()
        del output, hidden, observer

    return TargetGradients(
        position,
        torch.stack(head[::-1]),
        torch.stack(layer_input[::-1]),
        torch.stack(attention_write[::-1]),
        torch.stack(mlp_write[::-1]),
    )
