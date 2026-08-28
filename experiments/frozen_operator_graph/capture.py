"""Exact frozen replay capture for Llama-like causal language models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .basis import q_to_kv_mapping
from .schema import ExactSampleCapture, LayerCapture


def _tensor_output(output: Any, *, name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"{name} did not return a tensor-first output")


def _attention_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError(
            "self-attention must return (attention_output, attention_weights); "
            "load the checkpoint with attn_implementation='eager'"
        )
    message, weights = output[0], output[1]
    if not torch.is_tensor(message) or not torch.is_tensor(weights):
        raise RuntimeError("self-attention output/weights must be tensors")
    return message, weights


class ExactLlamaReplay:
    """Teacher-force a frozen checkpoint and expose its actual layer computation.

    This class intentionally supports only the validated Llama-like pre-norm
    path with ``input_layernorm``, ``self_attn``, ``post_attention_layernorm``
    and ``mlp``.  Unsupported architectures fail instead of silently switching
    to an approximate capture.
    """

    def __init__(self, model: Any, *, checkpoint: str = "<in-memory>") -> None:
        self.model = model
        self.checkpoint = str(checkpoint)
        self.model.eval()
        self.model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("failed to freeze model parameters")
        config = self.model.config
        self.head_count = int(config.num_attention_heads)
        self.kv_head_count = int(
            getattr(config, "num_key_value_heads", self.head_count)
        )
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.head_count)
        )
        if self.head_count * self.head_dim != self.hidden_size:
            raise ValueError("hidden size is incompatible with attention heads")
        self.q_to_kv = q_to_kv_mapping(self.head_count, self.kv_head_count)
        for index, layer in enumerate(self.layers):
            required = (
                "input_layernorm",
                "self_attn",
                "post_attention_layernorm",
                "mlp",
            )
            missing = [name for name in required if not hasattr(layer, name)]
            if missing:
                raise TypeError(
                    f"layer {index} is not the required Llama-like pre-norm layer: "
                    f"missing {missing}"
                )
            attention = layer.self_attn
            if not hasattr(attention, "v_proj") or not hasattr(attention, "o_proj"):
                raise TypeError(f"layer {index} must expose v_proj and o_proj")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        torch_dtype: torch.dtype | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> "ExactLlamaReplay":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Exact replay requires transformers in the experiment environment"
            ) from error
        kwargs: dict[str, Any] = {
            "attn_implementation": "eager",
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if revision is not None:
            kwargs["revision"] = revision
        identifier = str(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
        model.to(device)
        path = Path(identifier).expanduser()
        identity = str(path.resolve()) if path.exists() else identifier
        if revision is not None and not path.exists():
            identity = f"{identity}@{revision}"
        return cls(model, checkpoint=identity)

    @property
    def backbone(self) -> Any:
        backbone = getattr(self.model, "model", None)
        if backbone is None or not hasattr(backbone, "layers"):
            raise TypeError("model must expose .model.layers")
        return backbone

    @property
    def layers(self) -> list[Any]:
        return list(self.backbone.layers)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def capture(
        self,
        token_ids: torch.Tensor,
        response_start: int,
        *,
        conservation_atol: float = 5e-3,
        conservation_rtol: float = 5e-3,
        attention_validator: Callable[[list[torch.Tensor]], dict[str, object] | None]
        | None = None,
    ) -> ExactSampleCapture:
        """Capture exact response attention, values, residuals, and MLP updates."""

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        response_start = int(response_start)
        if not 0 < response_start < int(token_ids.numel()):
            raise ValueError("response_start must split prompt and response")
        tokens = int(token_ids.numel())
        response = tokens - response_start
        layers = self.layers

        names = (
            "residual_input",
            "pre_attention_hidden",
            "attention_output",
            "attention",
            "value_states",
            "o_proj_input",
            "post_attention_residual",
            "pre_mlp_hidden",
            "mlp_output",
            "layer_output",
        )
        captured: dict[str, list[torch.Tensor | None]] = {
            name: [None] * len(layers) for name in names
        }
        handles = []

        def store(name: str, index: int, tensor: torch.Tensor) -> None:
            if captured[name][index] is not None:
                raise RuntimeError(f"{name} hook fired more than once at layer {index}")
            captured[name][index] = tensor.detach().cpu()

        def layer_input_hook(index: int):
            def hook(_module: Any, arguments: tuple[Any, ...]) -> None:
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise RuntimeError("decoder layer input hook expected hidden states")
                store("residual_input", index, arguments[0][0])

            return hook

        def simple_output_hook(name: str, index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                store(name, index, _tensor_output(output, name=name)[0])

            return hook

        def post_attention_input_hook(index: int):
            def hook(_module: Any, arguments: tuple[Any, ...]) -> None:
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise RuntimeError("post-attention norm hook expected hidden states")
                store("post_attention_residual", index, arguments[0][0])

            return hook

        def value_hook(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                if not torch.is_tensor(output):
                    raise RuntimeError("v_proj hook expected a tensor")
                expected = (1, tokens, self.kv_head_count * self.head_dim)
                if tuple(output.shape) != expected:
                    raise RuntimeError(
                        f"layer {index} v_proj shape {tuple(output.shape)} != {expected}"
                    )
                value = output[0].reshape(tokens, self.kv_head_count, self.head_dim)
                store("value_states", index, value)

            return hook

        def o_proj_input_hook(index: int):
            def hook(_module: Any, arguments: tuple[Any, ...]) -> None:
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise RuntimeError("o_proj pre-hook expected concatenated head context")
                expected = (1, tokens, self.head_count * self.head_dim)
                if tuple(arguments[0].shape) != expected:
                    raise RuntimeError(
                        f"layer {index} o_proj input shape {tuple(arguments[0].shape)} "
                        f"!= {expected}"
                    )
                context = arguments[0][0].reshape(
                    tokens, self.head_count, self.head_dim
                )
                store("o_proj_input", index, context)

            return hook

        def attention_hook(index: int):
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

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(layer_input_hook(index)))
            handles.append(
                layer.input_layernorm.register_forward_hook(
                    simple_output_hook("pre_attention_hidden", index)
                )
            )
            handles.append(layer.self_attn.v_proj.register_forward_hook(value_hook(index)))
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    o_proj_input_hook(index)
                )
            )
            handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    post_attention_input_hook(index)
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    simple_output_hook("pre_mlp_hidden", index)
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(simple_output_hook("mlp_output", index))
            )
            handles.append(
                layer.register_forward_hook(simple_output_hook("layer_output", index))
            )

        try:
            attention_mask = torch.ones(
                (1, tokens),
                dtype=torch.long,
                device=self.device,
            )
            with torch.inference_mode():
                output = self.backbone(
                    input_ids=token_ids[None],
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
            final_hidden = getattr(output, "last_hidden_state", None)
            if final_hidden is None:
                final_hidden = output[0]
            final_hidden = final_hidden[0].detach().cpu()
            del output
        finally:
            for handle in handles:
                handle.remove()

        missing = [
            f"{name}[{index}]"
            for name in names
            for index, value in enumerate(captured[name])
            if value is None
        ]
        if missing:
            raise RuntimeError("capture hooks did not fire: " + ", ".join(missing))

        layer_captures = []
        for index in range(len(layers)):
            values = {name: captured[name][index] for name in names}
            assert all(value is not None for value in values.values())
            layer_captures.append(
                LayerCapture(
                    attention=values["attention"],
                    value_states=values["value_states"],
                    o_proj_input=values["o_proj_input"],
                    residual_input=values["residual_input"],
                    pre_attention_hidden=values["pre_attention_hidden"],
                    attention_output=values["attention_output"],
                    post_attention_residual=values["post_attention_residual"],
                    pre_mlp_hidden=values["pre_mlp_hidden"],
                    mlp_output=values["mlp_output"],
                    layer_output=values["layer_output"],
                )
            )

        binding = None
        if attention_validator is not None:
            binding = attention_validator(
                [layer.attention for layer in layer_captures]
            )
        return ExactSampleCapture(
            checkpoint=self.checkpoint,
            token_ids=token_ids.detach().cpu(),
            response_start=response_start,
            final_hidden=final_hidden,
            layers=tuple(layer_captures),
            q_to_kv=self.q_to_kv.clone(),
            head_count=self.head_count,
            kv_head_count=self.kv_head_count,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            attention_cache_binding=binding,
        ).validate(atol=conservation_atol, rtol=conservation_rtol)
